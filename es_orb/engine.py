"""ORBバックテストのコア(numba)。

ルール概要
- ORはRTH開始からN分(5/15/30/60)の高安。
- シグナル: OR完成後の足の終値がORH超(ロング)/ORL未満(ショート)。ヒゲだけ抜けた足は見送り。
- エントリー: シグナル足の次の足の始値で成行(+1tickスリッページ)。
- 損切り: 0=ブレイク足のlow/high, 1=ブレイク足の前3本のlow/high, 2=OR中値, 3=逆側のOR。
  いずれも1tick外側にバッファ。約定は損切り価格から1tickスリッページ。
- 利確: エントリー価格からRR×リスク。指値なのでスリッページなし。
- 同一足で損切りと利確の両方に届いたら損切り扱い(回数を記録)。
- 引けまで残ったら最終足の終値で成行決済(1tickスリッページ)。
- 手数料: 往復$4。
- 再エントリー(b): 一度シグナルが出た側は、終値がORの内側に戻るまで再度シグナルを出さない。
  ポジション保有中・フィルター不通過・回数上限などで見送ったブレイクでもその側は無効化される。
- mode: 0=1日1トレード, 1=同方向のみ再エントリー, 2=逆方向のみ, 3=両方可。
"""
import numpy as np
from numba import njit

TICK = 0.25
POINT_VALUE = 50.0
COMMISSION_PTS = 4.0 / POINT_VALUE

# トレード記録の列
T_DAY, T_DIR, T_SIG, T_EMIN, T_ENTRY, T_STOP, T_TGT, T_EXIT, T_XMIN, T_REASON, T_BOTH, \
    T_WICK, T_WICKOPP, T_ATRR, T_ORR, T_NUM, T_RISK, T_PNL = range(18)
NCOL = 18
COLS = ["day", "dir", "sig_bar", "entry_min", "entry", "stop", "target", "exit", "exit_min",
        "reason", "both_hit", "wick", "wick_opp", "atr_ratio", "or_ratio", "trade_num", "risk", "pnl_pts"]
REASONS = {0: "stop", 1: "target", 2: "eod", 3: "gap_stop"}


@njit(cache=True)
def simulate(O, H, L, C, P3L, P3H, ATR, MIN, NB, orh_arr, orl_arr, or_min,
             stop_type, rr, mode, max_tr, end_min,
             atr_lo, atr_hi, orr_lo, orr_hi, wk_lo, wk_hi, out, anomalies):
    """1設定を全日で回す。out にトレード、anomalies に損切り不成立(ギャップ)の見送りを書く。
    戻り値: (トレード数, 見送り数)"""
    D = O.shape[0]
    nt = 0
    na = 0
    for d in range(D):
        nb = NB[d]
        orh = orh_arr[d]
        orl = orl_arr[d]
        if nb == 0 or np.isnan(orh):
            continue
        width = orh - orl
        mid = (orh + orl) / 2.0
        armL = True
        armS = True
        wL = 0
        wS = 0
        ntr = 0
        first_dir = 0
        last_dir = 0
        skip_wick = False
        k = 0
        while k < nb and MIN[d, k] < or_min:
            k += 1
        while k < nb:
            c = C[d, k]
            # ヒゲ抜け(終値はORの内側)のカウント
            if not skip_wick:
                if armL and H[d, k] > orh and c <= orh:
                    wL += 1
                if armS and L[d, k] < orl and c >= orl:
                    wS += 1
            skip_wick = False
            sig = 0
            if c > orh:
                if armL:
                    sig = 1
                armL = False
            elif c < orl:
                if armS:
                    sig = -1
                armS = False
            else:
                if not armL:
                    armL = True
                    wL = 0
                if not armS:
                    armS = True
                    wS = 0
            if sig == 0:
                k += 1
                continue

            ok = k + 1 < nb and MIN[d, k + 1] < end_min and ntr < max_tr
            if ok and ntr > 0:
                if mode == 0:
                    ok = False
                elif mode == 1:
                    ok = sig == first_dir
                elif mode == 2:
                    ok = sig == -last_dir
            rng = H[d, k] - L[d, k]
            a = ATR[d, k]
            ar = rng / a if a > 0 else np.nan
            orr = rng / width if width > 0 else np.nan
            wk = wL if sig == 1 else wS
            wko = wS if sig == 1 else wL
            if ok and (atr_lo > 0 or atr_hi < 1e8):
                ok = (not np.isnan(ar)) and ar >= atr_lo and ar < atr_hi
            if ok and (orr_lo > 0 or orr_hi < 1e8):
                ok = (not np.isnan(orr)) and orr >= orr_lo and orr < orr_hi
            if ok:
                ok = wk >= wk_lo and wk <= wk_hi
            if not ok:
                k += 1
                continue

            j = k + 1
            entry = O[d, j] + sig * TICK
            if stop_type == 0:
                lvl = L[d, k] if sig == 1 else H[d, k]
            elif stop_type == 1:
                lvl = P3L[d, k] if sig == 1 else P3H[d, k]
            elif stop_type == 2:
                lvl = mid
            else:
                lvl = orl if sig == 1 else orh
            stop = lvl - sig * TICK
            risk = sig * (entry - stop)
            if not (risk > 0):
                # 次足の始値が損切りラインを越えていて損切りが成立しない -> 見送り
                anomalies[na, 0] = d
                anomalies[na, 1] = sig
                anomalies[na, 2] = MIN[d, j]
                anomalies[na, 3] = ntr
                na += 1
                k += 1
                continue
            tgt = entry + sig * rr * risk

            reason = 2
            both = 0
            xp = C[d, nb - 1] - sig * TICK
            x = nb - 1
            for jj in range(j, nb):
                if jj > j:
                    if sig * (O[d, jj] - stop) <= 0:
                        xp = O[d, jj] - sig * TICK
                        reason = 3
                        x = jj
                        break
                    if sig * (O[d, jj] - tgt) >= 0:
                        xp = tgt
                        reason = 1
                        x = jj
                        break
                if sig == 1:
                    hs = L[d, jj] <= stop
                    ht = H[d, jj] >= tgt
                else:
                    hs = H[d, jj] >= stop
                    ht = L[d, jj] <= tgt
                if hs:
                    xp = stop - sig * TICK
                    reason = 0
                    both = 1 if ht else 0
                    x = jj
                    break
                if ht:
                    xp = tgt
                    reason = 1
                    x = jj
                    break

            r = out[nt]
            r[0] = d
            r[1] = sig
            r[2] = k
            r[3] = MIN[d, j]
            r[4] = entry
            r[5] = stop
            r[6] = tgt
            r[7] = xp
            r[8] = MIN[d, x]
            r[9] = reason
            r[10] = both
            r[11] = wk
            r[12] = wko
            r[13] = ar
            r[14] = orr
            r[15] = ntr
            r[16] = risk
            r[17] = sig * (xp - entry) - COMMISSION_PTS
            nt += 1
            ntr += 1
            if first_dir == 0:
                first_dir = sig
            last_dir = sig

            # 保有中(エントリー足〜決済足の1本前)のブレイクはその側を無効化。保有中は再アームもしない
            for b in range(j, x):
                if C[d, b] > orh:
                    armL = False
                elif C[d, b] < orl:
                    armS = False
            if reason == 2:
                break
            # 決済足の終値は決済後に確定するので、次のループで通常どおり評価する(ヒゲはカウントしない)
            k = x
            skip_wick = True
            continue
    return nt, na


