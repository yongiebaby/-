"""ボラティリティ・レジーム・コンフルエンスの特徴量。

すべてブレイク足の終値時点、またはそれ以前に確定している情報だけで計算する。
方向を持つ特徴量(aligned)は、ブレイク方向に有利なほど大きい値になるよう符号を合わせる
(ロングは +, ショートは -)。値は (D, 78) の配列で、シグナル足 k の位置に入る。
"""
import numpy as np
import pandas as pd

from data import OR_MINUTES, TICK

ATR_DAYS = 14
TREND_DAYS = 20
RVOL_DAYS = 20
REGIME_DAYS = 252
VALUE_AREA = 0.70

# name: (表示名, グループ, 種類, ORごとに異なるか, 方向に合わせて符号反転するか)
FEATURES = {
    "atr_regime": ("日足ATR14の1年パーセンタイル", "volatility", "cont", False, False),
    "or_atr": ("OR幅 ÷ 日足ATR14", "volatility", "cont", True, False),
    "on_range_atr": ("オーバーナイト値幅 ÷ 日足ATR14", "volatility", "cont", False, False),
    "pd_range_atr": ("前日RTH値幅 ÷ 日足ATR14", "volatility", "cont", False, False),
    "bar_atr": ("ブレイク足値幅 ÷ 5分ATR20", "volatility", "cont", False, False),
    "bar_or": ("ブレイク足値幅 ÷ OR幅", "volatility", "cont", True, False),
    "trend20": ("始値 − 日足SMA20（÷ATR、順方向+）", "regime", "cont", False, True),
    "gap": ("ギャップ（始値 − 前日終値、÷ATR、順方向+）", "regime", "cont", False, True),
    "on_return": ("オーバーナイトの値動き（÷ATR、順方向+）", "regime", "cont", False, True),
    "open_vs_pdva": ("始値と前日RTHバリューエリアの位置（順方向+1／内側0／逆−1）", "profile", "cat", False, True),
    "open_vs_onva": ("始値とオーバーナイトのバリューエリアの位置（順方向+1／内側0／逆−1）", "profile", "cat", False, True),
    "px_vs_onpoc": ("ブレイク足終値 − オーバーナイトPOC（÷ATR、順方向+）", "profile", "cont", False, True),
    "room_pd": ("前日高値/安値までの余地（÷ATR、越えていれば負）", "level", "cont", False, True),
    "room_on": ("オーバーナイト高値/安値までの余地（÷ATR、越えていれば負）", "level", "cont", False, True),
    "vwap_dist": ("終値 − RTH VWAP（÷ATR、順方向+）", "confluence", "cont", False, True),
    "ema_trend": ("5分足EMA20 − EMA50（÷ATR、順方向+）", "confluence", "cont", False, True),
    "rvol_or": ("OR出来高 ÷ 過去20日平均", "volume", "cont", True, False),
    "rvol_bar": ("ブレイク足出来高 ÷ 同時刻の過去20日平均", "volume", "cont", False, False),
    "delta_ratio": ("ブレイク足のデルタ ÷ 出来高（順方向+）", "volume", "cont", False, True),
}


def _profile(h, l, v):
    """5分足の出来高を高安の範囲に均等配分したtick単位のプロファイルから POC/VAH/VAL を求める。"""
    ok = v > 0
    h, l, v = h[ok], l[ok], v[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    lo = np.round(l.min() / TICK).astype(np.int64)
    hi = np.round(h.max() / TICK).astype(np.int64)
    a = np.round(l / TICK).astype(np.int64) - lo
    b = np.round(h / TICK).astype(np.int64) - lo
    w = v / (b - a + 1)
    # 区間加算(差分配列)
    diff = np.zeros(hi - lo + 2)
    np.add.at(diff, a, w)
    np.add.at(diff, b + 1, -w)
    prof = np.cumsum(diff)[:-1]
    poc = int(np.argmax(prof))
    total = prof.sum()
    i = j = poc
    acc = prof[poc]
    while acc < VALUE_AREA * total and (i > 0 or j < len(prof) - 1):
        up = prof[j + 1] if j < len(prof) - 1 else -1
        dn = prof[i - 1] if i > 0 else -1
        if up >= dn:
            j += 1
            acc += up
        else:
            i -= 1
            acc += dn
    return (lo + poc) * TICK, (lo + j) * TICK, (lo + i) * TICK


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def compute(B):
    df = B["df"]
    O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    V = df["Volume"].to_numpy(float)
    DL = df["Delta"].to_numpy(float)
    et = df["et"]
    tmin = (et.dt.hour * 60 + et.dt.minute).to_numpy()

    # --- 全RTHセッション(除外日含む)の日足情報 ---
    sess = B["all_sessions"]
    S = len(sess)
    sH, sL, sC, sO = (np.full(S, np.nan) for _ in range(4))
    pva = np.full((S, 3), np.nan)  # そのRTHセッションの POC, VAH, VAL
    ona = np.full((S, 5), np.nan)  # そのセッション前のオーバーナイト: H, L, POC, VAH, VAL
    on_first = np.full(S, np.nan)
    prev_last = -1
    for si, (d, ii) in enumerate(sess):
        sH[si], sL[si], sC[si], sO[si] = H[ii].max(), L[ii].min(), C[ii[-1]], O[ii[0]]
        pva[si] = _profile(H[ii], L[ii], V[ii])
        if prev_last >= 0:
            on = np.arange(prev_last + 1, ii[0])
            on = on[~((tmin[on] >= 16 * 60) & (tmin[on] < 18 * 60))]  # 引け後〜再開までを除く
            if len(on):
                ona[si, 0], ona[si, 1] = H[on].max(), L[on].min()
                ona[si, 2:] = _profile(H[on], L[on], V[on])
                on_first[si] = O[on[0]]
        prev_last = ii[-1]

    prevC = np.concatenate(([np.nan], sC[:-1]))
    tr = np.maximum(sH - sL, np.maximum(np.abs(sH - prevC), np.abs(sL - prevC)))
    atr14 = pd.Series(tr).rolling(ATR_DAYS).mean().shift(1).to_numpy()  # 前日までのATR
    atr_pct = pd.Series(atr14).rolling(REGIME_DAYS, min_periods=60).rank(pct=True).to_numpy()
    sma20 = pd.Series(sC).rolling(TREND_DAYS).mean().shift(1).to_numpy()

    pos = {d: si for si, (d, _) in enumerate(sess)}
    D = len(B["NB"])
    si_of = np.array([pos[d.date()] for d in B["dates"]])
    p = si_of - 1  # 前日セッション

    day = {}
    a = atr14[si_of]
    day["atr"] = a
    day["atr_regime"] = atr_pct[si_of]
    day["on_range_atr"] = (ona[si_of, 0] - ona[si_of, 1]) / a
    day["pd_range_atr"] = (sH[p] - sL[p]) / a
    rth_open = sO[si_of]
    day["trend20"] = (rth_open - sma20[si_of]) / a
    day["gap"] = (rth_open - sC[p]) / a
    day["on_return"] = (rth_open - on_first[si_of]) / a

    def va_loc(x, vah, val):
        return np.where(x > vah, 1.0, np.where(x < val, -1.0, 0.0))

    day["open_vs_pdva"] = va_loc(rth_open, pva[p, 1], pva[p, 2])
    day["open_vs_onva"] = va_loc(rth_open, ona[si_of, 3], ona[si_of, 4])

    # --- バー単位 ---
    G = B["GIDX"]
    valid = G >= 0
    g = np.where(valid, G, 0)
    shape = G.shape
    cl = np.where(valid, C[g], np.nan)
    hh = np.where(valid, H[g], np.nan)
    ll = np.where(valid, L[g], np.nan)
    vv = np.where(valid, V[g], np.nan)
    dd = np.where(valid, DL[g], np.nan)
    tp = (hh + ll + cl) / 3
    vwap = np.nancumsum(tp * vv, axis=1) / np.nancumsum(vv, axis=1)
    e20, e50 = _ema(C, 20), _ema(C, 50)
    ema_diff = np.where(valid, e20[g] - e50[g], np.nan)

    A = a[:, None]
    bar = {}
    bar["vwap_dist"] = (cl - vwap) / A
    bar["ema_trend"] = ema_diff / A
    bar["px_vs_onpoc"] = (cl - ona[si_of, 2][:, None]) / A
    bar["delta_ratio"] = dd / vv
    bar["bar_atr"] = (hh - ll) / B["ATR"]
    # 同じ時刻の足の過去20日平均出来高
    slot = np.clip(B["MIN"] // 5, 0, shape[1] - 1)
    vslot = np.full(shape, np.nan)
    for di in range(D):
        n = B["NB"][di]
        vslot[di, slot[di, :n]] = vv[di, :n]
    vavg = pd.DataFrame(vslot).rolling(RVOL_DAYS, min_periods=10).mean().shift(1).to_numpy()
    bar["rvol_bar"] = vv / np.take_along_axis(vavg, slot, axis=1)

    # 方向別: 余地 (ロング: 前日高値 − 終値, ショート: 終値 − 前日安値)
    roomL = {"room_pd": (sH[p][:, None] - cl) / A, "room_on": (ona[si_of, 0][:, None] - cl) / A}
    roomS = {"room_pd": (cl - sL[p][:, None]) / A, "room_on": (cl - ona[si_of, 1][:, None]) / A}

    # ORごと
    per_or = {"or_atr": [], "bar_or": [], "rvol_or": []}
    for oi, m in enumerate(OR_MINUTES):
        w = B["ORH"][oi] - B["ORL"][oi]
        per_or["or_atr"].append(np.broadcast_to((w / a)[:, None], shape))
        per_or["bar_or"].append((hh - ll) / w[:, None])
        orv = np.nansum(np.where(B["MIN"] < m, vv, 0), axis=1)
        avg = pd.Series(orv).rolling(RVOL_DAYS, min_periods=10).mean().shift(1).to_numpy()
        per_or["rvol_or"].append(np.broadcast_to((orv / avg)[:, None], shape))

    def get(name, oi, sign):
        """(D,78) の特徴量。sign=+1 ロング, -1 ショート。"""
        _, _, _, por, aligned = FEATURES[name]
        if name in roomL:
            return (roomL if sign == 1 else roomS)[name]
        if por:
            x = per_or[name][oi]
        elif name in day:
            x = np.broadcast_to(day[name][:, None], shape)
        else:
            x = bar[name]
        return x * sign if aligned else x

    return get, dict(day=day, dow=B["dates"].dayofweek.to_numpy())
