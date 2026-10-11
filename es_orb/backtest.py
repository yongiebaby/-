"""グリッド実行・指標計算・ウォークフォワード最適化。"""
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd

import engine as E
from data import OR_MINUTES

STOP_NAMES = {0: "break_bar", 1: "prev3", 2: "or_mid", 3: "or_opposite"}
MODE_NAMES = {0: "single", 1: "same_dir", 2: "opp_dir", 3: "both"}
RRS = (0.2, 0.5, 1.0, 1.5, 2.0)
MODE_MAX = [(0, 1)] + [(m, x) for m in (1, 2, 3) for x in (2, 3, 99)]
END_MINS = (120, 240, 360, 390)  # 11:30 / 13:30 / 15:30 / 16:00 ET までにエントリー
INF = 1e9

ATR_LO = (0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
ATR_HI = (INF, 1.5, 2.0, 3.0)
ORR_LO = (0, 0.1, 0.2, 0.35, 0.5)
ORR_HI = (INF, 0.5, 0.75, 1.0, 1.5)
WICK = ((0, 999), (0, 0), (0, 1), (1, 999), (2, 999))

WF_STEPS = [((2020, 2021), 2022), ((2021, 2022), 2023), ((2022, 2023), 2024), ((2023, 2024), 2025)]
MIN_TRADES_IS = 30

BASE_KEYS = ["or_min", "stop", "rr", "mode", "max_tr", "end_min"]
FILT_KEYS = ["atr_lo", "atr_hi", "orr_lo", "orr_hi", "wk_lo", "wk_hi"]
NO_FILTER = dict(atr_lo=0, atr_hi=INF, orr_lo=0, orr_hi=INF, wk_lo=0, wk_hi=999)

_B = None


def _init(B):
    global _B
    _B = B


def run_config(B, cfg):
    """1設定を回し、トレードのDataFrameと損切り不成立の見送り件数を返す。"""
    D = len(B["NB"])
    out = np.zeros((D * 30, E.NCOL))
    an = np.zeros((D * 30, 4))
    oi = OR_MINUTES.index(cfg["or_min"])
    n, na = E.simulate(B["O"], B["H"], B["L"], B["C"], B["P3L"], B["P3H"], B["ATR"], B["MIN"], B["NB"],
                       B["ORH"][oi], B["ORL"][oi], cfg["or_min"],
                       cfg["stop"], cfg["rr"], cfg["mode"], cfg["max_tr"], cfg["end_min"],
                       cfg["atr_lo"], cfg["atr_hi"], cfg["orr_lo"], cfg["orr_hi"], cfg["wk_lo"], cfg["wk_hi"],
                       out, an)
    tr = pd.DataFrame(out[:n], columns=E.COLS)
    tr["year"] = B["year"][tr["day"].astype(int)]
    tr["date"] = B["dates"][tr["day"].astype(int)]
    tr["r_mult"] = tr["pnl_pts"] / tr["risk"]
    tr["pnl_usd"] = tr["pnl_pts"] * E.POINT_VALUE
    anom = pd.DataFrame(an[:na], columns=["day", "dir", "entry_min", "trade_num"])
    anom["year"] = B["year"][anom["day"].astype(int)]
    return tr, anom


def metrics(pnl_usd, r_mult, both=None, anom=0):
    n = len(pnl_usd)
    if n == 0:
        return dict(trades=0, net_usd=0.0, win_rate=np.nan, pf=np.nan, exp_r=np.nan, max_dd=0.0,
                    np_mdd=np.nan, both_hit=0, gap_skip=anom)
    eq = np.cumsum(pnl_usd)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    mdd = float(np.max(peak - eq))
    gw = pnl_usd[pnl_usd > 0].sum()
    gl = -pnl_usd[pnl_usd < 0].sum()
    net = float(eq[-1])
    return dict(trades=n, net_usd=net, win_rate=float((pnl_usd > 0).mean()),
                pf=float(gw / gl) if gl > 0 else np.inf, exp_r=float(r_mult.mean()), max_dd=mdd,
                np_mdd=net / mdd if mdd > 0 else (np.inf if net > 0 else 0.0),
                both_hit=int(both.sum()) if both is not None else 0, gap_skip=anom)


def window_metrics(tr, anom, years):
    m = tr["year"].isin(years).to_numpy()
    a = int(anom["year"].isin(years).sum())
    return metrics(tr["pnl_usd"].to_numpy()[m], tr["r_mult"].to_numpy()[m], tr["both_hit"].to_numpy()[m], a)


def windows(B):
    yrs = sorted(set(B["year"]))
    w = {"all": tuple(yrs)}
    w.update({str(y): (y,) for y in yrs})
    for is_y, oos_y in WF_STEPS:
        w[f"IS{is_y[0]}-{is_y[1]}"] = is_y
    return w


def _eval(cfg):
    tr, anom = run_config(_B, cfg)
    row = dict(cfg)
    for name, yrs in _WINDOWS.items():
        for k, v in window_metrics(tr, anom, yrs).items():
            row[f"{name}|{k}"] = v
    return row


_WINDOWS = {}


def grid_eval(B, cfgs, wins, procs=4):
    global _WINDOWS
    _WINDOWS = wins
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_init, initargs=(B,)) as pool:
        rows = pool.map(_eval, cfgs, chunksize=64)
    return pd.DataFrame(rows)


def base_grid():
    cfgs = []
    for orm, st, rr, (mode, mx), end in itertools.product(OR_MINUTES, range(4), RRS, MODE_MAX, END_MINS):
        cfgs.append(dict(or_min=orm, stop=st, rr=rr, mode=mode, max_tr=mx, end_min=end, **NO_FILTER))
    return cfgs


def filter_grid():
    out = []
    for al, ah in itertools.product(ATR_LO, ATR_HI):
        if al >= ah:
            continue
        for ol, oh in itertools.product(ORR_LO, ORR_HI):
            if ol >= oh:
                continue
            for wl, wh in WICK:
                out.append(dict(atr_lo=al, atr_hi=ah, orr_lo=ol, orr_hi=oh, wk_lo=wl, wk_hi=wh))
    return out


def add_scores(df, win, min_trades=MIN_TRADES_IS):
    """IS窓での総合スコア(パーセンタイル順位の平均)。トレード数不足は対象外。"""
    ok = df[f"{win}|trades"] >= min_trades
    s = pd.Series(np.nan, index=df.index)
    npm = df.loc[ok, f"{win}|np_mdd"].replace(np.inf, 1e6)
    r_np = npm.rank(pct=True)
    r_exp = df.loc[ok, f"{win}|exp_r"].rank(pct=True)
    r_win = df.loc[ok, f"{win}|win_rate"].rank(pct=True)
    df["score1"] = s.copy()
    df["score2"] = s.copy()
    df.loc[ok, "score1"] = (r_np + r_exp) / 2  # ネット損益/最大DD × 期待値
    df.loc[ok, "score2"] = (r_np + r_win) / 2  # ネット損益/最大DD × 勝率
    return df


def walk_forward(B, stage1, top_k=10, procs=4, log=print):
    """各ステップ: (1) ベース設定3200通りをISで評価 -> score1/score2の上位k件ずつ
    (2) その各設定にブレイク足フィルター・ヒゲ本数フィルターのグリッドを掛けてISで評価
    (3) 候補全体(フィルターなし+あり)で再順位付けし、最良をOOSに適用。"""
    fgrid = filter_grid()
    results, stage2_all = [], []
    present = set(B["year"])
    for is_y, oos_y in WF_STEPS:
        if not set(is_y) <= present or oos_y not in present:
            log(f"skip step {is_y}->{oos_y}: data missing")
            continue
        win = f"IS{is_y[0]}-{is_y[1]}"
        s1 = add_scores(stage1.copy(), win)
        pool = pd.concat([s1.nlargest(top_k, "score1"), s1.nlargest(top_k, "score2")]).drop_duplicates(BASE_KEYS)
        cfgs = [dict({k: r[k] for k in BASE_KEYS}, **f) for _, r in pool.iterrows() for f in fgrid]
        for c in cfgs:
            c["or_min"], c["stop"], c["mode"], c["max_tr"], c["end_min"] = (
                int(c["or_min"]), int(c["stop"]), int(c["mode"]), int(c["max_tr"]), int(c["end_min"]))
        log(f"step {is_y}->{oos_y}: stage2 {len(cfgs)} configs")
        s2 = grid_eval(B, cfgs, {win: is_y, f"OOS{oos_y}": (oos_y,)}, procs)
        s2["step"] = oos_y
        stage2_all.append(s2)
        cand = pd.concat([s1[BASE_KEYS + FILT_KEYS + [c for c in s1.columns if c.startswith(win + "|")]],
                          s2.drop(columns=[c for c in s2.columns if c.startswith("OOS")] + ["step"])],
                         ignore_index=True).drop_duplicates(BASE_KEYS + FILT_KEYS)
        cand = add_scores(cand, win)
        for sc in ("score1", "score2"):
            for variant, pool_df in (("filtered", cand), ("base_only", s1)):
                best = pool_df.loc[pool_df[sc].idxmax()]
                cfg = {k: best[k] for k in BASE_KEYS + FILT_KEYS}
                for k in ("or_min", "stop", "mode", "max_tr", "end_min"):
                    cfg[k] = int(cfg[k])
                tr, anom = run_config(B, cfg)
                is_m = window_metrics(tr, anom, is_y)
                oos_m = window_metrics(tr, anom, (oos_y,))
                oos_tr = tr[tr["year"] == oos_y].copy()
                results.append(dict(step=oos_y, is_years=f"{is_y[0]}-{is_y[1]}", score=sc, variant=variant,
                                    cfg=cfg, score_value=best[sc], is_m=is_m, oos_m=oos_m, oos_trades=oos_tr,
                                    n_candidates=int(pool_df[sc].notna().sum())))
    return results, (pd.concat(stage2_all, ignore_index=True) if stage2_all else pd.DataFrame())
