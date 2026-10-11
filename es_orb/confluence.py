"""特徴量フィルター(ボラティリティ・レジーム・コンフルエンス)の検証とウォークフォワード。

採用条件: ISの各年でトレード数 >= MIN_TRADES_PER_YEAR。満たさない設定は順位付けの対象外。
"""
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats

import backtest as BT
import features as F
from data import OR_MINUTES

MIN_TRADES_PER_YEAR = 100
QS = (0.2, 0.4, 0.6, 0.8)
TOP_BASE = 10
TOP_SINGLES_FOR_PAIRS = 6
WICK_CONDS = [((0, 0), "ヒゲ0本"), ((0, 1), "ヒゲ0–1本"), ((1, 999), "ヒゲ1本以上"), ((0, 2), "ヒゲ0–2本")]

_CTX = {}


# ---------------------------------------------------------------- conditions
def conditions(name, values):
    """ISのシグナル時点の値から条件のリストを作る: (ラベル, lo, hi)。lo <= x <= hi で通過。"""
    kind = F.FEATURES[name][2]
    if kind == "cat":
        return [("= +1", 1, 1), ("≥ 0", 0, 1), ("≤ 0", -1, 0), ("= 0", 0, 0)]
    v = values[np.isfinite(values)]
    if len(v) < 50:
        return []
    q = {k: float(np.quantile(v, k)) for k in QS}
    return [(f"≥ 下位{int(k * 100)}%点 ({q[k]:.3g})", q[k], np.inf) for k in (0.2, 0.4, 0.6)] + \
           [(f"≤ 下位{int(k * 100)}%点 ({q[k]:.3g})", -np.inf, q[k]) for k in (0.4, 0.6, 0.8)]


def mask(get, name, oi, lo, hi):
    out = []
    for sign in (1, -1):
        x = get(name, oi, sign)
        m = (x >= lo) & (x <= hi)
        out.append(np.ascontiguousarray(m | ~np.isfinite(x)))  # 値がない日は通過扱い
    return tuple(out)


def combine(m1, m2):
    return (m1[0] & m2[0], m1[1] & m2[1])


def signal_values(get, name, oi, tr):
    d = tr["day"].to_numpy(int)
    k = tr["sig_bar"].to_numpy(int)
    s = tr["dir"].to_numpy(int)
    xl, xs = get(name, oi, 1), get(name, oi, -1)
    return np.where(s == 1, xl[d, k], xs[d, k])


# ---------------------------------------------------------------- evaluation
def _eval_one(job):
    B, get, wins = _CTX["B"], _CTX["get"], _CTX["wins"]
    cfg, filt = job
    oi = OR_MINUTES.index(cfg["or_min"])
    allow = None
    for f in filt:
        if f[0] == "wick":
            continue
        m = mask(get, f[0], oi, f[2], f[3])
        allow = m if allow is None else combine(allow, m)
    c = dict(cfg)
    for f in filt:
        if f[0] == "wick":
            c["wk_lo"], c["wk_hi"] = f[2], f[3]
    tr, anom = BT.run_config(B, c, allow)
    row = dict(cfg, filters=filt, label=" & ".join(f"{f[0]} {f[1]}" for f in filt) or "なし")
    for name, yrs in wins.items():
        for k, v in BT.window_metrics(tr, anom, yrs).items():
            row[f"{name}|{k}"] = v
    return row


def evaluate(B, get, jobs, wins, procs):
    _CTX.update(B=B, get=get, wins=wins)
    with mp.get_context("fork").Pool(procs) as pool:
        return pd.DataFrame(pool.map(_eval_one, jobs, chunksize=16))


def eligible(df, is_years):
    ok = np.ones(len(df), dtype=bool)
    for y in is_years:
        ok &= df[f"{y}|trades"].to_numpy() >= MIN_TRADES_PER_YEAR
    return ok


def scores(df, win, is_years):
    ok = eligible(df, is_years)
    df = df.copy()
    df["eligible"] = ok
    npm = df.loc[ok, f"{win}|np_mdd"].replace(np.inf, 1e6).rank(pct=True)
    df["score1"] = np.nan
    df["score2"] = np.nan
    df.loc[ok, "score1"] = (npm + df.loc[ok, f"{win}|exp_r"].rank(pct=True)) / 2
    df.loc[ok, "score2"] = (npm + df.loc[ok, f"{win}|win_rate"].rank(pct=True)) / 2
    return df


def _cfg(r):
    c = {k: r[k] for k in BT.BASE_KEYS}
    for k in ("or_min", "stop", "mode", "max_tr", "end_min"):
        c[k] = int(c[k])
    c["rr"] = float(c["rr"])
    c.update(BT.NO_FILTER)
    return c


def walk_forward(B, stage1, get, procs=4, log=print):
    present = set(B["year"])
    results, pools = [], []
    for is_y, oos_y in BT.WF_STEPS:
        if not set(is_y) <= present or oos_y not in present:
            continue
        win = f"IS{is_y[0]}-{is_y[1]}"
        wins = {win: is_y, str(is_y[0]): (is_y[0],), str(is_y[1]): (is_y[1],), f"OOS{oos_y}": (oos_y,)}
        s1 = scores(stage1, win, is_y)
        base = pd.concat([s1.nlargest(TOP_BASE, "score1"), s1.nlargest(TOP_BASE, "score2")]).drop_duplicates(BT.BASE_KEYS)
        log(f"step {oos_y}: eligible base {int(s1['eligible'].sum())}/{len(s1)}, pool {len(base)}")

        # 単独条件(閾値はそのベース設定のISシグナルから決める)
        jobs = []
        for _, r in base.iterrows():
            cfg = _cfg(r)
            tr, _ = BT.run_config(B, cfg)
            tr = tr[tr["year"].isin(is_y)]
            oi = OR_MINUTES.index(cfg["or_min"])
            jobs.append((cfg, ()))
            for name in F.FEATURES:
                vals = signal_values(get, name, oi, tr)
                for lab, lo, hi in conditions(name, vals):
                    jobs.append((cfg, ((name, lab, lo, hi),)))
            for (lo, hi), lab in WICK_CONDS:
                jobs.append((cfg, (("wick", lab, lo, hi),)))
        singles = evaluate(B, get, jobs, wins, procs)
        singles = scores(singles, win, is_y)

        # 2条件の組み合わせ: ベースごとに単独条件の上位を組み合わせる
        jobs = []
        for key, g in singles[singles["filters"].map(len) == 1].groupby(BT.BASE_KEYS):
            cfg = _cfg(g.iloc[0])
            for sc in ("score1", "score2"):
                top = g[g["eligible"]].nlargest(TOP_SINGLES_FOR_PAIRS, sc)["filters"].tolist()
                for a, b in itertools.combinations(top, 2):
                    if a[0][0] != b[0][0]:
                        jobs.append((cfg, tuple(sorted(a + b))))
        jobs = list({(tuple(sorted(c.items())), f): (c, f) for c, f in jobs}.values())
        pairs = evaluate(B, get, jobs, wins, procs) if jobs else pd.DataFrame()
        log(f"step {oos_y}: singles {len(singles)}, pairs {len(pairs)}")

        pool = pd.concat([singles, pairs], ignore_index=True)
        pool = scores(pool, win, is_y)
        pool["step"] = oos_y
        pools.append(pool)
        for sc in ("score1", "score2"):
            for variant, cand in (("base_only", pool[pool["filters"].map(len) == 0]),
                                  ("single", pool[pool["filters"].map(len) <= 1]),
                                  ("pair", pool)):
                cand = cand[cand["eligible"]]
                if cand.empty:
                    continue
                best = cand.loc[cand[sc].idxmax()]
                results.append(dict(step=oos_y, is_years=f"{is_y[0]}-{is_y[1]}", score=sc, variant=variant,
                                    cfg=_cfg(best), filters=best["filters"], label=best["label"],
                                    score_value=best[sc], row=best))
    return results, pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()


def oos_trades(B, get, res):
    cfg = dict(res["cfg"])
    oi = OR_MINUTES.index(cfg["or_min"])
    allow = None
    for f in res["filters"]:
        if f[0] == "wick":
            cfg["wk_lo"], cfg["wk_hi"] = f[2], f[3]
            continue
        m = mask(get, f[0], oi, f[2], f[3])
        allow = m if allow is None else combine(allow, m)
    tr, anom = BT.run_config(B, cfg, allow)
    return tr, anom


# ---------------------------------------------------------------- univariate stats
REF_SETS = [(orm, st, rr) for orm in OR_MINUTES for st in (1, 3) for rr in (1.0, 2.0)]


def feature_stats(B, get):
    """各特徴量とトレード結果(R)の関係。1日1回・受付16:00・フィルターなしの基準設定で集計。

    - 上位20%(カテゴリは+1) − 下位20%(カテゴリは−1) の期待値の差と Welch のt値(全期間)
    - 同じ差を年ごとに計算し、全期間と同じ符号だった年数(一貫性)
    - 全期間の5分位ごとの期待値
    スピアマン相関は損切り幅とコスト比の関係を拾ってしまうため使わない。
    """
    summary, quint = [], []
    years = sorted(set(B["year"]))
    for orm, st, rr in REF_SETS:
        cfg = dict(or_min=orm, stop=st, rr=rr, mode=0, max_tr=1, end_min=390, **BT.NO_FILTER)
        tr, _ = BT.run_config(B, cfg)
        oi = OR_MINUTES.index(orm)
        R = tr["r_mult"].to_numpy()
        yr = tr["year"].to_numpy()
        for name, (label, group, kind, _, _) in F.FEATURES.items():
            x = signal_values(get, name, oi, tr)
            ok = np.isfinite(x)
            if kind == "cat":
                bins = pd.Series(x[ok]).map({-1.0: "−1", 0.0: "0", 1.0: "+1"})
                hi_m, lo_m = ok & (x == 1), ok & (x == -1)
            else:
                bins = pd.qcut(pd.Series(x[ok]), 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"], duplicates="drop")
                q20, q80 = np.quantile(x[ok], [0.2, 0.8])
                hi_m, lo_m = ok & (x >= q80), ok & (x <= q20)
            for b, g in pd.Series(R[ok]).groupby(bins.to_numpy(), observed=True):
                quint.append(dict(or_min=orm, stop=st, rr=rr, feature=name, bucket=str(b), trades=len(g),
                                  exp_r=g.mean(), win_rate=(g > 0).mean()))
            hi, lo = R[hi_m], R[lo_m]
            diff = hi.mean() - lo.mean()
            t = stats.ttest_ind(hi, lo, equal_var=False).statistic if len(hi) > 5 and len(lo) > 5 else np.nan
            ydiff = []
            for y in years:
                h, l_ = R[hi_m & (yr == y)], R[lo_m & (yr == y)]
                ydiff.append(h.mean() - l_.mean() if len(h) > 10 and len(l_) > 10 else np.nan)
            ydiff = np.array(ydiff)
            summary.append(dict(or_min=orm, stop=st, rr=rr, feature=name, label=label, group=group,
                                n_hi=len(hi), n_lo=len(lo), exp_hi=hi.mean(), exp_lo=lo.mean(),
                                top_minus_bottom=diff, t_stat=t,
                                years_same_sign=int(np.sum(np.sign(ydiff) == np.sign(diff))),
                                years=int(np.isfinite(ydiff).sum()),
                                **{f"diff_{y}": v for y, v in zip(years, ydiff)}))
    return pd.DataFrame(summary), pd.DataFrame(quint)
