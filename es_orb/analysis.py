"""全期間の記述統計(ヒゲ本数・2時間帯・ブレイク足サイズ・異例カウント)。"""
import itertools

import numpy as np
import pandas as pd

import backtest as BT
from data import OR_MINUTES

TIME_BINS = [0, 120, 240, 360, 390]
TIME_LABELS = ["09:30-11:30", "11:30-13:30", "13:30-15:30", "15:30-16:00"]
ATR_BINS = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf]
ORR_BINS = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, np.inf]


def _agg(g):
    m = BT.metrics(g["pnl_usd"].to_numpy(), g["r_mult"].to_numpy(), g["both_hit"].to_numpy())
    return pd.Series({k: m[k] for k in ("trades", "net_usd", "win_rate", "pf", "exp_r", "max_dd")})


def _bucket_table(tr, col, keys):
    rows = []
    for b, g in tr.groupby(col, observed=True):
        rows.append(dict(keys, bucket=str(b), **_agg(g).to_dict()))
    return rows


def _lbl(bins):
    return [f"{a:g}-{b:g}" if np.isfinite(b) else f"{a:g}+" for a, b in zip(bins[:-1], bins[1:])]


def descriptive(B):
    wick, wick_opp, tbin, atrb, orrb, anoms, both = [], [], [], [], [], [], []
    for orm, st, rr in itertools.product(OR_MINUTES, range(4), BT.RRS):
        keys = dict(or_min=orm, stop=st, rr=rr)
        for mode, mx, mname in ((0, 1, "single"), (3, 99, "both")):
            cfg = dict(or_min=orm, stop=st, rr=rr, mode=mode, max_tr=mx, end_min=390, **BT.NO_FILTER)
            tr, anom = BT.run_config(B, cfg)
            k2 = dict(keys, mode=mname)
            tr["tb"] = pd.cut(tr["entry_min"], TIME_BINS, right=False, labels=TIME_LABELS)
            tbin += _bucket_table(tr, "tb", k2)
            both.append(dict(k2, trades=len(tr), both_hit=int(tr["both_hit"].sum()),
                             stop_exits=int((tr["reason"] == 0).sum())))
            if mode != 0:
                continue
            if rr == BT.RRS[0]:
                # 損切り不成立の見送りはRRに依存しないので1回だけ記録
                anoms.append(dict(or_min=orm, stop=st, signals_skipped=len(anom),
                                  trades=len(tr), pct=len(anom) / max(1, len(anom) + len(tr))))
            tr["wb"] = tr["wick"].clip(upper=3).astype(int).map({0: "0", 1: "1", 2: "2", 3: "3+"})
            tr["wob"] = tr["wick_opp"].clip(upper=3).astype(int).map({0: "0", 1: "1", 2: "2", 3: "3+"})
            wick += _bucket_table(tr, "wb", k2)
            wick_opp += _bucket_table(tr, "wob", k2)
            tr["ab"] = pd.cut(tr["atr_ratio"], ATR_BINS, right=False, labels=_lbl(ATR_BINS))
            tr["ob"] = pd.cut(tr["or_ratio"], ORR_BINS, right=False, labels=_lbl(ORR_BINS))
            atrb += _bucket_table(tr, "ab", k2)
            orrb += _bucket_table(tr, "ob", k2)
    return {k: pd.DataFrame(v) for k, v in dict(
        wick=wick, wick_opp=wick_opp, time=tbin, atr_ratio=atrb, or_ratio=orrb,
        gap_skip=anoms, both_hit=both).items()}
