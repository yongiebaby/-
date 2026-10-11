"""結果CSVとHTMLレポートの生成: python report.py --out ../results"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis  # noqa: E402
import backtest as BT  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
METRICS = ["trades", "net_usd", "win_rate", "pf", "exp_r", "max_dd"]


def _clean(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        if np.isinf(v):
            return 999.0
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _records(df):
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def cfg_label(c):
    f = []
    if c["atr_lo"] > 0 or c["atr_hi"] < 1e8:
        f.append(f"ATR比 {c['atr_lo']:g}–{'∞' if c['atr_hi'] >= 1e8 else format(c['atr_hi'], 'g')}")
    if c["orr_lo"] > 0 or c["orr_hi"] < 1e8:
        f.append(f"OR比 {c['orr_lo']:g}–{'∞' if c['orr_hi'] >= 1e8 else format(c['orr_hi'], 'g')}")
    if (c["wk_lo"], c["wk_hi"]) != (0, 999):
        f.append(f"ヒゲ {c['wk_lo']}–{'∞' if c['wk_hi'] >= 999 else c['wk_hi']}本")
    return f


def build(out, findings=None):
    st = pickle.load(open(os.path.join(out, "_state.pkl"), "rb"))
    B, stage1, wf = st["B"], st["stage1"], st["wf"]

    desc = analysis.descriptive(B)
    for k, df in desc.items():
        df.to_csv(os.path.join(out, f"analysis_{k}.csv"), index=False)

    # ウォークフォワード
    wf_rows, eq = [], {}
    for r in wf:
        c = r["cfg"]
        row = dict(step=r["step"], is_years=r["is_years"], score=r["score"], variant=r["variant"],
                   or_min=c["or_min"], stop=BT.STOP_NAMES[c["stop"]], rr=c["rr"], mode=BT.MODE_NAMES[c["mode"]],
                   max_tr=c["max_tr"], end_min=c["end_min"], filters=" / ".join(cfg_label(c)) or "なし",
                   n_candidates=r["n_candidates"])
        row.update({f"is_{k}": r["is_m"][k] for k in METRICS + ["both_hit", "gap_skip"]})
        row.update({f"oos_{k}": r["oos_m"][k] for k in METRICS + ["both_hit", "gap_skip"]})
        wf_rows.append(row)
        key = f"{r['score']}|{r['variant']}"
        eq.setdefault(key, []).append(r["oos_trades"])
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(out, "walkforward_selection.csv"), index=False)

    oos_summary, curves = [], {}
    for key, parts in eq.items():
        tr = pd.concat(parts, ignore_index=True).sort_values(["date", "trade_num"])
        tr.assign(variant=key).to_csv(os.path.join(out, f"oos_trades_{key.replace('|', '_')}.csv"), index=False)
        m = BT.metrics(tr["pnl_usd"].to_numpy(), tr["r_mult"].to_numpy(), tr["both_hit"].to_numpy())
        oos_summary.append(dict(variant=key, **{k: m[k] for k in METRICS + ["both_hit"]}))
        daily = tr.groupby("date")["pnl_usd"].sum().cumsum()
        curves[key] = [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in daily.items()]
    pd.DataFrame(oos_summary).to_csv(os.path.join(out, "walkforward_oos_summary.csv"), index=False)

    # ベースグリッド(ウィンドウ別指標)をコンパクトに
    wins = [w for w in BT.windows(B)]
    base = []
    for r in stage1.to_dict("records"):
        item = dict(o=int(r["or_min"]), s=int(r["stop"]), rr=r["rr"], m=int(r["mode"]), x=int(r["max_tr"]),
                    e=int(r["end_min"]), w={})
        for w in wins:
            item["w"][w] = [_clean(r[f"{w}|{k}"]) for k in METRICS + ["both_hit", "gap_skip"]]
        base.append(item)

    d = B["days"]
    ex = B["excluded"]
    payload = dict(
        meta=dict(
            first=str(B["dates"].min().date()), last=str(B["dates"].max().date()), days=int(len(d)),
            per_year={str(k): int(v) for k, v in pd.Series(B["year"]).value_counts().sort_index().items()},
            excluded=int(len(ex)),
            excluded_list=[str(x) for x in ex["date"]],
            partial_days=[f"{r.date} ({r.bars}本)" for r in d[d["bars"] < 78].itertuples()],
            windows=wins,
            min_trades=BT.MIN_TRADES_IS,
        ),
        base=base,
        wf=_records(wf_df),
        oos_summary=_records(pd.DataFrame(oos_summary)),
        curves=curves,
        desc={k: _records(v) for k, v in desc.items()},
    )
    html = open(TEMPLATE, encoding="utf-8").read().replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if findings and os.path.exists(findings):
        html = html.replace("<!--FINDINGS-->", open(findings, encoding="utf-8").read())
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("report written", os.path.join(out, "report.html"), f"{len(html) / 1e6:.1f}MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--findings", help="レポート冒頭に差し込む所見のHTML断片")
    a = ap.parse_args()
    build(a.out, a.findings)
