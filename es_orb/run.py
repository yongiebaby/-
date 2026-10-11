"""実行: python run.py --out ../results data1.csv [data2.csv ...]  (zipも可)"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as BT  # noqa: E402
import confluence as CF  # noqa: E402
import data  # noqa: E402
import features as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    B = data.load(a.inputs)
    get, _ = F.compute(B)
    print(f"loaded {len(B['NB'])} days + features ({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    stage1 = BT.grid_eval(B, BT.base_grid(), BT.windows(B), a.procs)
    print(f"stage1 {len(stage1)} configs ({time.time() - t:.1f}s)", flush=True)
    stage1.to_csv(os.path.join(a.out, "grid_base_all_windows.csv"), index=False)

    t = time.time()
    fstats, fquint = CF.feature_stats(B, get)
    fstats.to_csv(os.path.join(a.out, "feature_stats.csv"), index=False)
    fquint.to_csv(os.path.join(a.out, "feature_quintiles.csv"), index=False)
    print(f"feature stats ({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    wf, pools = CF.walk_forward(B, stage1, get, a.procs, log=lambda m: print(m, flush=True))
    print(f"walk-forward ({time.time() - t:.1f}s)", flush=True)
    keep = [c for c in pools.columns if c != "filters"]
    pools[keep].to_csv(os.path.join(a.out, "wf_candidates.csv.gz"), index=False)
    for r in wf:
        tr, anom = CF.oos_trades(B, get, r)
        r["trades"] = tr
        r["anom"] = anom
    B_small = {k: v for k, v in B.items() if k not in ("df", "all_sessions")}
    with open(os.path.join(a.out, "_state.pkl"), "wb") as f:
        pickle.dump(dict(B=B_small, stage1=stage1, wf=wf, fstats=fstats, fquint=fquint), f)


if __name__ == "__main__":
    main()
