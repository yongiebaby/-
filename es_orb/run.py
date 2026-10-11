"""実行: python run.py --out ../results data1.csv [data2.csv ...]  (zipも可)"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as BT  # noqa: E402
import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--top-k", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    B = data.load(a.inputs)
    print(f"loaded {len(B['NB'])} days ({time.time() - t:.1f}s)")

    wins = BT.windows(B)
    cfgs = BT.base_grid()
    t = time.time()
    stage1 = BT.grid_eval(B, cfgs, wins, a.procs)
    print(f"stage1 {len(cfgs)} configs ({time.time() - t:.1f}s)")
    stage1.to_csv(os.path.join(a.out, "grid_base_all_windows.csv"), index=False)

    t = time.time()
    wf, stage2 = BT.walk_forward(B, stage1, a.top_k, a.procs)
    print(f"walk-forward ({time.time() - t:.1f}s)")
    stage2.to_csv(os.path.join(a.out, "grid_filters_is_oos.csv.gz"), index=False)
    with open(os.path.join(a.out, "_state.pkl"), "wb") as f:
        pickle.dump(dict(B=B, stage1=stage1, wf=wf), f)


if __name__ == "__main__":
    main()
