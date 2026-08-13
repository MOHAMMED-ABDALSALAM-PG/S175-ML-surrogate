"""Quantify the effect of the physical-bounds clamp on the reported metrics.

The evaluation clamps predictions to each output's physically admissible
range (non-negativity for speed/power/torque/lat_acc/roll/fuel, [0, 100] for
the four probability/percentage outputs) before scoring. This script reports,
for a given run, how many predictions the clamp actually changes and what the
per-output R2 is with and without it, so the paper can state that the
clamp is a physical-consistency projection and not a source of accuracy.

  python scripts/13_clamp_report.py --splits S1_random S4_corner

Output: results/evaluation/clamp_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s175 import data as sdata
from s175.columns import OUTPUT_COLS, CLASS_VALID, CLASS_ALL_NEG, PHYSICAL_BOUNDS
from s175.metrics import clamp_physical, per_output
from s175.models import MLP
from s175.splits import load_split

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="v3_masked_peroutput")
    ap.add_argument("--splits", nargs="+", default=["S1_random", "S4_corner"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250_000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    log(f"loaded {len(X):,} rows | device {device}")

    report = {}
    for split_name in args.splits:
        run_dir = (ROOT / "runs" / args.config_name / split_name /
                   f"seed_{args.seed}")
        ck = torch.load(run_dir / "regressor.pt", map_location="cpu",
                        weights_only=True)
        reg = MLP(**ck["model_config"])
        reg.load_state_dict(ck["model"])
        reg.to(device).eval()
        xm = np.asarray(ck["x_scaler"]["mean"], dtype=np.float32)
        xs = np.asarray(ck["x_scaler"]["std"], dtype=np.float32)
        ym = np.asarray(ck["y_scaler"]["mean"], dtype=np.float32)
        ys = np.asarray(ck["y_scaler"]["std"], dtype=np.float32)

        te = load_split(SPLIT_DIR, split_name).test
        n = len(te)
        log(f"=== {split_name}: {n:,} test rows ===")
        y_pred = np.empty((n, len(OUTPUT_COLS)), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, args.chunk):
                xb = ((X[te[s:s + args.chunk]] - xm) / xs).astype(np.float32)
                out = reg(torch.from_numpy(xb).to(device)).float().cpu().numpy()
                y_pred[s:s + args.chunk] = out * ys + ym

        y_true = np.asarray(Y[te])
        cls_te = cls[te]
        defined = cls_te != CLASS_ALL_NEG
        valid = cls_te == CLASS_VALID
        clamped = clamp_physical(y_pred)

        rows = {}
        for j, c in enumerate(OUTPUT_COLS):
            m = valid if c == "fuel" else defined
            raw = per_output(y_true[m][:, [j]], y_pred[m][:, [j]], cols=[c])[c]
            cl = per_output(y_true[m][:, [j]], clamped[m][:, [j]], cols=[c])[c]
            n_changed = int(np.sum(y_pred[m][:, j] != clamped[m][:, j]))
            rows[c] = {
                "bounds": list(PHYSICAL_BOUNDS[c]),
                "n_scored_rows": int(m.sum()),
                "n_predictions_clamped": n_changed,
                "pct_predictions_clamped": 100.0 * n_changed / int(m.sum()),
                "R2_unclamped": raw["R2"], "R2_clamped": cl["R2"],
                "MAE_unclamped": raw["MAE"], "MAE_clamped": cl["MAE"],
            }
            log(f"  {c:12s} clamped {rows[c]['pct_predictions_clamped']:6.3f}% "
                f"| R2 {raw['R2']:.6f} -> {cl['R2']:.6f}")
        report[split_name] = rows

    out = ROOT / "results" / "evaluation" / "clamp_report.json"
    out.write_text(json.dumps(report, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
