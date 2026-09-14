"""Score every grid regressor on the S0_screen selection partition.

The selection partition is the held-out 10% of the screening rows -- never
trained on, and distinct from every final test set. Metrics mirror
03_evaluate exactly: predictions are inverse-scaled with the run's own
scalers, clamped to physical bounds, and scored with the oracle mask (nine
non-fuel outputs on rows that are not completely infeasible, fuel on
fully-valid rows only).

Writes selection_eval.json next to each run's checkpoint. Runs that already
have one are skipped, so this is resumable and cheap to re-invoke.

  python scripts/21_alpha_grid_eval.py [--force]
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
sys.path.insert(0, str(ROOT / "scripts"))

from s175 import data as sdata
from s175.metrics import clamp_physical, safe_direction_score
from s175.splits import load_split

ev = __import__("03_evaluate")

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"
SPLIT = "S0_screen"
CHUNK = 250_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate runs that already have selection_eval.json")
    args = ap.parse_args()

    # regressor.pt is written well before run.json, so a run that is still
    # training (or was interrupted) has a checkpoint and no config to read:
    # that is a run to skip, not a crash.
    run_dirs = sorted(ROOT.glob(f"runs/grid_a*/{SPLIT}/seed_*"))
    todo = [d for d in run_dirs
            if (d / "regressor.pt").exists() and (d / "run.json").exists()
            and (args.force or not (d / "selection_eval.json").exists())]
    log(f"{len(run_dirs)} grid run dirs, {len(todo)} to evaluate")
    if not todo:
        return 0

    arr, _meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    sel = load_split(SPLIT_DIR, SPLIT).test
    y_true = np.asarray(Y[sel])
    cls_sel = cls[sel]
    log(f"selection partition: {len(sel):,} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in todo:
        run = json.loads((d / "run.json").read_text())
        reg, reg_ck = ev.load_component(d, "regressor", device)
        x_scaler = ev.Scaler(reg_ck["x_scaler"])
        y_scaler = ev.Scaler(reg_ck["y_scaler"])
        y_scaled, _, _ = ev.predict_all((reg, None, None), X, sel, x_scaler,
                                        device, chunk=CHUNK)
        y_pred = clamp_physical(y_scaler.inverse(y_scaled))
        per_out, counts = ev.regression_metrics(y_true, y_pred, cls_sel)
        met, total = safe_direction_score(per_out)
        rel = float(np.mean([per_out[c]["rel_MAE_sd_pct"] for c in per_out]))
        r2 = float(np.mean([per_out[c]["R2"] for c in per_out]))
        result = {
            "config": run["config"]["name"],
            "alpha_speed": run["config"]["alpha_speed"],
            "alpha_other": run["config"]["alpha_others"],
            "seed": run["seed"],
            "split": SPLIT,
            "partition": "selection (the split's held-out test part)",
            "n_rows": counts,
            "safe_direction": f"{met}/{total}",
            "mean_rel_MAE_sd_pct": rel,
            "mean_R2": r2,
            "per_output": per_out,
        }
        (d / "selection_eval.json").write_text(json.dumps(result, indent=2))
        log(f"{run['config']['name']} seed {run['seed']}: "
            f"safe {met}/{total}  mean MAE {rel:.3f}% of SD  "
            f"mean R2 {r2:.5f}  speed OE "
            f"{per_out['speed']['overpred_pct_positive']:.1f}%")
        del reg
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
