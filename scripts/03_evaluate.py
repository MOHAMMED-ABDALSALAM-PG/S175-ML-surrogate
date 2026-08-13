"""Evaluate every completed run of the matrix on its held-out TEST split.

Run:
  python scripts/03_evaluate.py                       # all runs found under runs/
  python scripts/03_evaluate.py --split S1_random --seed 0

For each run this loads the three checkpoints (with the scalers and the
validation-selected thresholds stored inside them), streams the test rows
through the models, and reports:

*  Regressor (oracle feasibility): per-output MAE/RMSE/R2 etc. on the rows
   where the true value is defined -- the nine non-fuel outputs on rows that
   are not completely infeasible, fuel on fully-valid rows only. This mirrors
   the masked training objective. Predictions are clamped to physical bounds
   before scoring (metrics.clamp_physical).
*  Each classifier at its stored threshold, with denominators and Wilson CIs.
*  The assembled two-stage surrogate: three-class confusion matrix against
   the simulator's feasibility class, with the dangerous cells (surrogate
   emits numbers where the simulator has none) called out explicitly.

Per-run results go to runs/<config>/<split>/seed_<n>/eval.json; the aggregate
tables and figures go to results/evaluation/ and figures/.
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
from s175.columns import (OUTPUT_COLS, FUEL_IDX, CLASS_VALID, CLASS_FUEL_ONLY,
                          CLASS_ALL_NEG, CLASS_NAMES)
from s175.metrics import (binary_classifier, clamp_physical, per_output,
                          safe_direction_score, wilson_interval)
from s175.models import MLP
from s175.splits import load_split

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"
RUNS = ROOT / "runs"
CHUNK = 2_000_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_component(run_dir: Path, name: str, device):
    ckpt = torch.load(run_dir / f"{name}.pt", map_location="cpu", weights_only=True)
    model = MLP(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


class Scaler:
    def __init__(self, state: dict):
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)

    def transform(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    def inverse(self, a):
        return (a * self.std + self.mean).astype(np.float32)


@torch.no_grad()
def predict_all(models, X, idx, x_scaler, device, chunk=CHUNK):
    """One pass over the test rows: regressor output (scaled) + both
    classifier probabilities, gathered chunk-wise from the memmap.
    Classifiers may be None (regressor-only ablation runs)."""
    reg, c1, c2 = models
    n = len(idx)
    y_scaled = np.empty((n, len(OUTPUT_COLS)), dtype=np.float32)
    p1 = np.empty(n, dtype=np.float32) if c1 is not None else None
    p2 = np.empty(n, dtype=np.float32) if c2 is not None else None
    for s in range(0, n, chunk):
        xb = torch.from_numpy(x_scaler.transform(X[idx[s:s + chunk]])).to(device)
        y_scaled[s:s + chunk] = reg(xb).float().cpu().numpy()
        if c1 is not None:
            p1[s:s + chunk] = torch.sigmoid(c1(xb).squeeze(-1)).float().cpu().numpy()
        if c2 is not None:
            p2[s:s + chunk] = torch.sigmoid(c2(xb).squeeze(-1)).float().cpu().numpy()
    return y_scaled, p1, p2


def regression_metrics(y_true, y_pred, cls_test):
    """Oracle-masked per-output metrics.

    Nine non-fuel outputs: rows not completely infeasible (their targets are
    all defined there). Fuel: fully-valid rows only (fuel-only rows have no
    fuel target). Mirrors the training mask exactly.
    """
    defined = cls_test != CLASS_ALL_NEG
    valid = cls_test == CLASS_VALID
    wide = per_output(y_true[defined], y_pred[defined])
    fuel = per_output(y_true[valid], y_pred[valid])
    out = {c: (fuel[c] if c == "fuel" else wide[c]) for c in OUTPUT_COLS}
    met, total = safe_direction_score(out)
    return out, {"n_rows_nonfuel_outputs": int(defined.sum()),
                 "n_rows_fuel_output": int(valid.sum()),
                 "safe_direction_met": met, "safe_direction_total": total}


def evaluate_run(run_dir: Path, X, Y, cls, device, chunk=CHUNK) -> dict:
    split_name = run_dir.parent.name
    seed = int(run_dir.name.split("_")[1])
    sp = load_split(SPLIT_DIR, split_name)
    te = sp.test
    log(f"=== {split_name} seed {seed}: {len(te):,} test rows ===")

    reg, reg_ck = load_component(run_dir, "regressor", device)
    # loss-ablation arms train only the regressor; score what exists
    have_clf = (run_dir / "clf1.pt").exists() and (run_dir / "clf2.pt").exists()
    if have_clf:
        c1, c1_ck = load_component(run_dir, "clf1", device)
        c2, c2_ck = load_component(run_dir, "clf2", device)
        # all three components must share one input scaler
        for name, ck in (("clf1", c1_ck), ("clf2", c2_ck)):
            if not np.allclose(ck["x_scaler"]["mean"], reg_ck["x_scaler"]["mean"]):
                raise ValueError(f"{run_dir}: {name} x_scaler differs from regressor's")
        thr1, thr2 = float(c1_ck["threshold"]), float(c2_ck["threshold"])
    else:
        c1 = c2 = None
        thr1 = thr2 = None
        log("  regressor-only run (no classifier checkpoints)")
    x_scaler = Scaler(reg_ck["x_scaler"])
    y_scaler = Scaler(reg_ck["y_scaler"])

    t0 = time.perf_counter()
    y_scaled, p1, p2 = predict_all((reg, c1, c2), X, te, x_scaler, device,
                                   chunk=chunk)
    infer_s = time.perf_counter() - t0
    log(f"  inference {infer_s:.1f}s ({len(te)/infer_s:,.0f} rows/s)")

    y_pred = clamp_physical(y_scaler.inverse(y_scaled))
    y_true = np.asarray(Y[te])
    cls_te = cls[te]

    # --- regressor, oracle feasibility -----------------------------------
    reg_out, reg_meta = regression_metrics(y_true, y_pred, cls_te)

    if not have_clf:
        result = {
            "split": split_name, "seed": seed,
            "regressor_only": True,
            "n_test_rows": int(len(te)),
            "test_class_counts": sdata.class_counts(cls_te),
            "inference_seconds": infer_s,
            "regressor": {"meta": reg_meta, "per_output": reg_out},
        }
        (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
        log(f"  R2 speed {reg_out['speed']['R2']:.4f} power {reg_out['power']['R2']:.4f} "
            f"fuel {reg_out['fuel']['R2']:.4f} | safe-direction "
            f"{reg_meta['safe_direction_met']}/{reg_meta['safe_direction_total']}")
        return result

    # --- classifiers at their stored thresholds --------------------------
    # clf1 sees every row; clf2 is scored on the population it faces at
    # inference: the rows that are not completely infeasible.
    lab1 = cls_te == CLASS_ALL_NEG
    m1 = binary_classifier(lab1, p1 > thr1)
    sub2 = cls_te != CLASS_ALL_NEG
    lab2 = cls_te[sub2] == CLASS_FUEL_ONLY
    m2 = binary_classifier(lab2, p2[sub2] > thr2)

    # --- assembled surrogate: three-class decision -----------------------
    pred_cls = np.full(len(te), CLASS_VALID, dtype=np.int8)
    pred_cls[p2 > thr2] = CLASS_FUEL_ONLY
    pred_cls[p1 > thr1] = CLASS_ALL_NEG   # C1 overrides, as in Surrogate.forward
    conf = np.zeros((3, 3), dtype=np.int64)
    for t in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG):
        for p in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG):
            conf[t, p] = int(np.sum((cls_te == t) & (pred_cls == p)))
    n = conf.sum()
    # dangerous: simulator says a value is undefined, surrogate supplies one
    fuel_missed = conf[CLASS_FUEL_ONLY, CLASS_VALID]
    allneg_missed = conf[CLASS_ALL_NEG, CLASS_VALID] + conf[CLASS_ALL_NEG, CLASS_FUEL_ONLY]
    n_fuel_pos = conf[CLASS_FUEL_ONLY].sum()
    n_allneg_pos = conf[CLASS_ALL_NEG].sum()
    e2e = {
        "confusion_matrix_true_x_pred": conf.tolist(),
        "class_order": [CLASS_NAMES[c] for c in
                        (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG)],
        "accuracy": float(np.trace(conf) / n),
        "dangerous_allneg_passed": int(allneg_missed),
        "dangerous_allneg_rate": float(allneg_missed / n_allneg_pos) if n_allneg_pos else float("nan"),
        "dangerous_allneg_rate_ci95": list(wilson_interval(int(allneg_missed), int(n_allneg_pos))),
        "dangerous_fuel_passed": int(fuel_missed),
        "dangerous_fuel_rate": float(fuel_missed / n_fuel_pos) if n_fuel_pos else float("nan"),
        "dangerous_fuel_rate_ci95": list(wilson_interval(int(fuel_missed), int(n_fuel_pos))),
    }

    result = {
        "split": split_name, "seed": seed,
        "n_test_rows": int(len(te)),
        "test_class_counts": sdata.class_counts(cls_te),
        "thresholds": {"clf1": thr1, "clf2": thr2},
        "inference_seconds": infer_s,
        "regressor": {"meta": reg_meta, "per_output": reg_out},
        "clf1": m1, "clf2": m2,
        "end_to_end": e2e,
    }
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    log(f"  R2 speed {reg_out['speed']['R2']:.4f} power {reg_out['power']['R2']:.4f} "
        f"fuel {reg_out['fuel']['R2']:.4f} | clf1 FNR {m1['false_negative_rate']:.2e} "
        f"clf2 FNR {m2['false_negative_rate']:.2e} | e2e acc {e2e['accuracy']:.4f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="v3_masked_peroutput")
    ap.add_argument("--split", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate runs that already have eval.json")
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="rows per inference chunk; lower it when the GPU is "
                         "shared with another job")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    log(f"loaded {len(X):,} rows | device {device}")

    run_dirs = sorted((RUNS / args.config_name).glob("*/seed_*"))
    if args.split:
        run_dirs = [d for d in run_dirs if d.parent.name == args.split]
    if args.seed is not None:
        run_dirs = [d for d in run_dirs if d.name == f"seed_{args.seed}"]
    run_dirs = [d for d in run_dirs if (d / "run.json").exists()]
    if not run_dirs:
        raise SystemExit("no completed runs found")

    results = []
    for d in run_dirs:
        if (d / "eval.json").exists() and not args.force:
            log(f"SKIP {d.parent.name}/{d.name} (eval.json exists)")
            results.append(json.loads((d / "eval.json").read_text()))
            continue
        results.append(evaluate_run(d, X, Y, cls, device, chunk=args.chunk))

    outdir = ROOT / "results" / "evaluation"
    outdir.mkdir(parents=True, exist_ok=True)
    # the canonical all_runs.json belongs to the main config; ablation and
    # baseline configs aggregate into their own file so they never clobber it
    agg = ("all_runs.json" if args.config_name == "v3_masked_peroutput"
           else f"all_runs_{args.config_name}.json")
    (outdir / agg).write_text(json.dumps(results, indent=2))
    log(f"wrote {outdir / agg} ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
