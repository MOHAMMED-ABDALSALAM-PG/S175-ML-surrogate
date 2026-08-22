"""Re-evaluate the final model on test rows that are strictly disjoint from
the 5% screening subsample.

The screening subsample (algorithm screening + both alpha sweeps) predates the
final campaign and was drawn from the full dataset, so ~5% of the final test
rows are expected to also appear in it. This script identifies the overlap
EXACTLY -- every subsample row is mapped back to its row number in the full
factorial table via the lattice index -- and repeats the headline evaluation
of the principal run on the disjoint remainder, so the paper can report
contamination-free numbers.

  python scripts/15_clean_test_eval.py

Output: results/evaluation/clean_test_eval.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from s175 import data as sdata
from s175.columns import (FEATURE_COLS, OUTPUT_COLS, CLASS_VALID,
                          CLASS_FUEL_ONLY, CLASS_ALL_NEG)
from s175.metrics import binary_classifier, clamp_physical
from s175.splits import load_split

baseline = __import__("14_interpolation_baseline")
ev = __import__("03_evaluate")

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
SAMPLE = Path("/home/macierz/mohabdal/S175_sample_5pct_stratified.csv")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"
RUN_DIR = ROOT / "runs" / "v3_masked_peroutput" / "S1_random" / "seed_0"
CHUNK = 250_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# the screening-sample CSV names one column differently
CSV_NAME = {c: ("rpm" if c == "EngRPM" else c) for c in FEATURE_COLS}


def sample_row_numbers(X, levels, strides, perm) -> np.ndarray:
    """Exact row number in the full table of every screening-subsample row."""
    df = pd.read_csv(SAMPLE, sep=";", usecols=list(CSV_NAME.values()))
    log(f"screening subsample: {len(df):,} rows")
    key = np.zeros(len(df), dtype=np.int64)
    for k, c in enumerate(FEATURE_COLS):
        lv = levels[c]                       # float32 grid values
        q = df[CSV_NAME[c]].to_numpy(dtype=np.float32)  # match in float32 exactly
        idx = np.searchsorted(lv, q)
        idx = np.clip(idx, 0, len(lv) - 1)
        if not np.all(lv[idx] == q):
            bad = int(np.sum(lv[idx] != q))
            raise RuntimeError(f"{c}: {bad} subsample values not on the grid")
        key += idx.astype(np.int64) * strides[k]
    rows = perm[key]
    assert len(np.unique(rows)) == len(rows), "duplicate subsample rows"
    return rows


def main() -> int:
    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    log(f"loaded {len(X):,} rows")

    levels = {c: np.unique(np.asarray(X[:, k], dtype=np.float32))
              for k, c in enumerate(FEATURE_COLS)}
    strides, perm = baseline.build_lattice_index(X, levels)
    srows = sample_row_numbers(X, levels, strides, perm)
    in_sample = np.zeros(len(X), dtype=bool)
    in_sample[srows] = True

    report = {"n_sample_rows": int(len(srows)), "overlap_by_split": {}}
    for split_name in ["S1_random", "S2_level_Hs", "S2_level_Chi",
                       "S2_level_draft", "S2_level_Vwind", "S3_block",
                       "S4_corner"]:
        te = load_split(SPLIT_DIR, split_name).test
        n_ov = int(in_sample[te].sum())
        report["overlap_by_split"][split_name] = {
            "n_test": int(len(te)), "n_overlap": n_ov,
            "pct_overlap": 100.0 * n_ov / len(te)}
        log(f"{split_name:16s} test {len(te):>11,} overlap {n_ov:>9,} "
            f"({100.0 * n_ov / len(te):.3f}%)")

    # ---- headline evaluation on the disjoint S1 test rows ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = load_split(SPLIT_DIR, "S1_random").test
    te_clean = te[~in_sample[te]]
    log(f"clean S1 test: {len(te_clean):,} of {len(te):,} rows")

    reg, reg_ck = ev.load_component(RUN_DIR, "regressor", device)
    c1, c1_ck = ev.load_component(RUN_DIR, "clf1", device)
    c2, c2_ck = ev.load_component(RUN_DIR, "clf2", device)
    x_scaler = ev.Scaler(reg_ck["x_scaler"])
    y_scaler = ev.Scaler(reg_ck["y_scaler"])
    thr1, thr2 = float(c1_ck["threshold"]), float(c2_ck["threshold"])

    y_scaled, p1, p2 = ev.predict_all((reg, c1, c2), X, te_clean, x_scaler,
                                      device, chunk=CHUNK)
    y_pred = clamp_physical(y_scaler.inverse(y_scaled))
    y_true = np.asarray(Y[te_clean])
    cls_te = cls[te_clean]

    reg_out, reg_meta = ev.regression_metrics(y_true, y_pred, cls_te)
    lab1 = cls_te == CLASS_ALL_NEG
    m1 = binary_classifier(lab1, p1 > thr1)
    sub2 = cls_te != CLASS_ALL_NEG
    lab2 = cls_te[sub2] == CLASS_FUEL_ONLY
    m2 = binary_classifier(lab2, p2[sub2] > thr2)
    pred_cls = np.full(len(te_clean), CLASS_VALID, dtype=np.int8)
    pred_cls[p2 > thr2] = CLASS_FUEL_ONLY
    pred_cls[p1 > thr1] = CLASS_ALL_NEG
    acc3 = float(np.mean(pred_cls == cls_te))

    orig = json.loads((RUN_DIR / "eval.json").read_text())
    comp = {}
    for c in OUTPUT_COLS:
        a, b = orig["regressor"]["per_output"][c], reg_out[c]
        comp[c] = {"R2_full_test": a["R2"], "R2_clean_test": b["R2"],
                   "MAE_full_test": a["MAE"], "MAE_clean_test": b["MAE"],
                   "overpred_full": a["overpred_pct_positive"],
                   "overpred_clean": b["overpred_pct_positive"]}
        log(f"  {c:12s} R2 {a['R2']:.6f} -> {b['R2']:.6f} | "
            f"OE% {a['overpred_pct_positive']:.2f} -> "
            f"{b['overpred_pct_positive']:.2f}")
    log(f"  clf1 FNR {orig['clf1']['false_negative_rate']:.6f} -> "
        f"{m1['false_negative_rate']:.6f} | clf2 FNR "
        f"{orig['clf2']['false_negative_rate']:.6f} -> "
        f"{m2['false_negative_rate']:.6f} | 3-class acc "
        f"{orig['end_to_end']['accuracy']:.6f} -> {acc3:.6f}")

    report["clean_S1_eval"] = {
        "n_clean_test_rows": int(len(te_clean)),
        "safe_direction": f"{reg_meta['safe_direction_met']}/"
                          f"{reg_meta['safe_direction_total']}",
        "per_output": comp,
        "clf1_fnr": m1["false_negative_rate"],
        "clf2_fnr": m2["false_negative_rate"],
        "three_class_accuracy": acc3,
    }
    out = ROOT / "results" / "evaluation" / "clean_test_eval.json"
    out.write_text(json.dumps(report, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
