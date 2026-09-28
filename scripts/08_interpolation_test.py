"""ML surrogate vs the physical simulator on OFF-GRID inputs.

  python scripts/08_interpolation_test.py

The training data is a full factorial grid. The strongest generalisation
evidence is fresh SIMULATOR runs at input values that lie BETWEEN the grid
levels -- points the surrogate has never seen in any split. Two such runs
exist (produced with WeatherRouting ver. 0.993 in May 2026):

  midpoint : full factorial of midpoints between selected adjacent grid levels (4,608 rows)
  random   : full factorial of selected off-grid values that are not midpoints
             (54,675 rows; the "irregular" design of the paper; no random-number
             generator was involved)  -- value lists: results/evaluation/offgrid_design_levels.json

This script verifies -- not assumes -- that every varying input value in
those files is off-grid (minimum distance to the nearest training grid level
is reported per feature), then scores the CURRENT v3 model
(runs/v3_masked_peroutput/S1_random/seed_0) under the same protocol as
03_evaluate.py: clamped predictions, oracle feasibility masking, fuel scored
on fully-valid rows only.

Honesty note carried into the output: the two files differ in size, in the
number of distinct levels, and in their distance-to-grid distributions, so
midpoint-vs-random differences must not be over-interpreted; the supported
claim is that accuracy holds at off-grid inputs in both.

Outputs:
  results/evaluation/interpolation_test.json
  results/evaluation/interpolation_table.csv
  figures/fig_interpolation_offgrid.(png|pdf)
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from s175 import data as sdata
from s175.columns import (ALL_COLS, FEATURE_COLS, OUTPUT_COLS, HEADER_TOKENS,
                          CLASS_VALID, CLASS_ALL_NEG, CLASS_FUEL_ONLY)
from s175.metrics import clamp_physical, per_output, safe_direction_score, binary_classifier

ev = __import__("03_evaluate")

SIM_FILES = {
    "midpoint": Path.home() / "S175_simulation" / "S175" / "output_interpolation.txt",
    "random": Path.home() / "S175_simulation" / "S175" / "output_interpolation_random.txt",
}
RUN_DIR = ROOT / "runs" / "v3_masked_peroutput" / "S1_random" / "seed_0"
OUT = ROOT / "results" / "evaluation"
FIG = ROOT / "figures"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_sim_output(path: Path) -> pd.DataFrame:
    """Parse a WeatherRouting output file, verifying the header schema."""
    lines = path.read_text(errors="replace").splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if ln.count(";") == len(ALL_COLS) - 1 and "mean draft" in ln.lower():
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"{path}: no header line found")
    fields = [s.strip().lower() for s in lines[header_i].split(";")]
    for i, (field, token) in enumerate(zip(fields, HEADER_TOKENS)):
        if token not in field:
            raise ValueError(f"{path}: column {i} expected {token!r}, got {field!r}")
    body = "\n".join(ln for ln in lines[header_i + 1:] if ln.count(";") == len(ALL_COLS) - 1)
    df = pd.read_csv(io.StringIO(body), sep=";", header=None, names=ALL_COLS,
                     dtype=np.float64)
    if df.isna().any().any():
        raise ValueError(f"{path}: NaNs after parsing")
    return df


def grid_levels(X) -> dict[str, np.ndarray]:
    """The exact training grid levels of each varying input."""
    levels = {}
    for k, c in enumerate(FEATURE_COLS):
        levels[c] = np.unique(np.asarray(X[:, k], dtype=np.float64))
    return levels


def offgrid_report(df: pd.DataFrame, levels: dict) -> dict:
    """Per feature: distinct test values and their distance to the nearest
    training grid level. Raises if any value coincides with a grid level."""
    rep = {}
    for c in FEATURE_COLS:
        vals = np.unique(df[c].to_numpy())
        dists = np.array([np.abs(levels[c] - v).min() for v in vals])
        on_grid = vals[dists < 1e-6]
        rep[c] = {
            "n_distinct_test_values": int(len(vals)),
            "test_values": [round(float(v), 6) for v in vals],
            "min_distance_to_grid": float(dists.min()),
            "max_distance_to_grid": float(dists.max()),
            "on_grid_values": [float(v) for v in on_grid],
        }
        if len(on_grid):
            raise ValueError(f"{c}: test values {on_grid} lie ON the training "
                             f"grid -- this would not be an off-grid test")
    return rep


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr, _ = sdata.load(ev.RAW, cache_dir=ev.CACHE_DIR)
    X_train_grid, _ = sdata.split_xy(arr)
    levels = grid_levels(X_train_grid)
    log("training grid levels extracted: " +
        ", ".join(f"{c}:{len(v)}" for c, v in levels.items()))

    models, x_scaler, y_scaler = __import__("05_paper_figures").load_model_bundle(
        RUN_DIR, device)
    reg, c1, c2 = models
    c1_thr = torch.load(RUN_DIR / "clf1.pt", map_location="cpu",
                        weights_only=True)["threshold"]
    c2_thr = torch.load(RUN_DIR / "clf2.pt", map_location="cpu",
                        weights_only=True)["threshold"]

    results = {}
    for name, path in SIM_FILES.items():
        df = parse_sim_output(path)
        rep = offgrid_report(df, levels)
        Xs = df[FEATURE_COLS].to_numpy(np.float32)
        Ys = df[OUTPUT_COLS].to_numpy(np.float32)
        cls = sdata.feasibility(Ys, strict=False)
        counts = sdata.class_counts(cls)
        log(f"{name}: {len(df):,} rows parsed, classes {counts}")

        with torch.no_grad():
            xb = torch.from_numpy(x_scaler.transform(Xs)).to(device)
            y_pred = clamp_physical(y_scaler.inverse(reg(xb).float().cpu().numpy()))
            p1 = torch.sigmoid(c1(xb).squeeze(-1)).float().cpu().numpy()
            p2 = torch.sigmoid(c2(xb).squeeze(-1)).float().cpu().numpy()

        reg_out, reg_meta = ev.regression_metrics(Ys, y_pred, cls)
        m1 = binary_classifier(cls == CLASS_ALL_NEG, p1 > c1_thr)
        sub = cls != CLASS_ALL_NEG
        m2 = (binary_classifier(cls[sub] == CLASS_FUEL_ONLY, p2[sub] > c2_thr)
              if sub.any() and (cls[sub] == CLASS_FUEL_ONLY).any() else None)
        results[name] = {
            "sim_file": str(path), "n_rows": int(len(df)),
            "class_counts": counts,
            "offgrid_verification": rep,
            "regressor": {"meta": reg_meta, "per_output": reg_out},
            "clf1": m1, "clf2": m2,
        }
        met, tot = reg_meta["safe_direction_met"], reg_meta["safe_direction_total"]
        log(f"{name}: R2 speed {reg_out['speed']['R2']:.4f} "
            f"fuel {reg_out['fuel']['R2']:.4f} | safe-direction {met}/{tot}")

    results["_note"] = (
        "midpoint and random differ in size, level count and distance-to-grid "
        "distribution; do not over-interpret their difference. The supported "
        "claim: accuracy holds at off-grid inputs in both designs."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interpolation_test.json").write_text(json.dumps(results, indent=2))

    with open(OUT / "interpolation_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["output", "midpoint_R2", "random_R2", "midpoint_MAE",
                    "random_MAE", "midpoint_overpred_pct_positive",
                    "random_overpred_pct_positive", "safe_direction"])
        for c in OUTPUT_COLS:
            a = results["midpoint"]["regressor"]["per_output"][c]
            b = results["random"]["regressor"]["per_output"][c]
            w.writerow([c, a["R2"], b["R2"], a["MAE"], b["MAE"],
                        a["overpred_pct_positive"], b["overpred_pct_positive"],
                        a["safe_direction"]])
    log(f"wrote {OUT / 'interpolation_table.csv'}")

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    xs = np.arange(len(OUTPUT_COLS))
    wd = 0.38
    for off, name, color in ((-wd / 2, "midpoint", "#31538f"),
                             (wd / 2, "random", "#7fa5d1")):
        r2 = [results[name]["regressor"]["per_output"][c]["R2"] for c in OUTPUT_COLS]
        n = results[name]["n_rows"]
        ax.bar(xs + off, r2, wd, label=f"{name} off-grid ({n:,} sim rows)",
               color=color)
    ax.set_ylim(0.9, 1.005)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_xticks(xs, OUTPUT_COLS, rotation=30, ha="right")
    ax.set_ylabel("R² against fresh simulator runs")
    ax.set_title("Surrogate accuracy at inputs strictly between training grid "
                 "levels (v3, S1_random seed 0)")
    ax.legend(fontsize=9)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_interpolation_offgrid.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_interpolation_offgrid.pdf", bbox_inches="tight")
    log("wrote figures/fig_interpolation_offgrid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
