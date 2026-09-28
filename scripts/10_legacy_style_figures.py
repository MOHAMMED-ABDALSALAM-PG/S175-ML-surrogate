"""Render the paper's per-output figure set from the trained runs.

Regenerates the standard figure types -- pred-vs-actual grids, confusion
matrices, residual histograms, the sea-condition error heatmap, and the
throughput curve -- from the v3_masked_peroutput runs.

Figures (default: S1_random seed 0, the headline run):

  step7_pipeline_scatter_all_outputs    2x5 pred-vs-actual, full pipeline,
                                        missed-infeasible rows marked
  step7_regression_scatter_all_outputs  2x5, regressor with oracle feasibility
  step7_classification_confusion        Clf1 / Clf2 / combined 3-class,
                                        absolute counts
  step7_residual_distributions          2x5 residual histograms
  step7_error_heatmap_sea_conditions    MAE/range % by output and Hs band
  step8_inference_speed                 measured throughput vs batch size

The sea-condition heatmap normalises MAE by each output's GLOBAL range, not
the per-band range: a per-band normalisation divides by zero wherever an
output is constant within a band (slam is identically 0 in calm seas).

Run:
  python scripts/10_legacy_style_figures.py                 # S1_random seed 0
  python scripts/10_legacy_style_figures.py --split S4_corner --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s175 import data as sdata
from s175.columns import (OUTPUT_COLS, FEATURE_COLS, UNITS, CLASS_VALID,
                          CLASS_FUEL_ONLY, CLASS_ALL_NEG)
from s175.metrics import clamp_physical, r2
from s175.models import MLP
from s175.splits import load_split

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"
RUNS = ROOT / "runs"
CHUNK = 2_000_000

# palette: one data series (blue) + status red for dangerous marks; the
# two-series timing chart pairs blue with dark orange (CVD-safe pairing),
# reinforced by distinct markers so identity is never color-alone.
BLUE = "#2a78d6"
DARKBLUE = "#0b3a75"
ORANGE = "#c2410c"
RED = "#d11a2a"

SEA_BANDS = [(0.0, 2.0, "Calm (0-2m)"), (2.0, 5.0, "Moderate (2-5m)"),
             (5.0, 8.0, "Rough (5-8m)"), (8.0, 10.001, "Extreme (8-10m)")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_component(run_dir: Path, name: str, device):
    ck = torch.load(run_dir / f"{name}.pt", map_location="cpu", weights_only=True)
    model = MLP(**ck["model_config"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


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
    reg, c1, c2 = models
    n = len(idx)
    y_scaled = np.empty((n, len(OUTPUT_COLS)), dtype=np.float32)
    p1 = np.empty(n, dtype=np.float32)
    p2 = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        xb = torch.from_numpy(x_scaler.transform(X[idx[s:s + chunk]])).to(device)
        y_scaled[s:s + chunk] = reg(xb).float().cpu().numpy()
        p1[s:s + chunk] = torch.sigmoid(c1(xb).squeeze(-1)).float().cpu().numpy()
        p2[s:s + chunk] = torch.sigmoid(c2(xb).squeeze(-1)).float().cpu().numpy()
    return y_scaled, p1, p2


def save(fig, outdir: Path, name: str):
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {name}.png/.pdf")


def panel_label(c: str) -> str:
    return f"{c} [{UNITS[c]}]"


# ---------------------------------------------------------------------------
# figure builders
# ---------------------------------------------------------------------------
def fig_pipeline_scatter(y_true, y_pred, cls_te, pred_cls, n_test, outdir, rng):
    """2x5 pred-vs-actual under the full pipeline, old step7 layout."""
    fig, axes = plt.subplots(2, 5, figsize=(30, 11))
    fig.suptitle(f"Full Pipeline — Predicted vs Actual "
                 f"(Clf1 + Clf2 + Regressor, All Test Rows: {n_test:,})",
                 fontsize=16, fontweight="bold")
    for i, c in enumerate(OUTPUT_COLS):
        ax = axes[i // 5, i % 5]
        if c == "fuel":
            defined = cls_te == CLASS_VALID          # fuel target exists
            emitted = pred_cls == CLASS_VALID        # surrogate emits fuel
        else:
            defined = cls_te != CLASS_ALL_NEG
            emitted = pred_cls != CLASS_ALL_NEG
        both = defined & emitted
        # dangerous: simulator has no value, surrogate emitted one
        missed = ~defined & emitted
        # suppressed: simulator has a value, surrogate suppressed it
        fp = defined & ~emitted

        t, p = y_true[both, i], y_pred[both, i]
        mae = float(np.mean(np.abs(p - t)))
        r2v = r2(t, p)

        n_show = min(len(t), 150_000)
        sub = rng.choice(len(t), n_show, replace=False)
        ax.scatter(t[sub], p[sub], s=2, alpha=0.15, color=BLUE, edgecolors="none",
                   rasterized=True, label=f"Valid ({both.sum():,})")
        if missed.any():
            ax.scatter(y_true[missed, i], y_pred[missed, i], s=14, marker="x",
                       color=RED, alpha=0.8, rasterized=True,
                       label=f"Missed -1 ({missed.sum():,})")
        lo = min(float(t.min()), 0.0 if missed.any() else float(t.min()))
        hi = float(t.max())
        ax.plot([max(lo, float(t.min())), hi], [max(lo, float(t.min())), hi],
                "--", color=RED, lw=1.2, label="Perfect")
        ax.set_title(f"{c}\nMAE={mae:.4f}, R²={r2v:.4f}\n"
                     f"FN={missed.sum():,}, FP={fp.sum():,}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Actual [{UNITS[c]}]", fontsize=9)
        ax.set_ylabel(f"Predicted [{UNITS[c]}]", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, "step7_pipeline_scatter_all_outputs")


def fig_regression_scatter(y_true, y_pred, cls_te, outdir, rng):
    """2x5 regressor-only scatter with oracle feasibility, old layout."""
    fig, axes = plt.subplots(2, 5, figsize=(30, 11))
    fig.suptitle("Regressor — Predicted vs Actual (oracle feasibility, "
                 "test rows with a defined target)", fontsize=16, fontweight="bold")
    for i, c in enumerate(OUTPUT_COLS):
        ax = axes[i // 5, i % 5]
        defined = (cls_te == CLASS_VALID) if c == "fuel" else (cls_te != CLASS_ALL_NEG)
        t, p = y_true[defined, i], y_pred[defined, i]
        mae = float(np.mean(np.abs(p - t)))
        r2v = r2(t, p)
        sub = rng.choice(len(t), min(len(t), 150_000), replace=False)
        ax.scatter(t[sub], p[sub], s=2, alpha=0.15, color=BLUE, edgecolors="none",
                   rasterized=True, label=f"n={defined.sum():,}")
        ax.plot([t.min(), t.max()], [t.min(), t.max()], "--", color=RED, lw=1.2,
                label="Perfect")
        ax.set_title(f"{c}\nMAE={mae:.4f}, R²={r2v:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Actual [{UNITS[c]}]", fontsize=9)
        ax.set_ylabel(f"Predicted [{UNITS[c]}]", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "step7_regression_scatter_all_outputs")


def _confusion_panel(ax, conf, labels, cmap, title):
    ax.imshow(conf, cmap=cmap, vmin=0, vmax=conf.max())
    thresh = 0.55 * conf.max()
    for r in range(conf.shape[0]):
        for cc in range(conf.shape[1]):
            ax.text(cc, r, f"{conf[r, cc]:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if conf[r, cc] > thresh else "black")
    ax.set_xticks(range(len(labels)), labels, fontsize=11)
    ax.set_yticks(range(len(labels)), labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")


def fig_confusion(cls_te, pred_cls, p1, p2, thr1, thr2, outdir):
    """Clf1 / Clf2 / combined three-class, absolute counts -- old layout."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))

    lab1 = cls_te == CLASS_ALL_NEG
    pr1 = p1 > thr1
    c1 = np.array([[np.sum(~lab1 & ~pr1), np.sum(~lab1 & pr1)],
                   [np.sum(lab1 & ~pr1), np.sum(lab1 & pr1)]], dtype=np.int64)
    acc1 = (c1[0, 0] + c1[1, 1]) / c1.sum()
    _confusion_panel(axes[0], c1, ["Feasible", "Infeasible"], "Blues",
                     f"Classifier 1\nAcc={acc1:.4f}, FN={c1[1, 0]:,}")

    sub = cls_te != CLASS_ALL_NEG                    # population C2 faces
    lab2 = cls_te[sub] == CLASS_FUEL_ONLY
    pr2 = p2[sub] > thr2
    c2 = np.array([[np.sum(~lab2 & ~pr2), np.sum(~lab2 & pr2)],
                   [np.sum(lab2 & ~pr2), np.sum(lab2 & pr2)]], dtype=np.int64)
    acc2 = (c2[0, 0] + c2[1, 1]) / c2.sum()
    _confusion_panel(axes[1], c2, ["Fuel-Valid", "Fuel-Invalid"], "Oranges",
                     f"Classifier 2\nAcc={acc2:.4f}, FN={c2[1, 0]:,}")

    conf = np.zeros((3, 3), dtype=np.int64)
    for t in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG):
        for p in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG):
            conf[t, p] = int(np.sum((cls_te == t) & (pred_cls == p)))
    acc3 = np.trace(conf) / conf.sum()
    _confusion_panel(axes[2], conf, ["Valid", "Fuel-1", "All-1"], "Greens",
                     f"Combined 3-Class\nAcc={acc3:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "step7_classification_confusion")


def fig_residuals(y_true, y_pred, cls_te, outdir):
    """2x5 residual histograms on defined rows, old layout."""
    fig, axes = plt.subplots(2, 5, figsize=(30, 11))
    for i, c in enumerate(OUTPUT_COLS):
        ax = axes[i // 5, i % 5]
        defined = (cls_te == CLASS_VALID) if c == "fuel" else (cls_te != CLASS_ALL_NEG)
        res = y_pred[defined, i] - y_true[defined, i]
        # over-prediction on positive-truth rows, matching the table metric
        # (overpred_pct_positive); over all defined rows the clamped zeros of
        # the probability outputs dilute the rate and contradict the table
        pos = y_true[defined, i] > 1e-6
        over = float(np.mean(res[pos] > 0) * 100.0)
        mean = float(res.mean())
        lo, hi = np.percentile(res, [0.05, 99.95])
        # blue-on-white scheme: one blue for the data, neutral ink for the
        # zero reference, dark blue for the mean of the same (blue) series
        ax.hist(res, bins=120, range=(lo, hi), density=True, color=BLUE,
                alpha=0.75)
        ax.axvline(0.0, color="#4a4a4a", ls="--", lw=1.4, label="Zero error")
        ax.axvline(mean, color=DARKBLUE, lw=1.6, label=f"Mean={mean:.4f}")
        ax.set_title(f"{c}\nOverest={over:.1f}% (positive rows), Mean={mean:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Residual (Pred - Actual) [{UNITS[c]}]", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "step7_residual_distributions")


def fig_sea_heatmap(y_true, y_pred, cls_te, hs, outdir):
    """MAE / GLOBAL output range, % -- by output and Hs band.

    The old figure divided by the per-band range, which is zero wherever an
    output is constant within a band; global range is finite by construction.
    """
    grid = np.full((len(OUTPUT_COLS), len(SEA_BANDS)), np.nan)
    for i, c in enumerate(OUTPUT_COLS):
        defined = (cls_te == CLASS_VALID) if c == "fuel" else (cls_te != CLASS_ALL_NEG)
        rng_glob = float(y_true[defined, i].max() - y_true[defined, i].min())
        for j, (lo, hi, _name) in enumerate(SEA_BANDS):
            m = defined & (hs >= lo) & (hs < hi)
            if m.any() and rng_glob > 0:
                mae = float(np.mean(np.abs(y_pred[m, i] - y_true[m, i])))
                grid[i, j] = 100.0 * mae / rng_glob
    fig, ax = plt.subplots(figsize=(13, 10))
    vmax = max(1.0, float(np.nanmax(grid)))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                # ink switches on the cell's colour (its position on the scale),
                # not on the data max -- pale cells always take dark ink
                ax.text(j, i, f"{grid[i, j]:.3f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if grid[i, j] > 0.6 * vmax else "black")
    ax.set_xticks(range(len(SEA_BANDS)), [b[2] for b in SEA_BANDS], fontsize=11)
    ax.set_yticks(range(len(OUTPUT_COLS)), [panel_label(c) for c in OUTPUT_COLS],
                  fontsize=11)
    ax.set_title("Relative Error (MAE / global output range, %) by Output and "
                 "Sea Condition\n(Regressor, rows with a defined target)",
                 fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="MAE / global range (%)")
    fig.tight_layout()
    save(fig, outdir, "step7_error_heatmap_sea_conditions")


@torch.no_grad()
def fig_inference_speed(models, x_scaler, X, te, device, outdir):
    """Measured throughput vs batch size, regressor alone and full pipeline."""
    reg, c1, c2 = models
    sizes = [512, 4096, 16384, 65536, 262144]
    pool = torch.from_numpy(x_scaler.transform(X[te[:max(sizes) * 2]])).to(device)
    rows = {"regressor": [], "pipeline": []}
    for bs in sizes:
        xb = pool[:bs]
        for mode in ("regressor", "pipeline"):
            def run_once():
                y = reg(xb)
                if mode == "pipeline":
                    torch.sigmoid(c1(xb).squeeze(-1))
                    torch.sigmoid(c2(xb).squeeze(-1))
                return y
            for _ in range(3):
                run_once()
            torch.cuda.synchronize()
            reps = []
            for _ in range(10):
                t0 = time.perf_counter()
                run_once()
                torch.cuda.synchronize()
                reps.append(time.perf_counter() - t0)
            dt = float(np.median(reps))
            rows[mode].append(bs / dt)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sizes, rows["regressor"], "-o", color=BLUE, lw=2, ms=8,
            label="Regressor only")
    ax.plot(sizes, rows["pipeline"], "-s", color=ORANGE, lw=2, ms=8,
            label="Full pipeline (Clf1 + Clf2 + Regressor)")
    for x, y in zip(sizes[-1:], rows["regressor"][-1:]):
        ax.annotate(f"{y:,.0f} rows/s", (x, y), textcoords="offset points",
                    xytext=(-8, 10), ha="right", fontsize=9, color=BLUE)
    for x, y in zip(sizes[-1:], rows["pipeline"][-1:]):
        ax.annotate(f"{y:,.0f} rows/s", (x, y), textcoords="offset points",
                    xytext=(-8, -16), ha="right", fontsize=9, color=ORANGE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Throughput [rows/s]", fontsize=11)
    ax.set_title("Measured Inference Throughput on GPU", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, outdir, "step8_inference_speed")
    (outdir / "inference_speed.json").write_text(json.dumps(
        {"batch_sizes": sizes, "rows_per_s": rows,
         "gpu": torch.cuda.get_device_name(0)}, indent=2))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="v3_masked_peroutput")
    ap.add_argument("--split", default="S1_random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="rows per inference chunk; lower it when the GPU is "
                         "shared with another job")
    ap.add_argument("--clean", action="store_true",
                    help="score only test rows disjoint from the screening "
                         "subsample (requires results/evaluation/"
                         "screening_rows.npy from 03_evaluate.py --clean)")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["pipeline", "regression", "confusion",
                             "residuals", "seaheat", "speed"],
                    help="regenerate only the named figures (default: all); "
                         "note 'speed' RE-MEASURES and overwrites "
                         "inference_speed.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.config_name / args.split / f"seed_{args.seed}"
    outdir = ROOT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    arr, _meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    sp = load_split(SPLIT_DIR, args.split)
    te = sp.test
    if args.clean:
        srows = np.load(ROOT / "results" / "evaluation" / "screening_rows.npy")
        in_sample = np.zeros(len(X), dtype=bool)
        in_sample[srows] = True
        n0 = len(te)
        te = te[~in_sample[te]]
        log(f"--clean: excluded {n0 - len(te):,} screening-overlap rows "
            f"({len(te):,} remain)")
    log(f"{args.split} seed {args.seed}: {len(te):,} test rows | {device}")

    reg, reg_ck = load_component(run_dir, "regressor", device)
    c1, c1_ck = load_component(run_dir, "clf1", device)
    c2, c2_ck = load_component(run_dir, "clf2", device)
    x_scaler, y_scaler = Scaler(reg_ck["x_scaler"]), Scaler(reg_ck["y_scaler"])
    thr1, thr2 = float(c1_ck["threshold"]), float(c2_ck["threshold"])

    t0 = time.perf_counter()
    y_scaled, p1, p2 = predict_all((reg, c1, c2), X, te, x_scaler, device,
                                   chunk=args.chunk)
    log(f"inference {time.perf_counter()-t0:.1f}s")
    y_pred = clamp_physical(y_scaler.inverse(y_scaled))
    y_true = np.asarray(Y[te])
    cls_te = cls[te]
    hs = np.asarray(X[te, FEATURE_COLS.index("Hs")])

    pred_cls = np.full(len(te), CLASS_VALID, dtype=np.int8)
    pred_cls[p2 > thr2] = CLASS_FUEL_ONLY
    pred_cls[p1 > thr1] = CLASS_ALL_NEG

    wanted = set(args.only) if args.only else {
        "pipeline", "regression", "confusion", "residuals", "seaheat", "speed"}
    rng = np.random.default_rng(0)
    if "pipeline" in wanted:
        fig_pipeline_scatter(y_true, y_pred, cls_te, pred_cls, len(te), outdir, rng)
    if "regression" in wanted:
        fig_regression_scatter(y_true, y_pred, cls_te, outdir, rng)
    if "confusion" in wanted:
        fig_confusion(cls_te, pred_cls, p1, p2, thr1, thr2, outdir)
    if "residuals" in wanted:
        fig_residuals(y_true, y_pred, cls_te, outdir)
    if "seaheat" in wanted:
        fig_sea_heatmap(y_true, y_pred, cls_te, hs, outdir)
    if "speed" in wanted and device.type == "cuda":
        fig_inference_speed((reg, c1, c2), x_scaler, X, te, device, outdir)
    log("all figures done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
