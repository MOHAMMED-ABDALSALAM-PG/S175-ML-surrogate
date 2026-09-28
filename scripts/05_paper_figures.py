"""Paper figures and tables beyond the core metric report.

  python scripts/05_paper_figures.py            # uses S1_random seed 0

Outputs (figures/ at 300 dpi PNG + vector PDF, tables under results/evaluation/):
  fig_pred_vs_true.(png|pdf)        hexbin, all ten outputs, test rows only
  fig_error_by_input.(png|pdf)      relative MAE vs the level of each input
  fig_response_slices.(png|pdf)     surrogate vs simulator along each input axis
  timing_table.csv                  simulator vs surrogate, measured
  dataset_splits_table.csv          split sizes and class balance
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s175 import data as sdata
from s175.columns import (FEATURE_COLS, OUTPUT_COLS, UNITS, FUEL_IDX,
                          CLASS_VALID, CLASS_ALL_NEG)
from s175.metrics import clamp_physical
from s175.splits import load_split

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"
FIG = ROOT / "figures"
OUT = ROOT / "results" / "evaluation"
CHUNK = 2_000_000

# import the checkpoint loader from the evaluator rather than duplicating it
sys.path.insert(0, str(ROOT / "scripts"))
ev = __import__("03_evaluate")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_fig(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"wrote figures/{stem}.png")


def load_model_bundle(run_dir, device):
    reg, reg_ck = ev.load_component(run_dir, "regressor", device)
    c1, c1_ck = ev.load_component(run_dir, "clf1", device)
    c2, c2_ck = ev.load_component(run_dir, "clf2", device)
    return ((reg, c1, c2), ev.Scaler(reg_ck["x_scaler"]), ev.Scaler(reg_ck["y_scaler"]))


@torch.no_grad()
def predict_reg(reg, X_rows, x_scaler, y_scaler, device, chunk=CHUNK):
    out = np.empty((len(X_rows), len(OUTPUT_COLS)), dtype=np.float32)
    for s in range(0, len(X_rows), chunk):
        xb = torch.from_numpy(x_scaler.transform(X_rows[s:s + chunk])).to(device)
        out[s:s + chunk] = reg(xb).float().cpu().numpy()
    return clamp_physical(y_scaler.inverse(out))


# ---------------------------------------------------------------------------
def fig_pred_vs_true(y_true, y_pred, cls_te):
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.8))
    defined = cls_te != CLASS_ALL_NEG
    valid = cls_te == CLASS_VALID
    for j, c in enumerate(OUTPUT_COLS):
        ax = axes[j // 5][j % 5]
        rows = valid if c == "fuel" else defined
        t, p = y_true[rows, j], y_pred[rows, j]
        hb = ax.hexbin(t, p, gridsize=60, bins="log", cmap="viridis",
                       mincnt=1, linewidths=0)
        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.set_title(f"{c} [{UNITS[c]}]  R²={r2:.4f}", fontsize=9)
        ax.tick_params(labelsize=7)
        if j % 5 == 0:
            ax.set_ylabel("surrogate", fontsize=8)
        if j // 5 == 1:
            ax.set_xlabel("simulator", fontsize=8)
    fig.colorbar(hb, ax=axes, shrink=0.8, label="rows per hex (log)")
    save_fig(fig, "fig_pred_vs_true")


def fig_error_by_input(X_te, y_true, y_pred, cls_te):
    """Relative MAE (% of output sd) at each grid level of each input."""
    defined = cls_te != CLASS_ALL_NEG
    valid = cls_te == CLASS_VALID
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5), sharey=True)
    cmap = plt.get_cmap("tab10")
    for k, feat in enumerate(FEATURE_COLS):
        ax = axes[k // 3][k % 3]
        col = X_te[:, k]
        levels = np.unique(col)
        # a swept input can have hundreds of levels; bin to <=12 for legibility
        if len(levels) > 12:
            edges = np.quantile(levels, np.linspace(0, 1, 13))
            centers = 0.5 * (edges[:-1] + edges[1:])
        else:
            edges, centers = None, levels
        for j, c in enumerate(OUTPUT_COLS):
            rows = valid if c == "fuel" else defined
            t, p, x = y_true[rows, j], y_pred[rows, j], col[rows]
            sd = t.std()
            ys = []
            for i, lv in enumerate(centers):
                if edges is None:
                    m = x == lv
                else:
                    m = (x >= edges[i]) & (x <= edges[i + 1])
                ys.append(100 * np.abs(p[m] - t[m]).mean() / sd if m.any() else np.nan)
            ax.plot(centers, ys, lw=1.1, color=cmap(j % 10), label=c)
        ax.set_title(feat, fontsize=10)
        ax.grid(alpha=0.3)
        if k % 3 == 0:
            ax.set_ylabel("MAE / σ(output)  [%]", fontsize=9)
    axes[0][0].set_yscale("log")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=10, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Where the error lives: relative MAE vs the level of each input "
                 "(S1_random test)", y=1.06)
    fig.tight_layout()
    save_fig(fig, "fig_error_by_input")


def find_slices(X):
    """For each input, the 1-D sweep through the grid point whose other eight
    inputs sit at their median grid level."""
    medians = np.empty(len(FEATURE_COLS), dtype=np.float32)
    level_list = []
    for k in range(len(FEATURE_COLS)):
        lv = np.unique(np.asarray(X[:1_000_000, k]))  # levels repeat; head is enough
        full = np.unique(np.asarray(X[:: max(1, len(X) // 5_000_000), k]))
        lv = full if len(full) > len(lv) else lv
        level_list.append(lv)
        medians[k] = lv[len(lv) // 2]
    slice_rows: list[list[int]] = [[] for _ in FEATURE_COLS]
    for s in range(0, len(X), CHUNK):
        xb = np.asarray(X[s:s + CHUNK])
        eq = xb == medians  # (n, 9)
        n_eq = eq.sum(axis=1)
        cand = np.flatnonzero(n_eq >= len(FEATURE_COLS) - 1)
        for r in cand:
            miss = np.flatnonzero(~eq[r])
            if len(miss) == 0:
                for k in range(len(FEATURE_COLS)):
                    slice_rows[k].append(s + r)
            else:
                slice_rows[miss[0]].append(s + r)
    return medians, [np.array(sorted(v)) for v in slice_rows]


def fig_response_slices(X, Y, cls, reg, x_scaler, y_scaler, device):
    medians, slices = find_slices(X)
    show = ["speed", "fuel", "roll"]
    idxs = [OUTPUT_COLS.index(c) for c in show]
    colors = {"speed": "#31538f", "fuel": "#b03a2e", "roll": "#1e8449"}
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5))
    for k, feat in enumerate(FEATURE_COLS):
        ax = axes[k // 3][k % 3]
        rows = slices[k]
        if len(rows) == 0:
            ax.axis("off")
            continue
        rows = rows[cls[rows] != CLASS_ALL_NEG]
        xv = np.asarray(X[rows])[:, k]
        order = np.argsort(xv)
        rows, xv = rows[order], xv[order]
        y_t = np.asarray(Y[rows])
        y_p = predict_reg(reg, np.asarray(X[rows]), x_scaler, y_scaler, device)
        ax2 = ax.twinx()
        for c, j in zip(show, idxs):
            target = ax2 if c == "roll" else ax
            tt = y_t[:, j].astype(float)
            if c == "fuel":
                tt[cls[rows] != CLASS_VALID] = np.nan  # no fuel target there
            target.plot(xv, tt, "o", ms=3.5, color=colors[c], alpha=0.55)
            target.plot(xv, y_p[:, j], "-", lw=1.4, color=colors[c],
                        label=f"{c} [{UNITS[c]}]")
        ax.set_title(f"sweep {feat} (others at median grid level)", fontsize=9)
        ax.tick_params(labelsize=8)
        # fixed physical scale: an auto-zoomed twin axis would magnify a
        # 0.04 deg roll error into a visually dominant gap on flat sweeps
        ax2.set_ylim(0, 3.6)
        ax2.tick_params(labelsize=8, colors=colors["roll"])
        ax.grid(alpha=0.3)
        if k == 0:
            lines = [plt.Line2D([], [], color=colors[c], lw=1.4,
                                label=f"{c} [{UNITS[c]}]") for c in show]
            lines += [plt.Line2D([], [], color="grey", marker="o", ls="", ms=4,
                                 label="simulator (markers)"),
                      plt.Line2D([], [], color="grey", lw=1.4,
                                 label="surrogate (line)")]
            fig.legend(handles=lines, loc="upper center", ncol=5, fontsize=9,
                       bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("Surrogate (lines) against the simulator (markers) along each "
                 "input axis; roll on the right-hand axis", y=1.06)
    fig.tight_layout()
    save_fig(fig, "fig_response_slices")


# ---------------------------------------------------------------------------
def timing_table(models, x_scaler, device):
    sim = json.loads((ROOT / "results" / "simulator_timing.json").read_text())
    sim_single = float(sim["single_point"]["median_s"])
    per_point = [b["per_point_s"] for b in sim["batch"] if b.get("per_point_s")]
    sim_batch_best = float(min(per_point))

    reg, c1, c2 = models
    x1 = torch.from_numpy(x_scaler.transform(
        np.random.default_rng(0).random((1, len(FEATURE_COLS)), dtype=np.float32))).to(device)
    xb = torch.from_numpy(x_scaler.transform(
        np.random.default_rng(0).random((65536, len(FEATURE_COLS)), dtype=np.float32))).to(device)

    @torch.no_grad()
    def full(x):
        y = reg(x)
        torch.sigmoid(c1(x))
        torch.sigmoid(c2(x))
        return y

    for _ in range(20):
        full(x1)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(200):
        full(x1)
    torch.cuda.synchronize() if device.type == "cuda" else None
    single_s = (time.perf_counter() - t0) / 200

    for _ in range(5):
        full(xb)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        full(xb)
    torch.cuda.synchronize() if device.type == "cuda" else None
    batch_s = (time.perf_counter() - t0) / (reps * len(xb))

    rows = [
        ["simulator, single point", sim_single, 1.0],
        ["simulator, batched (best measured)", sim_batch_best,
         sim_single / sim_batch_best],
        ["surrogate, single point (GPU, all 3 nets)", single_s,
         sim_single / single_s],
        ["surrogate, batched 65k (GPU, per point)", batch_s,
         sim_single / batch_s],
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "timing_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "seconds_per_point",
                    "speedup_vs_simulator_single"])
        for r in rows:
            w.writerow(r)
    log("timing: " + " | ".join(f"{r[0]}: {r[1]:.3g}s (x{r[2]:,.0f})" for r in rows))


def dataset_table():
    man = json.loads((SPLIT_DIR / "manifest.json").read_text()) if \
        (SPLIT_DIR / "manifest.json").exists() else {}
    with open(OUT / "dataset_splits_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "train", "val", "test", "test_fully_valid",
                    "test_fuel_only", "test_all_infeasible", "description"])
        for d in sorted(SPLIT_DIR.iterdir()):
            if not (d / "meta.json").exists():
                continue
            m = json.loads((d / "meta.json").read_text())
            cc = m.get("class_counts", {}).get("test", {})
            w.writerow([m["name"], m["counts"]["train"], m["counts"]["val"],
                        m["counts"]["test"], cc.get("fully_valid", ""),
                        cc.get("fuel_only", ""), cc.get("all_infeasible", ""),
                        m.get("description", "").replace("\n", " ")])
    log(f"wrote {OUT / 'dataset_splits_table.csv'}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="score only test rows disjoint from the screening "
                         "subsample (requires results/evaluation/"
                         "screening_rows.npy from 03_evaluate.py --clean)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / "v3_masked_peroutput" / "S1_random" / "seed_0"
    arr, _ = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    sp = load_split(SPLIT_DIR, "S1_random")
    models, x_scaler, y_scaler = load_model_bundle(run_dir, device)
    log(f"reference model {run_dir} on {device}")

    te = sp.test
    if args.clean:
        srows = np.load(ROOT / "results" / "evaluation" / "screening_rows.npy")
        in_sample = np.zeros(len(X), dtype=bool)
        in_sample[srows] = True
        n0 = len(te)
        te = te[~in_sample[te]]
        log(f"--clean: excluded {n0 - len(te):,} screening-overlap rows "
            f"({len(te):,} remain)")
    y_pred = predict_reg(models[0], np.asarray(X[te]), x_scaler, y_scaler, device)
    y_true = np.asarray(Y[te])
    X_te = np.asarray(X[te])
    cls_te = cls[te]
    log("test predictions ready")

    fig_pred_vs_true(y_true, y_pred, cls_te)
    fig_error_by_input(X_te, y_true, y_pred, cls_te)
    del X_te, y_true, y_pred
    fig_response_slices(X, Y, cls, models[0], x_scaler, y_scaler, device)
    timing_table(models, x_scaler, device)
    dataset_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
