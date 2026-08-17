"""Build the aggregate tables and figures from the per-run eval.json files.

Run after 03_evaluate.py:
  python scripts/04_report.py

Outputs:
  results/evaluation/regressor_table.csv    per split/seed/output metrics
  results/evaluation/classifier_table.csv   both classifiers, all runs
  results/evaluation/end_to_end_table.csv   3-class surrogate performance
  results/evaluation/s1_seed_spread.json    seed spread on S1_random
  figures/fig_r2_heatmap.(png|pdf)          R2 per split x output
  figures/fig_clf_fnr.(png|pdf)             classifier FN rates with 95% CI
  figures/fig_e2e_confusion.(png|pdf)       3-class confusion per split
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s175.columns import OUTPUT_COLS
from s175.metrics import aggregate_seeds

RUNS = ROOT / "runs" / "v3_masked_peroutput"
OUT = ROOT / "results" / "evaluation"
FIG = ROOT / "figures"

SPLIT_ORDER = ["S1_random", "S2_level_Hs", "S2_level_Chi", "S2_level_draft",
               "S2_level_Vwind", "S3_block", "S4_corner"]


def save_fig(fig, stem: str):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{stem}.png")


def load_evals(eval_name: str = "eval.json") -> list[dict]:
    evals = []
    for p in sorted(RUNS.glob(f"*/seed_*/{eval_name}")):
        evals.append(json.loads(p.read_text()))
    if not evals:
        raise SystemExit(f"no {eval_name} files -- run 03_evaluate.py first")
    evals.sort(key=lambda e: (SPLIT_ORDER.index(e["split"]), e["seed"]))
    return evals


def write_tables(evals):
    OUT.mkdir(parents=True, exist_ok=True)
    reg_fields = ["MAE", "RMSE", "R2", "bias_mean_error", "rel_MAE_sd_pct",
                  "overpred_pct_positive", "MAPE_positive_pct", "sMAPE_pct",
                  "safe_direction", "safe_direction_met", "pct_below_zero"]
    with open(OUT / "regressor_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "seed", "output"] + reg_fields)
        for e in evals:
            for c in OUTPUT_COLS:
                m = e["regressor"]["per_output"][c]
                w.writerow([e["split"], e["seed"], c] + [m[k] for k in reg_fields])

    clf_fields = ["recall", "precision", "f1", "accuracy", "false_negative_rate",
                  "false_negative_rate_ci95", "FN", "n_positive", "n_total"]
    with open(OUT / "classifier_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "seed", "classifier", "threshold"] + clf_fields)
        for e in evals:
            for name in ("clf1", "clf2"):
                m = e[name]
                w.writerow([e["split"], e["seed"], name, e["thresholds"][name]]
                           + [m[k] for k in clf_fields])

    with open(OUT / "end_to_end_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "seed", "accuracy", "balanced_accuracy", "macro_f1",
                    "recall_valid", "recall_fuel_only", "recall_all_infeasible",
                    "dangerous_allneg_passed", "dangerous_allneg_rate",
                    "dangerous_fuel_passed", "dangerous_fuel_rate",
                    "n_test_rows"])
        for e in evals:
            z = e["end_to_end"]
            rec = z.get("per_class_recall", [float("nan")] * 3)
            w.writerow([e["split"], e["seed"], z["accuracy"],
                        z.get("balanced_accuracy", float("nan")),
                        z.get("macro_f1", float("nan")),
                        rec[0], rec[1], rec[2],
                        z["dangerous_allneg_passed"], z["dangerous_allneg_rate"],
                        z["dangerous_fuel_passed"], z["dangerous_fuel_rate"],
                        e["n_test_rows"]])
    print(f"wrote 3 tables under {OUT}")

    s1 = [e["regressor"]["per_output"] for e in evals if e["split"] == "S1_random"]
    if len(s1) > 1:
        spread = {m: aggregate_seeds(s1, metric=m) for m in ("R2", "MAE", "RMSE")}
        (OUT / "s1_seed_spread.json").write_text(json.dumps(spread, indent=2))
        print("wrote s1_seed_spread.json "
              f"(n_seeds={len(s1)})")


def fig_r2_heatmap(evals):
    """R2 per split x output; S1_random shown as the mean over its seeds."""
    splits = [s for s in SPLIT_ORDER if any(e["split"] == s for e in evals)]
    mat = np.full((len(splits), len(OUTPUT_COLS)), np.nan)
    for i, s in enumerate(splits):
        rows = [e for e in evals if e["split"] == s]
        for j, c in enumerate(OUTPUT_COLS):
            mat[i, j] = np.mean([e["regressor"]["per_output"][c]["R2"] for e in rows])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0.9, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(OUTPUT_COLS)), OUTPUT_COLS, rotation=35, ha="right")
    labels = [s + (f" (n={sum(e['split'] == s for e in evals)})"
                   if sum(e["split"] == s for e in evals) > 1 else "")
              for s in splits]
    ax.set_yticks(range(len(splits)), labels)
    for i in range(len(splits)):
        for j in range(len(OUTPUT_COLS)):
            # single-hue white->blue scale: dark ink on pale (low) cells,
            # white ink once the cell is deep blue (high end of the scale)
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if mat[i, j] > 0.965 else "black")
    ax.set_title("Test-set R² per output and holdout regime (clamped predictions)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="R² (floor of scale = 0.90)")
    save_fig(fig, "fig_r2_heatmap")


def fig_clf_fnr(evals):
    splits = [s for s in SPLIT_ORDER if any(e["split"] == s for e in evals)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, name, title in ((axes[0], "clf1", "C1: completely infeasible"),
                            (axes[1], "clf2", "C2: fuel-only infeasible")):
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(splits):
            rows = [e for e in evals if e["split"] == s]
            for e in rows:
                m = e[name]
                xs.append(i)
                ys.append(m["false_negative_rate"])
                lo.append(m["false_negative_rate"] - m["false_negative_rate_ci95"][0])
                hi.append(m["false_negative_rate_ci95"][1] - m["false_negative_rate"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3, color="#31538f")
        ax.axhline(0.01, color="#b03a2e", ls="--", lw=1,
                   label="1% (validation recall floor 0.99)")
        ax.set_xticks(range(len(splits)), splits, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("test false-negative rate (log scale, Wilson 95% CI)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Feasibility classifiers at their validation-selected thresholds")
    save_fig(fig, "fig_clf_fnr")


def fig_e2e_confusion(evals):
    """Row-normalised 3-class confusion per split (seed 0)."""
    picks = []
    for s in SPLIT_ORDER:
        rows = [e for e in evals if e["split"] == s and e["seed"] == 0]
        if rows:
            picks.append(rows[0])
    cols = 4
    rows_n = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.5 * cols, 3.3 * rows_n),
                             squeeze=False)
    short = ["valid", "fuel-only", "all-infeas"]
    im = None
    for k, e in enumerate(picks):
        ax = axes[k // cols][k % cols]
        conf = np.array(e["end_to_end"]["confusion_matrix_true_x_pred"], dtype=float)
        rs = conf.sum(axis=1, keepdims=True)
        norm = np.divide(conf, rs, out=np.zeros_like(conf), where=rs > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(3), short, fontsize=8)
        ax.set_yticks(range(3), short, fontsize=8)
        ax.set_xlabel("surrogate", fontsize=8)
        ax.set_ylabel("simulator", fontsize=8)
        ax.set_title(f"{e['split']}  (acc {e['end_to_end']['accuracy']:.4f})",
                     fontsize=9)
    for k in range(len(picks), rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.7, label="row-normalised rate")
    fig.suptitle("Assembled surrogate: three-class feasibility decision (seed 0)")
    save_fig(fig, "fig_e2e_confusion")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="build tables/figures from eval_clean.json (test rows "
                         "disjoint from the screening subsample -- the paper's "
                         "primary evaluation)")
    args = ap.parse_args()
    evals = load_evals("eval_clean.json" if args.clean else "eval.json")
    print(f"{len(evals)} evaluated runs"
          + (" (screening-disjoint)" if args.clean else ""))
    write_tables(evals)
    fig_r2_heatmap(evals)
    fig_clf_fnr(evals)
    fig_e2e_confusion(evals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
