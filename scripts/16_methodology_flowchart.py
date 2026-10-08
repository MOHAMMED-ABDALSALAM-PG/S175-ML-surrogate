"""Methodology flowchart (paper Figure 2), generated instead of hand-drawn.

The previous hand-made diagram showed only the uniform alpha search and a
final configuration of "masked, asymmetric, alpha = 1.5"; it predated the
joint loss-weight grid and the selected configuration (speed 1.5, others
0.33). Generating the figure keeps it in lockstep with the method.

  python scripts/16_methodology_flowchart.py
Out:  figures/fig_methodology_flowchart.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

INK = "#1c2b3a"
EDGE = "#31538f"
FILL = "#eef3f9"
FILL_STAGE = "#dbe7f4"
FILL_FINAL = "#31538f"


def box(ax, x, y, w, h, text, fill=FILL, edge=EDGE, ink=INK, fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                linewidth=1.1, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=ink, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color=EDGE, shrinkA=1, shrinkB=1))


def main() -> int:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- top row: data pipeline -----------------------------------------
    box(ax, 0.015, 0.80, 0.29, 0.15,
        "S175 simulator\nfull factorial dataset: 126,153,720 cases\n"
        "($H_s$ refined to 0.25 m, $\\chi$ to 5$^\\circ$)", bold=True)
    box(ax, 0.355, 0.80, 0.29, 0.15,
        "Stratified split over distinct\ninput combinations (80/10/10)\n"
        "+ structured holdout regimes")
    box(ax, 0.695, 0.80, 0.29, 0.15,
        "Model-family screening on a\nstratified 5% subsample:\n"
        "RF / XGBoost / MLP $\\rightarrow$ MLP selected")
    arrow(ax, 0.305, 0.875, 0.353, 0.875)
    arrow(ax, 0.645, 0.875, 0.693, 0.875)

    # ---- stage 1: feasibility classification ----------------------------
    box(ax, 0.015, 0.42, 0.30, 0.30,
        "Stage 1 \u2014 feasibility classification\n\n"
        "$C_1$: complete infeasibility\n(all outputs $= -1$)\n\n"
        "$C_2$: fuel-only infeasibility\n(only fuel $= -1$)\n\n"
        "thresholds $\\tau_1, \\tau_2$: most selective with\n"
        "validation recall $\\geq 0.99$", fill=FILL_STAGE)

    # ---- stage 2: regression + loss development -------------------------
    box(ax, 0.355, 0.60, 0.63, 0.12,
        "Stage 2 \u2014 multi-output MLP regressor (512-256-128), "
        "masked asymmetric loss\ntrained on fully valid + fuel-only "
        "infeasible cases; fuel term masked where undefined",
        fill=FILL_STAGE)
    box(ax, 0.355, 0.44, 0.30, 0.115,
        "Joint loss-weight grid (subsample):\n"
        "$\\alpha_\\mathrm{speed} \\in \\{1.0, 1.5, 2.0, 3.0, 5.0\\} \\times "
        "\\alpha_\\mathrm{other} \\in \\{1.5, ..., 0.33\\}$\n"
        "30 pairs $\\times$ 5 seeds, trained jointly")
    box(ax, 0.685, 0.44, 0.30, 0.115,
        "Prespecified rule: all ten outputs safe-side\n"
        "in every seed, then lowest standardised MAE\n"
        "$\\rightarrow$ only $(1.5, 0.33)$ eligible")
    box(ax, 0.355, 0.28, 0.63, 0.115,
        "Final regressor: masked loss, per-output weights "
        "($\\alpha_\\mathrm{speed} = 1.5$, $\\alpha_\\mathrm{other} = 0.33$)\n"
        "every output biased toward its safe error side", bold=True)

    arrow(ax, 0.84, 0.80, 0.67, 0.723)      # screening -> stage 2
    arrow(ax, 0.42, 0.80, 0.20, 0.725)      # split -> stage 1
    arrow(ax, 0.505, 0.60, 0.505, 0.558)    # stage 2 -> grid
    arrow(ax, 0.657, 0.497, 0.683, 0.497)   # grid -> selection rule
    arrow(ax, 0.835, 0.44, 0.75, 0.398)     # selection rule -> final

    # ---- assembly and evaluation -----------------------------------------
    box(ax, 0.015, 0.09, 0.47, 0.13,
        "Assembled two-stage surrogate\n"
        "$C_1 \\geq \\tau_1 \\rightarrow$ all $-1$;  else regressor;  "
        "$C_2 \\geq \\tau_2 \\rightarrow$ fuel $= -1$",
        fill=FILL_FINAL, edge=FILL_FINAL, ink="white", bold=True)
    box(ax, 0.515, 0.09, 0.47, 0.13,
        "Evaluation: held-out test set $\\cdot$ structured holdouts\n"
        "$\\cdot$ fresh off-grid simulator runs $\\cdot$ interpolation baseline")
    arrow(ax, 0.165, 0.42, 0.20, 0.222)     # stage 1 -> assembled
    arrow(ax, 0.60, 0.28, 0.42, 0.222)      # final regressor -> assembled
    arrow(ax, 0.485, 0.155, 0.513, 0.155)   # assembled -> evaluation

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig_methodology_flowchart.{ext}",
                    dpi=300, bbox_inches="tight")
    print("wrote figures/fig_methodology_flowchart.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
