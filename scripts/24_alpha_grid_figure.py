"""Loss-weight grid figure for the manuscript (companion to tab:alphagrid).

Two heatmap panels over the same 5 x 6 grid, from results/alpha_grid/
selection.json: (a) the worst seed's count of outputs on their safe side,
(b) the mean standardised MAE over seeds. The selected pair is outlined.

  python scripts/24_alpha_grid_figure.py
Out:  figures/fig_alpha_grid.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "alpha_grid" / "selection.json"

ALPHA_SPEED = [1.0, 1.5, 2.0, 3.0, 5.0]      # columns
ALPHA_OTHER = [1.5, 1.0, 0.8, 0.67, 0.5, 0.33]  # rows, toward safety
EDGE = "#1c2b3a"


def fmt(a: float) -> str:
    return f"{a:g}"


def main() -> int:
    sel = json.loads(SRC.read_text())
    by_pair = {(g["alpha_speed"], g["alpha_other"]): g for g in sel["grid"]}
    picked = (sel["selected"]["alpha_speed"], sel["selected"]["alpha_other"])

    safe = np.array([[min(int(s.split("/")[0])
                          for s in by_pair[(sp, ao)]["safe_by_seed"])
                      for sp in ALPHA_SPEED] for ao in ALPHA_OTHER])
    mae = np.array([[by_pair[(sp, ao)]["mean_rel_MAE_sd_pct"]
                     for sp in ALPHA_SPEED] for ao in ALPHA_OTHER])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    panels = [
        (axes[0], safe, 0.0, 10.0, "{:.0f}", 6.5,
         "(a) Worst seed: outputs on their safe side",
         "outputs safe-side, of 10 (worst of 5 seeds)"),
        (axes[1], mae, 1.6, 2.3, "{:.2f}", 2.06,
         "(b) Mean standardised MAE over five seeds",
         "mean MAE [% of output SD]"),
    ]
    for ax, mat, vmin, vmax, cellfmt, ink_switch, title, cblabel in panels:
        im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ALPHA_SPEED)), [fmt(a) for a in ALPHA_SPEED])
        ax.set_yticks(range(len(ALPHA_OTHER)), [fmt(a) for a in ALPHA_OTHER])
        ax.set_xlabel(r"$\alpha_\mathrm{speed}$")
        ax.set_ylabel(r"$\alpha_\mathrm{others}$")
        ax.set_title(title, fontsize=10)
        for i in range(len(ALPHA_OTHER)):
            for j in range(len(ALPHA_SPEED)):
                # single-hue white->blue scale: dark ink on pale (low) cells,
                # white ink once the cell is deep blue (high end of the scale)
                ax.text(j, i, cellfmt.format(mat[i, j]), ha="center",
                        va="center", fontsize=8.5,
                        color="white" if mat[i, j] > ink_switch else "black")
        i = ALPHA_OTHER.index(picked[1])
        j = ALPHA_SPEED.index(picked[0])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=EDGE, linewidth=2.2))
        fig.colorbar(im, ax=ax, shrink=0.9, label=cblabel)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig_alpha_grid.{ext}",
                    dpi=300, bbox_inches="tight")
    print("wrote figures/fig_alpha_grid.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
