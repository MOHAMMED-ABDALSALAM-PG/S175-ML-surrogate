"""Per-point cost comparison figure: simulator vs surrogate (dot plot).

Every value is a measurement on des24 -- nothing estimated:
  simulator single point   8.939 s/pt   (median of 10 runs)
  simulator batched 864    16.59 ms/pt  (median of 3 runs, -CreateMetamodel)
  surrogate single point   2.646 ms/pt  (864 sequential calls, same inputs)
  surrogate batch 864      3.18 us/pt   (same 864 inputs as the simulator)
  surrogate batch 65,536   0.25 us/pt   (best measured, timing_table.csv)

Form: horizontal dot plot on a log time axis (bars would length-encode a log
quantity, which misreads). One row per measured configuration, direct value
labels, speed-up factors as a right-hand annotation column, accuracy stated
in the subtitle -- it is constant across all rows, so it is not an axis.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BLUE, DARKBLUE = "#2a78d6", "#0b3a75"
GRAY, INK, MUTED = "#4a4a4a", "#0b0b0b", "#52514e"

same = json.loads((ROOT / "results/evaluation/same_inputs_timing.json").read_text())
with (ROOT / "results/evaluation/timing_table.csv").open() as f:
    timing = {r["configuration"]: float(r["seconds_per_point"])
              for r in csv.DictReader(f)}

sim_single = same["sim_single_per_point_s"]
sim_batch = same["sim_batch864_per_point_ms"] / 1e3
sur_single = same["ml_single_per_point_ms"] / 1e3
sur_batch864 = same["ml_batch864_per_point_us"] / 1e6
sur_batch65k = timing["surrogate, batched 65k (GPU, per point)"]

def fmt_speedup(factor: float) -> str:
    if factor >= 1e6:
        exp = len(f"{factor:.0f}") - 1
        return f"$\\times{factor / 10 ** exp:.1f}\\times10^{{{exp}}}$"
    return f"$\\times${factor:,.0f}"


rows = [  # (label, seconds per point, is_surrogate, value text, speed-up text)
    ("Simulator\nsingle point", sim_single, False, "8.94 s", "reference"),
    ("Simulator\nbatched, 864 points", sim_batch, False, "16.6 ms",
     fmt_speedup(sim_single / sim_batch)),
    ("Surrogate\nsingle point, all 3 networks", sur_single, True, "2.65 ms",
     fmt_speedup(sim_single / sur_single)),
    ("Surrogate\nbatched, same 864 points", sur_batch864, True,
     "3.18 $\\mu$s", fmt_speedup(sim_single / sur_batch864)),
    ("Surrogate\nbatched 65,536 (best)", sur_batch65k, True, "0.25 $\\mu$s",
     fmt_speedup(sim_single / sur_batch65k)),
]

fig, ax = plt.subplots(figsize=(11, 4.6))
ys = range(len(rows))
for y, (label, t, is_sur, vtext, _stext) in zip(ys, rows):
    color = BLUE if is_sur else GRAY
    marker = "o" if is_sur else "s"
    ax.plot([t], [y], marker, color=color, ms=10, zorder=3)
    ax.annotate(vtext, (t, y), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=10, color=INK)

# right-hand annotation column: measured speed-up vs the single-point simulator
ax.annotate("speed-up vs\nsimulator, single point", (1.005, 1.0),
            xycoords="axes fraction", ha="left", va="bottom",
            fontsize=9, color=MUTED)
for y, (_l, _t, _s, _v, stext) in zip(ys, rows):
    ax.annotate(stext, (1.005, y), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=10,
                color=MUTED if y == 0 else INK)

ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=10)
ax.invert_yaxis()
ax.set_ylim(len(rows) - 0.4, -0.75)
ax.set_xscale("log")
ax.set_xlim(6e-8, 60)
ticks = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
ax.set_xticks(ticks, ["0.1 $\\mu$s", "1 $\\mu$s", "10 $\\mu$s",
                      "100 $\\mu$s", "1 ms", "10 ms", "100 ms", "1 s", "10 s"])
ax.set_xlabel("measured wall time per evaluated operating point (log scale)",
              fontsize=11)
ax.grid(axis="x", which="major", alpha=0.3)
ax.grid(axis="y", alpha=0.15)
ax.tick_params(axis="x", which="minor", bottom=False)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
# no legend box: every row label already names its entity, so identity is
# carried by text, not color alone
ax.set_title(
    "Measured cost per evaluated operating point -- same host\n", fontsize=13,
    fontweight="bold", loc="left")
ax.annotate("accuracy unchanged: worst-output $R^2$ = 0.9994 (test), "
            "0.9959 (off-grid)",
            (0, 1.03), xycoords="axes fraction", ha="left", fontsize=10,
            color=MUTED)
fig.tight_layout()
out = ROOT / "figures/fig_speed_vs_accuracy"
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print("wrote", out)
