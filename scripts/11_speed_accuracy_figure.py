"""Speed-versus-accuracy comparison figure: simulator and surrogate.

Every point is a measurement on des24 -- nothing estimated:
  simulator single point   8.939 s/pt   (median of 10 runs)
  simulator batched 864    16.59 ms/pt  (median of 3 runs, -CreateMetamodel)
  surrogate single point   2.646 ms/pt  (864 sequential calls, same inputs)
  surrogate batch 864      3.18 us/pt   (same 864 inputs as the simulator)
  surrogate batch ladder   512..262144  (inference_speed.json)

Accuracy (y) is the WORST output R^2, the conservative choice:
  held-out test split          min R^2 = 0.99940 (fuel)
  vs fresh off-grid sim runs   min R^2 = 0.99592 (slamming, midpoints)
The simulator is the reference the R^2 is computed against, so it sits at
R^2 = 1 by definition -- drawn as such, not presented as a measurement.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BLUE, LBLUE, GRAY, RED = "#2a78d6", "#7fb3e8", "#4a4a4a", "#d11a2a"

speed = json.loads((ROOT / "figures/fixed_campaign/inference_speed.json").read_text())
same = json.loads((ROOT / "results/evaluation/same_inputs_timing.json").read_text())

# surrogate per-point costs [s], all measured, sorted slow -> fast
sur_t = [same["ml_single_per_point_ms"] / 1e3,          # single point
         same["ml_batch864_per_point_us"] / 1e6]        # batch 864 (same inputs)
sur_t += [1.0 / r for r in speed["rows_per_s"]["pipeline"]]  # ladder 512..262144
sur_t = sorted(sur_t, reverse=True)
R2_TEST, R2_OFFGRID = 0.99940, 0.99592
sim_single, sim_batch = 8.939, 16.59e-3

fig, ax = plt.subplots(figsize=(10, 6.2))
# simulator reference
ax.scatter([sim_single, sim_batch], [1.0, 1.0], s=90, marker="s", color=GRAY, zorder=3)
ax.annotate("simulator, single point\n8.94 s/pt", (sim_single, 1.0),
            textcoords="offset points", xytext=(0, -30), ha="center", fontsize=9, color=GRAY)
ax.annotate("simulator, batched\n16.6 ms/pt", (sim_batch, 1.0),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color=GRAY)
ax.axhline(1.0, color=GRAY, lw=0.8, ls=":", alpha=0.6)
# surrogate curves: same measured costs, two accuracy references
ax.plot(sur_t, [R2_TEST] * len(sur_t), "-o", color=BLUE, lw=2, ms=7, zorder=3,
        label="surrogate -- held-out test (worst output $R^2$)")
ax.plot(sur_t, [R2_OFFGRID] * len(sur_t), "--o", color=LBLUE, lw=2, ms=7, zorder=3,
        label="surrogate -- vs fresh off-grid simulator runs (worst output $R^2$)")
ax.annotate("single point\n2.65 ms/pt", (sur_t[0], R2_TEST), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=9, color=BLUE)
ax.annotate("batched\n0.25 $\\mu$s/pt", (sur_t[-1], R2_TEST), textcoords="offset points",
            xytext=(10, 12), ha="center", fontsize=9, color=BLUE)
# speed-up arrows, measured pairs
for x0, x1, y, text in [
        (sim_single, sur_t[0], 0.9975, "$\\times$3,378\n(single vs single)"),
        (sim_batch, same["ml_batch864_per_point_us"] / 1e6, 0.99655,
         "$\\times$5,226\n(same 864 points)")]:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate(text, ((x0 * x1) ** 0.5, y), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9, color=RED)
ax.set_xscale("log")
ax.set_xlim(2e-8, 80)
ax.set_ylim(0.9945, 1.0012)
ax.invert_xaxis()   # faster to the right
ax.set_xlabel("measured wall time per evaluated point [s]  (log scale, faster $\\rightarrow$)",
              fontsize=11)
ax.set_ylabel("worst-output $R^2$  (axis truncated at 0.9945)", fontsize=11)
ax.set_title("Accuracy versus per-point cost -- all points measured on the same host",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out = ROOT / "figures/fixed_campaign/fig_speed_vs_accuracy"
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print("wrote", out)
