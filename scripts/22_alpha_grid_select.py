"""Apply the prespecified selection rule to the loss-weight grid.

The rule, fixed before the grid results existed:

1.  Eligibility. In every seed, each output errs on its safe side on the
    selection partition: speed under-predicted in more than half of its
    positive-target rows, each of the other nine over-predicted in more than
    half of theirs (per_output's safe_direction_met).
2.  Among eligible pairs, the lowest mean (over seeds) of the mean (over the
    ten outputs) MAE expressed as % of the output's selection-partition SD.
3.  Ties -- exact equality of the unrounded means -- break by the higher mean
    R^2 over seeds.
4.  If nothing is eligible, the search grid is extended with the
    EXTENSION_ALPHA_OTHER values below; the rule itself is never relaxed.

Writes results/alpha_grid/selection.json and grid_summary.csv (one row per
pair: mean and sd over seeds of the selection metrics).

  python scripts/22_alpha_grid_select.py [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "S0_screen"

# Prespecified extension path, used only if no pair is eligible.
EXTENSION_ALPHA_OTHER = [0.25, 0.2]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    want_seeds = sorted(args.seeds)

    evals = {}
    for p in sorted(ROOT.glob(f"runs/grid_a*/{SPLIT}/seed_*/selection_eval.json")):
        e = json.loads(p.read_text())
        evals.setdefault((e["alpha_speed"], e["alpha_other"]), {})[e["seed"]] = e

    if not evals:
        log("no selection_eval.json found -- run 21_alpha_grid_eval.py first")
        return 1

    incomplete = {k: sorted(set(want_seeds) - set(v))
                  for k, v in evals.items()
                  if set(want_seeds) - set(v)}
    if incomplete:
        for k, missing in sorted(incomplete.items()):
            log(f"  incomplete: alpha=({k[0]:g}, {k[1]:g}) missing seeds {missing}")
        log(f"{len(incomplete)} pairs incomplete -- selection needs every seed")
        return 1

    rows = []
    for (a_s, a_o), by_seed in sorted(evals.items()):
        seeds = [by_seed[s] for s in want_seeds]
        safe_all = all(e["safe_direction"] == "10/10" for e in seeds)
        rel = np.array([e["mean_rel_MAE_sd_pct"] for e in seeds])
        r2 = np.array([e["mean_R2"] for e in seeds])
        oe = np.array([e["per_output"]["speed"]["overpred_pct_positive"]
                       for e in seeds])
        rows.append({
            "alpha_speed": a_s, "alpha_other": a_o,
            "eligible": safe_all,
            "safe_by_seed": [e["safe_direction"] for e in seeds],
            "mean_rel_MAE_sd_pct": float(rel.mean()),
            "sd_rel_MAE_sd_pct": float(rel.std(ddof=1)) if len(rel) > 1 else 0.0,
            "mean_R2": float(r2.mean()),
            "sd_R2": float(r2.std(ddof=1)) if len(r2) > 1 else 0.0,
            "speed_OE_mean": float(oe.mean()),
            "speed_OE_sd": float(oe.std(ddof=1)) if len(oe) > 1 else 0.0,
        })

    out_dir = ROOT / "results" / "alpha_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "grid_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "safe_by_seed"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {out_dir / 'grid_summary.csv'} ({len(rows)} pairs)")

    eligible = [r for r in rows if r["eligible"]]
    log(f"{len(eligible)} of {len(rows)} pairs eligible "
        f"(all ten outputs on their safe side in every seed)")
    if not eligible:
        result = {
            "selected": None,
            "reason": "no pair eligible",
            "next_step": {"extend_alpha_other": EXTENSION_ALPHA_OTHER,
                          "note": "extend the grid; the rule is not relaxed"},
            "seeds": want_seeds,
            "grid": rows,
        }
        (out_dir / "selection.json").write_text(json.dumps(result, indent=2))
        log("no eligible pair -- extend the grid with "
            f"alpha_other in {EXTENSION_ALPHA_OTHER}")
        return 2

    best_mae = min(r["mean_rel_MAE_sd_pct"] for r in eligible)
    leaders = [r for r in eligible if r["mean_rel_MAE_sd_pct"] == best_mae]
    tie_broken = len(leaders) > 1
    chosen = max(leaders, key=lambda r: r["mean_R2"])

    result = {
        "selected": {"alpha_speed": chosen["alpha_speed"],
                     "alpha_other": chosen["alpha_other"]},
        "rule": ("eligible = all ten outputs on their safe side in every seed; "
                 "then lowest mean standardised MAE; ties by mean R^2"),
        "tie_broken_by_R2": tie_broken,
        "seeds": want_seeds,
        "selected_metrics": chosen,
        "eligible": sorted(
            ({k: r[k] for k in ("alpha_speed", "alpha_other",
                                "mean_rel_MAE_sd_pct", "sd_rel_MAE_sd_pct",
                                "mean_R2", "speed_OE_mean", "speed_OE_sd")}
             for r in eligible),
            key=lambda r: r["mean_rel_MAE_sd_pct"]),
        "grid": rows,
    }
    (out_dir / "selection.json").write_text(json.dumps(result, indent=2))
    log(f"selected: alpha_speed={chosen['alpha_speed']:g} "
        f"alpha_other={chosen['alpha_other']:g}  "
        f"(mean MAE {chosen['mean_rel_MAE_sd_pct']:.3f}% of SD, "
        f"mean R2 {chosen['mean_R2']:.5f}, "
        f"speed OE {chosen['speed_OE_mean']:.1f}"
        f"+/-{chosen['speed_OE_sd']:.1f}%)")
    log(f"wrote {out_dir / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
