"""Emit the manuscript's loss-weight grid table from the grid selection.

Reads results/alpha_grid/selection.json (22_alpha_grid_select.py) and writes,
under results/evaluation/latex/:

  table_alpha_grid.tex    tab:alphagrid  (5 x 6 joint grid, two panels:
                          worst-seed safe-side count and mean standardised MAE)
  alpha_grid_numbers.json every in-text number the selection story cites

  python scripts/23_alpha_grid_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "alpha_grid" / "selection.json"
DEST = ROOT / "results" / "evaluation" / "latex"

ALPHA_SPEED = [1.0, 1.5, 2.0, 3.0, 5.0]
ALPHA_OTHER = [1.5, 1.0, 0.8, 0.67, 0.5, 0.33]  # row order: toward safety


def fmt_alpha(a: float) -> str:
    return f"{a:g}"


def main() -> None:
    sel = json.loads(SRC.read_text())
    selected = sel["selected"]
    by_pair = {(g["alpha_speed"], g["alpha_other"]): g for g in sel["grid"]}
    assert len(by_pair) == len(ALPHA_SPEED) * len(ALPHA_OTHER)

    def worst_safe(g: dict) -> int:
        return min(int(s.split("/")[0]) for s in g["safe_by_seed"])

    def cell(g: dict, text: str) -> str:
        picked = (g["alpha_speed"] == selected["alpha_speed"]
                  and g["alpha_other"] == selected["alpha_other"])
        return f"\\textbf{{{text}}}" if picked else text

    header = " & ".join(fmt_alpha(a) for a in ALPHA_SPEED)
    ncols = 1 + len(ALPHA_SPEED)
    lines = [
        f"\\begin{{tabular}}{{c{'c' * len(ALPHA_SPEED)}}}",
        "\\hline",
        f" & \\multicolumn{{{len(ALPHA_SPEED)}}}{{c}}{{$\\alpha_\\mathrm{{speed}}$}} \\\\",
        f"$\\alpha_\\mathrm{{others}}$ & {header} \\\\",
        "\\hline",
        f"\\multicolumn{{{ncols}}}{{l}}{{\\emph{{(a) Worst seed: outputs on their safe side, of 10}}}} \\\\",
    ]
    for ao in ALPHA_OTHER:
        cells = [cell(by_pair[(sp, ao)], str(worst_safe(by_pair[(sp, ao)])))
                 for sp in ALPHA_SPEED]
        lines.append(f"{fmt_alpha(ao)} & {' & '.join(cells)} \\\\")
    lines += [
        "\\hline",
        f"\\multicolumn{{{ncols}}}{{l}}{{\\emph{{(b) Mean standardised MAE over five seeds [\\% of SD]}}}} \\\\",
    ]
    for ao in ALPHA_OTHER:
        cells = [cell(by_pair[(sp, ao)],
                      f"{by_pair[(sp, ao)]['mean_rel_MAE_sd_pct']:.2f}")
                 for sp in ALPHA_SPEED]
        lines.append(f"{fmt_alpha(ao)} & {' & '.join(cells)} \\\\")
    lines += ["\\hline", "\\end{tabular}"]

    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "table_alpha_grid.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")

    g_sel = by_pair[(selected["alpha_speed"], selected["alpha_other"])]
    mae_all = {f"({fmt_alpha(sp)}, {fmt_alpha(ao)})":
               by_pair[(sp, ao)]["mean_rel_MAE_sd_pct"]
               for sp in ALPHA_SPEED for ao in ALPHA_OTHER}
    best_pair = min(mae_all, key=mae_all.get)
    numbers = {
        "selected": selected,
        "n_pairs": len(by_pair),
        "n_seeds": len(sel["seeds"]),
        "n_eligible": len(sel["eligible"]),
        "tie_broken_by_R2": sel["tie_broken_by_R2"],
        "selected_worst_safe": worst_safe(g_sel),
        "selected_mean_rel_MAE_sd_pct": g_sel["mean_rel_MAE_sd_pct"],
        "selected_sd_rel_MAE_sd_pct": g_sel["sd_rel_MAE_sd_pct"],
        "selected_mean_R2": g_sel["mean_R2"],
        "selected_speed_OE_mean": g_sel["speed_OE_mean"],
        "selected_speed_OE_sd": g_sel["speed_OE_sd"],
        "grid_best_MAE_pair": best_pair,
        "grid_best_MAE": mae_all[best_pair],
        "grid_MAE_range": [min(mae_all.values()), max(mae_all.values())],
        "grid_min_mean_R2": min(g["mean_R2"] for g in by_pair.values()),
        "runners_up_worst_safe_9": sorted(
            f"({fmt_alpha(g['alpha_speed'])}, {fmt_alpha(g['alpha_other'])})"
            for g in by_pair.values() if worst_safe(g) == 9),
    }
    out_json = DEST / "alpha_grid_numbers.json"
    out_json.write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"wrote {out_json}")
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
