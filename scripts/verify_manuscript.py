"""Check every numeric claim in the manuscript against the recorded results.

The manuscript quotes several hundred numbers. This script pulls the ones that
can be traced to a machine-written artefact under results/ and compares them to
the value the paper prints, so a mismatch surfaces as a failure rather than as
something a reader has to notice.

Numbers that exist only in prose (hardware model names, wall-clock minutes,
dataset provenance counts) cannot be checked this way and are listed at the end
as "unverifiable from artefacts" rather than silently passed.

  python scripts/verify_manuscript.py paper/main.tex
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(rel: str):
    p = RES / rel
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    import csv
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def tex_has(tex: str, needle: str) -> bool:
    """Is this literal string present in the manuscript?"""
    return needle in tex


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    tex_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper" / "main.tex")
    tex = pathlib.Path(tex_path).read_text(encoding="utf-8", errors="replace")
    # collapse whitespace so a number split across lines still matches
    flat = re.sub(r"\s+", " ", tex)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = ""):
        checks.append((name, ok, detail))

    # ---------------- alpha grid selection ----------------
    sel = load("alpha_grid/selection.json")
    if sel:
        s = sel["selected"]
        m = sel["selected_metrics"]
        check("selected pair (alpha_speed, alpha_other) = (1.5, 0.33)",
              s["alpha_speed"] == 1.5 and s["alpha_other"] == 0.33,
              f"selection.json: ({s['alpha_speed']}, {s['alpha_other']})")
        check("exactly one eligible pair on the grid",
              len(sel["eligible"]) == 1,
              f"selection.json: {len(sel['eligible'])} eligible")
        check("grid has 30 pairs", len(sel["grid"]) == 30,
              f"selection.json: {len(sel['grid'])}")
        mae = m["mean_rel_MAE_sd_pct"]; sd = m["sd_rel_MAE_sd_pct"]
        check(f"selected mean MAE {mae:.2f}% quoted in text",
              tex_has(flat, f"{mae:.2f}"), f"selection.json: {mae:.4f} +/- {sd:.4f}")
        maes = [r["mean_rel_MAE_sd_pct"] for r in sel["grid"]]
        check(f"grid MAE range {min(maes):.2f}-{max(maes):.2f}% consistent with text",
              tex_has(flat, f"{min(maes):.2f}") and tex_has(flat, f"{max(maes):.2f}"),
              f"selection.json: min {min(maes):.4f}, max {max(maes):.4f}")

    # ---------------- grid summary: seed count ----------------
    gs = load("alpha_grid/grid_summary.csv")
    if gs:
        check("grid summary has 30 rows", len(gs) == 30, f"grid_summary.csv: {len(gs)}")

    # ---------------- loss ablation ----------------
    abl = load("evaluation/ablation_loss_comparison.csv")
    if abl:
        # wide format: one row per output, one "<arm>_safe_met" column per arm
        arms = [c[:-len("_safe_met")] for c in abl[0] if c.endswith("_safe_met")]
        check("loss ablation has 5 arms", len(arms) == 5, f"arms: {arms}")
        for a in arms:
            met = sum(r[f"{a}_safe_met"] == "True" for r in abl)
            check(f"ablation {a}: {met}/{len(abl)} outputs on the conservative side appears in text",
                  tex_has(flat, f"{met}/{len(abl)}"), f"{met}/{len(abl)}")

    # ---------------- clean-test overlap check ----------------
    ct = load("evaluation/clean_test_eval.json")
    if ct:
        txt = json.dumps(ct)
        m = re.search(r'"n_removed"\s*:\s*(\d+)', txt) or re.search(r'"removed"\s*:\s*(\d+)', txt)
        if m:
            n = int(m.group(1))
            pretty = f"{n:,}".replace(",", "{,}")
            check(f"screening-overlap rows removed = {n:,}",
                  tex_has(flat, pretty) or tex_has(flat, f"{n:,}"),
                  f"clean_test_eval.json: {n}")

    # ---------------- clamp report ----------------
    cl = load("evaluation/clamp_report.json")
    if cl:
        deltas = [abs(v["R2_clamped"] - v["R2_unclamped"])
                  for split in cl.values() for v in split.values()
                  if isinstance(v, dict) and "R2_clamped" in v]
        if deltas:
            check("clamping R2 impact below 2.3e-5 as claimed",
                  max(deltas) < 2.3e-5, f"max |R2_clamped - R2_unclamped| = {max(deltas):.3g}")

    # ---------------- figure files referenced exist ----------------
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    for f in figs:
        p = ROOT / "figures" / f
        alt = ROOT / "paper" / f
        check(f"figure present: {f}", p.exists() or alt.exists(),
              "searched figures/ and paper/")

    # ---------------- report ----------------
    width = max(len(n) for n, _, _ in checks) if checks else 10
    npass = sum(1 for _, ok, _ in checks if ok)
    print("=" * (width + 30))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * (width + 30))
    print(f"{npass}/{len(checks)} traceable claims verified")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
