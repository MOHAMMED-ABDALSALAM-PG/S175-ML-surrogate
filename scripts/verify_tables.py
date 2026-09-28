"""Compare every generated results table against the manuscript, value by value.

The manuscript reformats the generated tables for siunitx (S columns,
\\unit{}, automatic thousands separators), so a raw text diff is useless. This
script strips the tabular preamble and all LaTeX markup, normalises number
formatting, and compares the remaining numeric cells row by row.

  python scripts/verify_tables.py paper/main.tex
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LATEX = ROOT / "results" / "evaluation" / "latex"

PAIRS = {
    "table_classifiers_clean.tex": "tab:classifiers",
    "table_regression_clean.tex": "tab:regression",
    "table_seeds_clean.tex": "tab:seeds",
    "table_general_clean.tex": "tab:general",
    "table_alpha_grid.tex": "tab:alphagrid",
}

RULES = re.compile(r"\\(hline|toprule|midrule|bottomrule|cmidrule|addlinespace)"
                   r"(\(lr\))?(\{[^}]*\})?")


def body_after_preamble(block: str) -> str:
    """Everything after \\begin{tabular}{...}, i.e. drop the column spec."""
    i = block.find("\\begin{tabular}")
    if i < 0:
        return block
    j = block.index("{", i + len("\\begin{tabular}") - 1)
    # walk the (possibly multi-line) column-spec braces
    d = 0
    while j < len(block):
        if block[j] == "{":
            d += 1
        elif block[j] == "}":
            d -= 1
            if d == 0:
                break
        j += 1
    e = block.find("\\end{tabular}", j)
    return block[j + 1:e if e > 0 else len(block)]


def numbers_of_row(row: str) -> list[str]:
    row = re.sub(r"%.*", "", row)
    row = RULES.sub(" ", row)
    row = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", " ", row)
    row = re.sub(r"\\(safe|textbf|emph|unit|si|num|text)\s*", "", row)
    row = row.replace("{,}", "").replace("\\,", "")
    out = []
    for cell in row.split("&"):
        cell = re.sub(r"\\[a-zA-Z]+", " ", cell)
        cell = cell.replace("{", " ").replace("}", " ")
        for tok in re.findall(r"-?\d+(?:\.\d+)?", cell.replace(",", "")):
            out.append(tok)
    return out


def data_rows(block: str) -> list[list[str]]:
    rows = []
    for raw in body_after_preamble(block).split("\\\\"):
        if "&" not in raw:
            continue
        if "multicolumn" in raw and not re.search(r"\d", RULES.sub("", raw).split("&", 1)[-1]):
            continue
        nums = numbers_of_row(raw)
        if nums:
            rows.append(nums)
    return rows


def find_block(tex: str, label: str) -> str | None:
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return None
    b = tex.rfind("\\begin{table", 0, i)
    e = tex.find("\\end{table", i)
    if b < 0 or e < 0:
        return None
    return tex[b:tex.find("}", e) + 1]


def main() -> int:
    man = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                       else "S175-ML-surrogate.tex").read_text(encoding="utf-8",
                                                               errors="replace")
    bad = 0
    for fname, label in PAIRS.items():
        p = LATEX / fname
        print(f"\n=== {label}  <-  {fname} ===")
        if not p.exists():
            print("   generated file MISSING"); bad += 1; continue
        mblock = find_block(man, label)
        if mblock is None:
            print("   not found in manuscript"); bad += 1; continue
        g = data_rows(p.read_text(encoding="utf-8"))
        m = data_rows(mblock)
        gflat = [x for r in g for x in r]
        mflat = [x for r in m for x in r]
        if gflat == mflat:
            print(f"   MATCH -- {len(gflat)} numeric cells identical")
            continue
        # compare numerically, tolerating trailing-zero formatting
        def f(x):
            try: return float(x)
            except ValueError: return None
        gi, mi = [f(x) for x in gflat], [f(x) for x in mflat]
        if len(gi) == len(mi) and all(
                a is not None and b is not None and abs(a - b) < 1e-9
                for a, b in zip(gi, mi)):
            print(f"   MATCH -- {len(gi)} cells equal in value "
                  f"(formatting differs only)")
            continue
        print(f"   generated {len(gflat)} cells, manuscript {len(mflat)} cells")
        shown = 0
        for i, (a, b) in enumerate(zip(gflat, mflat)):
            if a != b:
                print(f"     cell {i:>3}: generated {a:>12} | manuscript {b:>12}")
                bad += 1; shown += 1
                if shown >= 15:
                    print("     ..."); break
        if len(gflat) != len(mflat):
            print(f"     !! cell-count mismatch ({len(gflat)} vs {len(mflat)})")
            bad += 1
    print(f"\n{'=' * 62}")
    print("ALL TABLES CONSISTENT" if bad == 0 else f"{bad} issue(s) flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
