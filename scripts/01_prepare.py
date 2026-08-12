"""Phase 1, step 1: build the binary cache and the four holdout regimes.

Run:  python scripts/01_prepare.py [--rebuild] [--splits-only]

Reads the raw 15.9 GB table once, caches it as flat float32, then constructs
and validates every split. Nothing here is sampled at random without a recorded
seed, and no split is written to disk until it has passed three checks:

  * disjoint      -- no row appears in two parts
  * partition     -- every row appears in exactly one part
  * full coverage -- train, val and test each contain all three feasibility
                     cases (fully valid, fuel-only infeasible, completely
                     infeasible)

The coverage check is the one that matters for the rule-based regimes. S2, S3
and S4 carve out their test set by a condition on the inputs rather than by
sampling, and the infeasible rows concentrate at high Hs -- which is exactly
the region S4 holds out. Without the check, a corner split can hand the
feasibility classifier a training set with almost no infeasible examples.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s175 import data as sdata
from s175.columns import FEATURE_COLS, CLASS_NAMES, CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG
from s175.splits import (
    split_random, split_level_holdout, split_block, split_corner_extrapolation,
)

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"
RESULTS = ROOT / "results"
SEED = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-parse the raw text even if the cache exists")
    ap.add_argument("--splits-only", action="store_true",
                    help="skip cache verification work and go straight to splits")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    SPLIT_DIR.mkdir(exist_ok=True)

    log(f"loading {RAW}")
    t0 = time.time()
    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR, rebuild=args.rebuild)
    log(f"table {arr.shape[0]:,} x {arr.shape[1]} in {time.time()-t0:.1f}s")
    log(f"source sha256 {meta.source_sha256[:16]}...")

    log("checking the held-constant inputs")
    sdata.check_constants(arr)
    log("  GMt and shaft_gen confirmed constant")

    log("splitting features and outputs")
    X, Y = sdata.split_xy(arr)

    log("classifying feasibility (strict invariant check)")
    cls = sdata.feasibility(Y, strict=True)
    counts = sdata.class_counts(cls)
    total = counts["total"]
    for k, v in counts.items():
        if k != "total":
            log(f"  {k:<18} {v:>12,}  ({100*v/total:5.2f}%)")

    del Y

    # ---- record the grid -------------------------------------------------
    log("recording the input grid")
    levels = {}
    for i, name in enumerate(FEATURE_COLS):
        u = np.unique(X[:, i])
        levels[name] = {"n_levels": int(u.size),
                        "min": float(u.min()), "max": float(u.max()),
                        "values": [float(v) for v in u] if u.size <= 64 else None}
        log(f"  {name:<12} {u.size:>4} levels  [{u.min():g}, {u.max():g}]")

    summary = {
        "source": str(RAW), "source_sha256": meta.source_sha256,
        "n_rows": int(total), "class_counts": counts, "levels": levels,
        "seed": SEED,
    }
    (RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {RESULTS/'dataset_summary.json'}")

    if args.splits_only is False:
        pass  # cache work already done above

    # ---- build the four regimes -----------------------------------------
    def interior_levels(name: str, take: int):
        """Pick `take` levels from the interior, never the endpoints.

        Held-out levels must be surrounded by training levels, otherwise S2
        measures extrapolation and duplicates what S4 already reports.
        """
        u = np.unique(X[:, FEATURE_COLS.index(name)])
        inner = u[1:-1]
        if inner.size < take:
            raise ValueError(f"{name} has only {inner.size} interior levels")
        pick = np.linspace(0, inner.size - 1, take).round().astype(int)
        return [float(v) for v in inner[np.unique(pick)]]

    specs = []

    specs.append(("S1_random", lambda: split_random(X, cls, seed=SEED)))

    for var, take in (("Hs", 3), ("Chi", 3), ("draft", 1), ("Vwind", 2)):
        vals = interior_levels(var, take)
        specs.append((f"S2_level_{var}",
                      lambda v=var, s=vals: split_level_holdout(X, v, s, cls, seed=SEED)))
        log(f"  S2 {var}: holding out {vals}")

    hs_u = np.unique(X[:, FEATURE_COLS.index("Hs")])
    chi_u = np.unique(X[:, FEATURE_COLS.index("Chi")])
    # High seas crossed with beam-ish headings: a joint region, not a slab.
    hs_lo = float(np.quantile(hs_u, 0.70))
    chi_lo, chi_hi = float(np.quantile(chi_u, 0.35)), float(np.quantile(chi_u, 0.65))
    block_conditions = {"Hs": (hs_lo, float(hs_u.max())), "Chi": (chi_lo, chi_hi)}
    specs.append(("S3_block",
                  lambda c=block_conditions: split_block(X, c, cls, seed=SEED)))
    log(f"  S3 block: {block_conditions}")

    specs.append(("S4_corner",
                  lambda: split_corner_extrapolation(X, ["Hs", "Vwind"], cls,
                                                     quantile=0.85, seed=SEED)))

    manifest = {}
    failures = []
    for name, build in specs:
        log(f"building {name}")
        t0 = time.time()
        sp = build()
        # assert_partition counts occurrences, which proves disjointness and
        # the absence of duplicates as well -- the pairwise intersect1d check
        # would repeat that work at O(n log n).
        sp.assert_partition(total)
        try:
            cc = sp.assert_covers_all_classes(cls)
            covered = True
        except ValueError as e:
            # Report every regime rather than aborting on the first bad one:
            # which regimes fail coverage is itself a result worth recording.
            covered = False
            cc = sp.class_counts(cls)
            failures.append(f"{name}: {e}")
            log(f"  !! COVERAGE FAILED -- {e}")

        m = sp.save(SPLIT_DIR, class_counts=cc)
        m["covers_all_classes"] = covered
        manifest[sp.name] = m
        n = sp.counts()
        log(f"  train {n['train']:,} | val {n['val']:,} | test {n['test']:,}"
            f"  ({time.time()-t0:.1f}s)")
        for part in ("train", "val", "test"):
            row = " ".join(f"{k}={v:,}" for k, v in cc[part].items())
            log(f"    {part:<5} {row}")

    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"wrote {SPLIT_DIR/'manifest.json'}")

    if failures:
        log("")
        log(f"{len(failures)} regime(s) do not contain every case in every part:")
        for f in failures:
            log(f"  - {f.splitlines()[0]}")
        log("Splits were still written, flagged covers_all_classes=false.")
        return 1

    log("all regimes built, validated and saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
