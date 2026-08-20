"""Build S0_screen: the loss-weight screening split.

The screening subsample (S175_sample_5pct_stratified.csv, 5% of the factorial
table) is the population on which the loss weights are selected. This script
maps every subsample row back to its exact row number in the full table via
the lattice index -- the same mapping 15_clean_test_eval uses to keep the
primary test numbers disjoint from these rows -- and partitions them 80/10/10
into train / val / selection, stratified like S1 (feasibility class crossed
with sea-state and wind bands).

The three parts are persisted with checksums like every other split. The
"test" part of S0_screen is the *selection* partition: loss weights are chosen
on it, and it is never trained on. The final S1/S2/S3/S4 test sets remain
disjoint from all screening decisions because the screening rows are already
excluded from the clean test evaluation.

  python scripts/19_screening_split.py

Output: splits/S0_screen/{train,val,test}.npy + meta.json + provenance.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from s175 import data as sdata
from s175.columns import FEATURE_COLS
from s175.splits import Split, make_strata, _stratified_three_way, load_split

baseline = __import__("14_interpolation_baseline")
clean = __import__("15_clean_test_eval")

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
SAMPLE = clean.SAMPLE
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"

SPLIT_SEED = 0
VAL_FRAC = 0.1
SELECTION_FRAC = 0.1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    out_dir = SPLIT_DIR / "S0_screen"
    if (out_dir / "meta.json").exists():
        sp = load_split(SPLIT_DIR, "S0_screen")   # checksum-verifies the files
        log(f"S0_screen already on disk and verified: "
            f"{len(sp.train):,} / {len(sp.val):,} / {len(sp.test):,} rows")
        return 0

    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    log(f"loaded {len(X):,} rows")

    levels = {c: np.unique(np.asarray(X[:, k], dtype=np.float32))
              for k, c in enumerate(FEATURE_COLS)}
    strides, perm = baseline.build_lattice_index(X, levels)
    srows = np.sort(clean.sample_row_numbers(X, levels, strides, perm))
    log(f"screening rows mapped: {len(srows):,} "
        f"({100.0 * len(srows) / len(X):.3f}% of the table)")

    strata = make_strata(X[srows], cls[srows])
    tr, va, te = _stratified_three_way(strata, SPLIT_SEED,
                                       VAL_FRAC, SELECTION_FRAC)
    sp = Split(
        "S0_screen", srows[tr], srows[va], srows[te],
        "Screening subsample (the historical 5% stratified sample, mapped to "
        "exact table rows), partitioned 80/10/10 into train / val / selection "
        "with S1's stratification. Loss weights are selected on the "
        "'test' (selection) part; the primary test sets stay disjoint via the "
        "clean test evaluation.",
    )
    sp.assert_disjoint()
    counts = sp.assert_covers_all_classes(cls)
    sp.save(SPLIT_DIR, class_counts=counts)
    log(f"saved S0_screen: train {len(sp.train):,} val {len(sp.val):,} "
        f"selection {len(sp.test):,}")

    # Overlap of the screening rows with every regime's test set, so the
    # separation argument is verifiable from disk alone.
    in_sample = np.zeros(len(X), dtype=bool)
    in_sample[srows] = True
    overlap = {}
    for name in ["S1_random", "S2_level_Hs", "S2_level_Chi", "S2_level_draft",
                 "S2_level_Vwind", "S3_block", "S4_corner"]:
        te_idx = load_split(SPLIT_DIR, name).test
        n_ov = int(in_sample[te_idx].sum())
        overlap[name] = {"n_test": int(len(te_idx)), "n_overlap": n_ov,
                         "pct_overlap": 100.0 * n_ov / len(te_idx)}
        log(f"  overlap with {name} test: {n_ov:,} rows "
            f"({overlap[name]['pct_overlap']:.3f}%)")

    provenance = {
        "sample_csv": str(SAMPLE),
        "sample_csv_sha256": sdata.sha256_file(SAMPLE),
        "n_sample_rows": int(len(srows)),
        "mapping": "lattice index (14_interpolation_baseline.build_lattice_index)",
        "split_seed": SPLIT_SEED,
        "fractions": {"train": 1.0 - VAL_FRAC - SELECTION_FRAC,
                      "val": VAL_FRAC, "selection": SELECTION_FRAC},
        "stratification": "make_strata (feasibility class x Hs band x Vwind band)",
        "dataset_sha256": meta.source_sha256,
        "test_set_overlap": overlap,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log(f"wrote {out_dir / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
