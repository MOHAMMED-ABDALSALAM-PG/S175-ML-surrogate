"""The ORIGINAL interpolation scheme as an accuracy baseline at off-grid points.

The simulator was originally deployed with a linear interpolation scheme over
the precomputed factorial table. This script reproduces that scheme exactly --
9-D multilinear interpolation on the full 126M-row lattice -- and scores it on
the same fresh off-grid simulator runs used to validate the surrogate
(scripts/08_interpolation_test.py), so the paper can compare the two on
identical points instead of asserting the interpolant's inadequacy.

Two views are reported per output and design:

  clean   accuracy on the queries whose 2^9 bracketing corners are all
          feasible for that output (the interpolant's best case; a convex
          combination of valid corner values);
  naive   accuracy over all scored rows, blending the -1 sentinels exactly as
          a plain interpolator over the raw table would -- this is the
          feasibility-contamination failure mode the two-stage design removes.

The fraction of queries with at least one infeasible corner ("contaminated
cells") quantifies how often the naive scheme mixes physical values with
sentinels.

Run:  python scripts/14_interpolation_baseline.py
Out:  results/evaluation/interpolation_baseline.json
      results/evaluation/interpolation_baseline.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from s175 import data as sdata
from s175.columns import FEATURE_COLS, OUTPUT_COLS, FUEL_IDX
from s175.metrics import clamp_physical, per_output

interp_test = __import__("08_interpolation_test")
ev = __import__("03_evaluate")

RUN_DIR = ROOT / "runs" / "v3_masked_peroutput" / "S1_random" / "seed_0"

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
OUT = ROOT / "results" / "evaluation"
SENTINEL = -1.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_lattice_index(X, levels):
    """Map lattice coordinates to row numbers without assuming any row order.

    The dataset was generated in several computation runs on different
    machines and then combined, so the rows need not follow one nested-loop
    order. Each row's level indices are encoded into a canonical key and the
    inverse permutation (key -> row number) is built explicitly; the full
    factorial guarantees each key occurs exactly once.
    """
    sizes = np.array([len(levels[c]) for c in FEATURE_COLS], dtype=np.int64)
    assert int(sizes.prod()) == len(X), "lattice sizes do not multiply to n_rows"
    strides = np.empty(len(sizes), dtype=np.int64)
    acc = 1
    for k in range(len(sizes) - 1, -1, -1):
        strides[k] = acc
        acc *= sizes[k]

    n = len(X)
    key = np.zeros(n, dtype=np.int64)
    chunk = 20_000_000
    for k, c in enumerate(FEATURE_COLS):
        lv = levels[c]
        for s in range(0, n, chunk):
            col = np.asarray(X[s:s + chunk, k])
            key[s:s + chunk] += np.searchsorted(lv, col).astype(np.int64) * strides[k]
    perm = np.empty(n, dtype=np.int64)
    perm[key] = np.arange(n, dtype=np.int64)

    # verify on a random sample: decode a key, look the row up, compare inputs
    rng = np.random.default_rng(0)
    for key_i in rng.integers(0, n, 500):
        row = perm[key_i]
        rem = int(key_i)
        for k in range(len(sizes)):
            idx = rem // strides[k]
            rem -= idx * strides[k]
            if X[row, k] != levels[FEATURE_COLS[k]][idx]:
                raise RuntimeError("lattice index verification failed")
    log(f"lattice index built and verified; strides {strides.tolist()}")
    return strides, perm


def interpolate(queries, levels, strides, perm, Y_ram):
    """9-D multilinear interpolation. Returns blended outputs and, per output,
    whether any positively-weighted corner carries the -1 sentinel."""
    nq = len(queries)
    nk = len(FEATURE_COLS)
    lo = np.empty((nq, nk), dtype=np.int64)
    t = np.empty((nq, nk), dtype=np.float64)
    for k, c in enumerate(FEATURE_COLS):
        lv = levels[c].astype(np.float64)
        q = np.clip(queries[:, k], lv[0], lv[-1])
        hi_i = np.searchsorted(lv, q, side="left")
        hi_i = np.clip(hi_i, 1, len(lv) - 1)
        lo_i = hi_i - 1
        exact = q == lv[np.clip(hi_i, 0, len(lv) - 1)]
        t[:, k] = (q - lv[lo_i]) / (lv[hi_i] - lv[lo_i])
        t[exact, k] = 1.0  # sits exactly on the upper node
        lo[:, k] = lo_i

    blend = np.zeros((nq, len(OUTPUT_COLS)), dtype=np.float64)
    contaminated = np.zeros((nq, len(OUTPUT_COLS)), dtype=bool)
    for b in range(1 << nk):
        bits = np.array([(b >> k) & 1 for k in range(nk)], dtype=np.int64)
        w = np.ones(nq, dtype=np.float64)
        for k in range(nk):
            w *= t[:, k] if bits[k] else (1.0 - t[:, k])
        idx = perm[((lo + bits[None, :]) * strides[None, :]).sum(axis=1)]
        vals = Y_ram[idx].astype(np.float64)
        blend += w[:, None] * vals
        contaminated |= (w > 0)[:, None] & (vals == SENTINEL)
    return blend, contaminated


def main() -> int:
    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    log(f"loaded {len(X):,} rows")

    levels = interp_test.grid_levels(X)
    log("grid levels: " + ", ".join(f"{c}:{len(levels[c])}" for c in FEATURE_COLS))
    strides, perm = build_lattice_index(X, levels)

    t0 = time.perf_counter()
    Y_ram = np.array(Y)  # 5 GB; the corner gathers need RAM-speed random access
    log(f"outputs resident in RAM in {time.perf_counter()-t0:.0f}s")

    # the surrogate's regressor, for the matched clean-cell comparison
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg, reg_ck = ev.load_component(RUN_DIR, "regressor", device)
    x_scaler = ev.Scaler(reg_ck["x_scaler"])
    y_scaler = ev.Scaler(reg_ck["y_scaler"])

    results = {}
    rows_csv = []
    for design, path in interp_test.SIM_FILES.items():
        df = interp_test.parse_sim_output(path)
        q = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        y_true = df[OUTPUT_COLS].to_numpy(dtype=np.float64)
        log(f"=== {design}: {len(df):,} fresh off-grid rows ===")

        blend, contaminated = interpolate(q, levels, strides, perm, Y_ram)

        # surrogate predictions at the identical query points
        with torch.no_grad():
            xb = torch.from_numpy(
                x_scaler.transform(q.astype(np.float32))).to(device)
            y_sur = reg(xb).float().cpu().numpy()
        y_sur = clamp_physical(y_scaler.inverse(y_sur)).astype(np.float64)

        all_neg = (y_true == SENTINEL).all(axis=1)
        fuel_only = (y_true[:, FUEL_IDX] == SENTINEL) & ~all_neg
        defined = ~all_neg
        valid = ~all_neg & ~fuel_only

        per = {}
        for j, c in enumerate(OUTPUT_COLS):
            scored = valid if c == "fuel" else defined
            clean = scored & ~contaminated[:, j]
            m_clean = per_output(y_true[clean][:, [j]],
                                 blend[clean][:, [j]], cols=[c])[c]
            m_naive = per_output(y_true[scored][:, [j]],
                                 blend[scored][:, [j]], cols=[c])[c]
            m_sur = per_output(y_true[clean][:, [j]],
                               y_sur[clean][:, [j]], cols=[c])[c]
            per[c] = {
                "n_scored": int(scored.sum()),
                "n_clean": int(clean.sum()),
                "pct_contaminated": 100.0 * (1 - clean.sum() / scored.sum()),
                "clean": {"R2": m_clean["R2"], "MAE": m_clean["MAE"]},
                "naive": {"R2": m_naive["R2"], "MAE": m_naive["MAE"]},
                "surrogate_clean": {"R2": m_sur["R2"], "MAE": m_sur["MAE"]},
            }
            log(f"  {c:12s} interp clean R2 {m_clean['R2']:.4f} | surrogate "
                f"clean R2 {m_sur['R2']:.4f} (n={clean.sum():,}) | naive R2 "
                f"{m_naive['R2']:8.3f} | contaminated "
                f"{per[c]['pct_contaminated']:.1f}%")
            rows_csv.append([design, c, per[c]["n_scored"], per[c]["n_clean"],
                             round(per[c]["pct_contaminated"], 2),
                             m_clean["R2"], m_clean["MAE"],
                             m_sur["R2"], m_sur["MAE"],
                             m_naive["R2"], m_naive["MAE"]])
        results[design] = {
            "n_rows": int(len(df)),
            "note": ("clean = all bracketing corners feasible for the output; "
                     "naive = sentinels blended as a plain interpolator would"),
            "per_output": per,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interpolation_baseline.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "interpolation_baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "output", "n_scored", "n_clean",
                    "pct_contaminated", "clean_R2", "clean_MAE",
                    "surrogate_clean_R2", "surrogate_clean_MAE",
                    "naive_R2", "naive_MAE"])
        w.writerows(rows_csv)
    log(f"wrote {OUT / 'interpolation_baseline.json'} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
