"""The single dataset loader.

Replaces the ~15-line load/mask/split block that was copy-pasted across eight
scripts in the original project (which is how the float32/float64 and
stratified/unstratified inconsistencies arose).

Two things this adds over the original:

1.  Header verification. The raw file is positional; nothing previously checked
    that column 12 really is ship speed. A regenerated file with reordered
    columns would have been read silently with the wrong mapping.

2.  A binary cache. The original re-parsed 15.9 GB of text on every run. We
    parse once into a flat float32 binary (10.6 GB) and memory-map it
    thereafter. Flat rather than .npy so the chunks can be streamed straight to
    disk without ever materialising the whole array; the shape is recovered
    from the column count, and the row count is cross-checked against the meta.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import (
    ALL_COLS, FEATURE_COLS, OUTPUT_COLS, HEADER_TOKENS, N_PREAMBLE_LINES,
    INFEASIBLE, FUEL_IDX, CONSTANT_INPUTS,
    CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG,
)

CHUNK_ROWS = 4_000_000


def sha256_file(path: str | Path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_header(path: str | Path) -> str:
    """Read the raw header line and assert it matches the expected schema.

    Raises ValueError on any mismatch. Returns the header line.
    """
    with open(path, "r", errors="replace") as f:
        lines = [f.readline() for _ in range(N_PREAMBLE_LINES)]
    header = lines[N_PREAMBLE_LINES - 1]
    fields = [s.strip().lower() for s in header.split(";")]
    if len(fields) != len(ALL_COLS):
        raise ValueError(
            f"header has {len(fields)} fields, expected {len(ALL_COLS)}: {header!r}"
        )
    for i, (field, token) in enumerate(zip(fields, HEADER_TOKENS)):
        if token not in field:
            raise ValueError(
                f"column {i} header mismatch: expected to find {token!r} in {field!r}. "
                "The raw file schema has changed -- update columns.py deliberately."
            )
    return header.strip()


@dataclass
class DatasetMeta:
    source: str
    source_sha256: str
    n_rows: int
    n_cols: int
    header: str
    cache: str

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def build_cache(raw_path: str | Path, cache_path: str | Path,
                verify_checksum: bool = True) -> DatasetMeta:
    """Parse the raw semicolon text into a float32 .npy, once.

    Streams in chunks so peak memory stays near CHUNK_ROWS * 21 * 4 bytes
    rather than materialising the whole frame.
    """
    raw_path, cache_path = Path(raw_path), Path(cache_path)
    header = verify_header(raw_path)

    reader = pd.read_csv(
        raw_path, sep=";", skiprows=N_PREAMBLE_LINES, header=None,
        names=ALL_COLS, dtype=np.float32, chunksize=CHUNK_ROWS,
    )

    # Stream straight to a flat binary file. Concatenating chunks in memory
    # would peak at roughly twice the 10.6 GB array size for no benefit.
    tmp = cache_path.with_suffix(".building.bin")
    n_rows = 0
    with open(tmp, "wb") as fh:
        for ch in reader:
            arr = np.ascontiguousarray(ch.to_numpy(dtype=np.float32, copy=False))
            if np.isnan(arr).any():
                bad = int(np.isnan(arr).any(axis=1).sum())
                raise ValueError(
                    f"{bad} rows contain NaN after parsing -- check preamble handling")
            fh.write(arr.tobytes())
            n_rows += len(arr)
            print(f"    parsed {n_rows:,} rows", flush=True)
    os.replace(tmp, cache_path)

    meta = DatasetMeta(
        source=str(raw_path),
        source_sha256=sha256_file(raw_path) if verify_checksum else "skipped",
        n_rows=int(n_rows), n_cols=len(ALL_COLS),
        header=header, cache=str(cache_path),
    )
    meta.save(cache_path.with_suffix(".meta.json"))
    return meta


def load(raw_path: str | Path, cache_dir: str | Path = "data/cache",
         rebuild: bool = False, mmap: bool = True) -> tuple[np.ndarray, DatasetMeta]:
    """Return the full (n_rows, 21) float32 table, building the cache if needed."""
    raw_path = Path(raw_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (raw_path.stem + ".f32.bin")
    meta_path = cache_path.with_suffix(".meta.json")

    if rebuild or not cache_path.exists():
        meta = build_cache(raw_path, cache_path)
    else:
        meta = DatasetMeta(**json.loads(meta_path.read_text()))

    n_cols = len(ALL_COLS)
    arr = np.memmap(cache_path, dtype=np.float32, mode="r").reshape(-1, n_cols)
    if meta.n_rows and len(arr) != meta.n_rows:
        raise ValueError(f"cache has {len(arr):,} rows, meta records {meta.n_rows:,}")
    if not mmap:
        # np.asarray on a memmap returns a *view* still backed by the mapping;
        # an explicit copy is what actually brings the data into RAM.
        arr = np.array(arr)
    if arr.shape[1] != len(ALL_COLS):
        raise ValueError(f"cache has {arr.shape[1]} columns, expected {len(ALL_COLS)}")
    return arr, meta


def split_xy(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (nine varying inputs) and output matrix (ten outputs)."""
    fi = [ALL_COLS.index(c) for c in FEATURE_COLS]
    oi = [ALL_COLS.index(c) for c in OUTPUT_COLS]
    return arr[:, fi], arr[:, oi]


def feasibility(y: np.ndarray, strict: bool = True) -> np.ndarray:
    """Per-row feasibility class.

    Returns an int8 array of CLASS_VALID / CLASS_FUEL_ONLY / CLASS_ALL_NEG.

    strict=True asserts the simulator's documented invariant: infeasibility
    appears either in all ten outputs or in fuel alone. The original code only
    checked this for the partial rows, so a row with (say) roll == -1 and
    everything else valid would have been labelled fully valid and its -1 fed
    straight into the regression target and the output scaler.
    """
    all_neg = (y == INFEASIBLE).all(axis=1)
    fuel_neg = (y[:, FUEL_IDX] == INFEASIBLE) & ~all_neg

    cls = np.full(len(y), CLASS_VALID, dtype=np.int8)
    cls[fuel_neg] = CLASS_FUEL_ONLY
    cls[all_neg] = CLASS_ALL_NEG

    if strict:
        valid = cls == CLASS_VALID
        stray = (y[valid] == INFEASIBLE).any(axis=1)
        if stray.any():
            raise ValueError(
                f"{int(stray.sum())} rows classed fully-valid still contain a -1 "
                "sentinel. The all-or-fuel-only invariant does not hold; "
                "inspect before training."
            )
        partial = cls == CLASS_FUEL_ONLY
        others = np.delete(np.arange(y.shape[1]), FUEL_IDX)
        stray_p = (y[partial][:, others] == INFEASIBLE).any(axis=1)
        if stray_p.any():
            raise ValueError(
                f"{int(stray_p.sum())} fuel-only rows have -1 in another output."
            )
    return cls


def check_constants(arr: np.ndarray) -> None:
    """Assert the two held-constant inputs really are constant.

    The comparison is made in float32, not float64. The stored value of 0.55 is
    exactly 0.550000011920929 once round-tripped through float32, so widening it
    to a Python float before comparing against the literal 0.55 rejects a
    perfectly correct file.
    """
    for name, expected in CONSTANT_INPUTS.items():
        col = arr[:, ALL_COLS.index(name)]
        lo, hi = col.min(), col.max()
        target = np.float32(expected)
        if lo != hi:
            raise ValueError(
                f"{name} expected constant {expected}, found range "
                f"[{float(lo)}, {float(hi)}]"
            )
        if lo != target:
            raise ValueError(
                f"{name} expected constant {expected}, found {float(lo)}"
            )


def class_counts(cls: np.ndarray) -> dict[str, int]:
    return {
        "fully_valid": int((cls == CLASS_VALID).sum()),
        "fuel_only": int((cls == CLASS_FUEL_ONLY).sum()),
        "all_infeasible": int((cls == CLASS_ALL_NEG).sum()),
        "total": int(len(cls)),
    }
