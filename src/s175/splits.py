"""Split construction, and persistence of the resulting indices.

Four regimes, because a random row split on a dense factorial lattice only
measures interpolation between adjacent grid nodes. The original work reported
S1 alone and obtained R^2 > 0.9999; the genuinely off-grid simulator checks
landed near 0.99, and that gap is the honest story.

S1  random      stratified 80/10/10 over rows, as originally done
S2  level       hold out entire values of one input variable
S3  block       hold out a contiguous joint region of the operating envelope
S4  corner      train on the interior, test outside the sampled envelope

Indices are always written to disk with a checksum. The original code replayed
`train_test_split(random_state=42)` months later to reconstruct its test set,
which silently depends on the scikit-learn version -- the experiments ran on
1.8.0 and the environment now has 1.9.0, so that split is no longer recoverable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .columns import (
    FEATURE_COLS, CLASS_NAMES, CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG,
)


@dataclass
class Split:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    description: str

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val),
                "test": len(self.test)}

    def assert_disjoint(self) -> None:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ia, ib = getattr(self, a), getattr(self, b)
            if np.intersect1d(ia, ib, assume_unique=False).size:
                raise ValueError(f"{self.name}: {a} and {b} overlap")

    def assert_partition(self, n_rows: int) -> None:
        """Every row lands in exactly one part.

        Counting occurrences proves three things at once -- no row is missing,
        no row is duplicated within a part, and no row appears in two parts --
        in a single O(n) pass. The obvious implementation (pairwise
        `intersect1d` plus `unique` over the union) sorts arrays of 100M+
        indices six times per split, which measured at over twenty minutes for
        one regime on this dataset.
        """
        total = len(self.train) + len(self.val) + len(self.test)
        if total != n_rows:
            raise ValueError(
                f"{self.name}: parts total {total:,} rows, dataset has {n_rows:,}"
            )
        counts = np.zeros(n_rows, dtype=np.int64)
        for part in ("train", "val", "test"):
            a = np.asarray(getattr(self, part), dtype=np.int64)
            if a.size == 0:
                continue
            # Range-checked per part before the bincount: an oversized index in
            # val or test would otherwise surface as an opaque broadcast error
            # when the longer bincount array is added to `counts`.
            lo, hi = int(a.min()), int(a.max())
            if lo < 0 or hi >= n_rows:
                raise ValueError(
                    f"{self.name}: {part} has an index out of range for "
                    f"{n_rows:,} rows (min {lo:,}, max {hi:,})"
                )
            counts += np.bincount(a, minlength=n_rows)
        if not np.array_equal(counts, np.ones(n_rows, dtype=counts.dtype)):
            n_missing = int((counts == 0).sum())
            n_repeated = int((counts > 1).sum())
            raise ValueError(
                f"{self.name}: not a partition -- {n_missing:,} rows in no part, "
                f"{n_repeated:,} rows in more than one part"
            )

    def class_counts(self, cls: np.ndarray) -> dict[str, dict[str, int]]:
        """Feasibility-class breakdown of each part."""
        out = {}
        for part in ("train", "val", "test"):
            idx = getattr(self, part)
            sub = cls[idx]
            out[part] = {
                CLASS_NAMES[c]: int((sub == c).sum())
                for c in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG)
            }
        return out

    def assert_covers_all_classes(self, cls: np.ndarray,
                                  min_count: int = 1,
                                  parts: tuple[str, ...] = ("train", "val", "test"),
                                  ) -> dict[str, dict[str, int]]:
        """Every named part must contain all three feasibility classes.

        The rule-based regimes (S2 level, S3 block, S4 corner) carve out their
        test set by a condition on the inputs, not by sampling, so nothing
        otherwise guarantees the remaining data still spans every case. The
        infeasible rows concentrate at high Hs, which is precisely the region
        S4 holds out -- so an unchecked corner split can train a feasibility
        classifier on data containing almost no infeasible examples.

        Raises ValueError naming the offending part and class.
        """
        counts = self.class_counts(cls)
        missing = []
        for part in parts:
            for name, n in counts[part].items():
                if n < min_count:
                    missing.append(f"{part} has {n} '{name}' rows (need >= {min_count})")
        if missing:
            raise ValueError(
                f"{self.name}: split does not cover every case -- "
                + "; ".join(missing)
                + f". Full breakdown: {json.dumps(counts)}"
            )
        return counts

    def save(self, root: str | Path, cls: np.ndarray | None = None,
             class_counts: dict | None = None) -> dict:
        d = Path(root) / self.name
        d.mkdir(parents=True, exist_ok=True)
        digests = {}
        for part in ("train", "val", "test"):
            arr = np.asarray(getattr(self, part), dtype=np.int64)
            p = d / f"{part}.npy"
            np.save(p, arr)
            digests[part] = hashlib.sha256(arr.tobytes()).hexdigest()
        meta = {"name": self.name, "description": self.description,
                "counts": self.counts(), "sha256": digests}
        # Recorded so the coverage guarantee is auditable from disk alone,
        # without reloading the 10.6 GB table. Accept a precomputed breakdown
        # so the caller does not pay for the same pass twice.
        if class_counts is not None:
            meta["class_counts"] = class_counts
        elif cls is not None:
            meta["class_counts"] = self.class_counts(cls)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def _stratified_three_way(strata: np.ndarray, seed: int,
                          val_frac: float, test_frac: float) -> tuple[np.ndarray, ...]:
    """Stratified 3-way partition implemented directly, not via sklearn.

    Doing it here removes the library-version dependence that made the original
    split unrecoverable.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test.append(idx[:n_test])
        val.append(idx[n_test:n_test + n_val])
        train.append(idx[n_test + n_val:])
    return (np.sort(np.concatenate(train)), np.sort(np.concatenate(val)),
            np.sort(np.concatenate(test)))


def make_strata(X: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Feasibility class crossed with sea-state and wind bands."""
    hs = X[:, FEATURE_COLS.index("Hs")]
    vw = X[:, FEATURE_COLS.index("Vwind")]
    hs_b = np.digitize(hs, [2.01, 5.01, 8.01])
    vw_b = np.digitize(vw, [7.5, 17.5])
    return (hs_b * 100 + vw_b * 10 + cls).astype(np.int32)


def split_random(X: np.ndarray, cls: np.ndarray, seed: int = 0,
                 val_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    tr, va, te = _stratified_three_way(make_strata(X, cls), seed, val_frac, test_frac)
    return Split("S1_random", tr, va, te,
                 "Stratified random row split, 80/10/10. Comparable to the "
                 "original experiments. Measures interpolation between "
                 "adjacent grid nodes, not generalisation.")


def split_level_holdout(X: np.ndarray, variable: str, held_out_values,
                        cls: np.ndarray, seed: int = 0,
                        val_frac: float = 0.111) -> Split:
    """Hold out entire levels of one input variable.

    Every row whose `variable` takes a held-out value goes to test; the rest is
    split into train/val. Answers: can the surrogate reach a sea state, heading
    or draft it has never been shown?
    """
    col = X[:, FEATURE_COLS.index(variable)]
    held = np.isin(np.round(col.astype(np.float64), 6),
                   np.round(np.asarray(held_out_values, dtype=np.float64), 6))
    test = np.flatnonzero(held)
    rest = np.flatnonzero(~held)
    if test.size == 0:
        raise ValueError(f"no rows match {variable} in {held_out_values}; "
                         f"present levels: {np.unique(col)[:20]}")
    sub_tr, sub_va, _ = _stratified_three_way(make_strata(X[rest], cls[rest]),
                                              seed, val_frac, 0.0)
    return Split(f"S2_level_{variable}", rest[sub_tr], rest[sub_va], test,
                 f"Hold out {variable} = {list(held_out_values)} entirely. "
                 "Tests interpolation to unseen levels of one input.")


def split_block(X: np.ndarray, conditions: dict, cls: np.ndarray,
                seed: int = 0, val_frac: float = 0.111,
                name: str = "S3_block") -> Split:
    """Hold out a contiguous joint region.

    `conditions` maps variable name -> (lo, hi) inclusive. A row is held out
    only if it satisfies every condition, so this carves a box out of the
    operating envelope (e.g. high seas AND beam heading).
    """
    mask = np.ones(len(X), dtype=bool)
    for var, (lo, hi) in conditions.items():
        col = X[:, FEATURE_COLS.index(var)]
        mask &= (col >= lo) & (col <= hi)
    test = np.flatnonzero(mask)
    rest = np.flatnonzero(~mask)
    if test.size == 0:
        raise ValueError(f"block {conditions} selects no rows")
    sub_tr, sub_va, _ = _stratified_three_way(make_strata(X[rest], cls[rest]),
                                              seed, val_frac, 0.0)
    return Split(name, rest[sub_tr], rest[sub_va], test,
                 f"Hold out the joint region {conditions}. Tests whether a "
                 "whole operating region can be predicted from its surroundings.")


def split_corner_extrapolation(X: np.ndarray, variables, cls: np.ndarray,
                               quantile: float = 0.85, seed: int = 0,
                               val_frac: float = 0.111) -> Split:
    """Train on the interior, test beyond it.

    For each named variable, rows above its `quantile` level go to test. This
    is extrapolation, and it is where a surrogate is expected to fail -- which
    is exactly why the paper should report it.
    """
    mask = np.zeros(len(X), dtype=bool)
    thresholds = {}
    for var in variables:
        col = X[:, FEATURE_COLS.index(var)]
        levels = np.unique(col)
        thr = float(np.quantile(levels, quantile))
        thresholds[var] = thr
        mask |= col > thr
    test = np.flatnonzero(mask)
    rest = np.flatnonzero(~mask)
    sub_tr, sub_va, _ = _stratified_three_way(make_strata(X[rest], cls[rest]),
                                              seed, val_frac, 0.0)
    return Split("S4_corner", rest[sub_tr], rest[sub_va], test,
                 f"Extrapolation beyond {thresholds}. Expected to be the "
                 "weakest regime; establishes the honest limit of the surrogate.")


def load_split(root: str | Path, name: str) -> Split:
    d = Path(root) / name
    meta = json.loads((d / "meta.json").read_text())
    parts = {}
    for part in ("train", "val", "test"):
        arr = np.load(d / f"{part}.npy")
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        if digest != meta["sha256"][part]:
            raise ValueError(f"{name}/{part}.npy checksum mismatch -- file changed on disk")
        parts[part] = arr
    return Split(name, parts["train"], parts["val"], parts["test"],
                 meta["description"])
