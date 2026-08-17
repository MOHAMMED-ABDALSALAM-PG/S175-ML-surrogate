"""Evaluation metrics.

Changes against the original `step7_8_full.py` metric block:

*  Predictions are clamped to physical bounds before scoring. The unclamped
   v1 model predicted a negative slamming probability on 86.7% of rows.
*  Over-prediction rate is reported both raw and restricted to rows whose true
   value exceeds a stated threshold. The raw statistic is dominated by the zero
   mass for slamming (31% exact zeros) and green water (27%).
*  Relative error is reported against the standard deviation as well as the
   range. Range-normalisation understates the error by roughly 7-18x.
*  Classifier metrics carry denominators and Wilson confidence intervals, not
   bare false-negative counts.
"""
from __future__ import annotations

import math

import numpy as np

from .columns import OUTPUT_COLS, PHYSICAL_BOUNDS, SAFE_DIRECTION


def clamp_physical(y: np.ndarray, cols=OUTPUT_COLS) -> np.ndarray:
    """Clamp each column to its physical range. Returns a new array."""
    out = y.copy()
    for j, c in enumerate(cols):
        lo, hi = PHYSICAL_BOUNDS[c]
        if lo is not None:
            np.clip(out[:, j], lo, None, out=out[:, j])
        if hi is not None:
            np.clip(out[:, j], None, hi, out=out[:, j])
    return out


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def per_output(y_true: np.ndarray, y_pred: np.ndarray,
               positive_threshold: float = 1e-6,
               cols=OUTPUT_COLS) -> dict[str, dict]:
    """Full per-output metric set on the original scale."""
    res: dict[str, dict] = {}
    for j, c in enumerate(cols):
        t, p = y_true[:, j], y_pred[:, j]
        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        sd = float(np.std(t))
        rng = float(t.max() - t.min())
        pos = t > positive_threshold
        want_over = SAFE_DIRECTION[c] == "over"
        oe_all = float(100.0 * np.mean(p > t))
        oe_pos = float(100.0 * np.mean(p[pos] > t[pos])) if pos.any() else float("nan")
        res[c] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2(t, p),
            "max_abs_error": float(np.max(np.abs(err))),
            "bias_mean_error": float(np.mean(err)),
            "overpred_pct_all": oe_all,
            "overpred_pct_positive": oe_pos,
            "n_positive": int(pos.sum()),
            "rel_MAE_range_pct": 100.0 * mae / rng if rng > 0 else float("nan"),
            "rel_MAE_sd_pct": 100.0 * mae / sd if sd > 0 else float("nan"),
            # MAPE is undefined at y = 0 and slamming is 31% exact zeros, green
            # water 27%. Reporting it over all rows would divide by zero; over
            # near-zero rows it explodes without saying anything about accuracy.
            # So it is computed on positive-truth rows only, with the count of
            # excluded rows carried alongside it, and sMAPE given as the
            # all-rows alternative that stays finite.
            "MAPE_positive_pct": (
                float(100.0 * np.mean(np.abs(err[pos] / t[pos]))) if pos.any()
                else float("nan")
            ),
            "n_excluded_from_MAPE": int((~pos).sum()),
            "sMAPE_pct": float(
                100.0 * np.mean(
                    np.abs(err) / np.maximum((np.abs(t) + np.abs(p)) / 2.0, 1e-12)
                )
            ),
            "safe_direction": SAFE_DIRECTION[c],
            # judged on the positive-restricted rate, which is the honest one
            "safe_direction_met": bool(
                (oe_pos > 50.0) if want_over else (oe_pos < 50.0)
            ) if not math.isnan(oe_pos) else None,
            "pct_below_zero": float(100.0 * np.mean(p < 0.0)),
        }
    return res


def safe_direction_score(per_out: dict[str, dict]) -> tuple[int, int]:
    """How many outputs err on their safe side. Returns (met, total)."""
    vals = [v["safe_direction_met"] for v in per_out.values()
            if v["safe_direction_met"] is not None]
    return int(sum(vals)), len(vals)


def wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used for recall / false-negative rate, where the normal approximation is
    unreliable at the very small rates involved (FN rate ~0.15%).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def binary_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix plus the safety-relevant rates.

    Positive class = infeasible. A false negative is therefore the dangerous
    error: the surrogate returns numbers where the simulator has none.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    n_pos, n_neg = tp + fn, tn + fp
    recall = tp / n_pos if n_pos else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    fnr = fn / n_pos if n_pos else float("nan")
    lo, hi = wilson_interval(fn, n_pos)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "n_positive": n_pos, "n_negative": n_neg, "n_total": int(len(y_true)),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else float("nan"),
        "precision": float(prec), "recall": float(recall),
        "f1": float(2 * prec * recall / (prec + recall)) if (prec + recall) else float("nan"),
        "false_negative_rate": float(fnr),
        "false_negative_rate_ci95": [float(lo), float(hi)],
        "specificity": float(tn / n_neg) if n_neg else float("nan"),
    }


def aggregate_seeds(runs: list[dict[str, dict]], metric: str = "MAE",
                    cols=OUTPUT_COLS) -> dict[str, dict]:
    """Median and 95% range of a metric across seeds, per output.

    With five seeds the percentile interval is coarse; we report the observed
    min/max alongside the median rather than implying a smooth distribution.
    """
    out = {}
    for c in cols:
        vals = np.array([r[c][metric] for r in runs], dtype=float)
        out[c] = {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()),
            "n_seeds": int(len(vals)),
            "values": [float(v) for v in vals],
        }
    return out
