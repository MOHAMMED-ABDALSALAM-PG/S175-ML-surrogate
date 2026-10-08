#!/usr/bin/env python3
"""Model-family screening on one shared partition: single regressor (step 1) vs two-stage (step 2).

Both screening steps of the model-family comparison (scripts 17 and 17b) are repeated on ONE
partition of the stratified 5 % subsample -- the 80/10/10 split of script 17 (strata: Hs band x
V_wind band x speed-infeasible flag, random_state 42) -- so that the two approaches are scored on
identical test rows. Hyperparameters are those of scripts 17/17b; only the training seed varies.

  A  single ship-speed regressor trained on all rows, complete infeasibility encoded as -1;
     infeasible := prediction <= t, t = the validation-maximum-F1 point of the exact
     precision-recall curve
  B  classifier of complete infeasibility (threshold 0.5, and the validation-maximum-F1 point of
     the exact precision-recall curve) + ship-speed regressor trained on valid rows only

Metrics on the test partition: infeasibility F1 / precision / recall / missed / false alarms, and
ship-speed MAE, RMSE, R^2 on (i) every test row whose true speed is valid (identical rows for A
and B) and (ii) rows that are valid and also predicted valid by the approach's own detector.

Usage:
  17c_screening_comparison.py --approach A --family MLP --seed 0 [--threads 4] [--limit N]
  17c_screening_comparison.py --aggregate
One JSON per (approach, family, seed) in results/evaluation/screening_comparison/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluation" / "screening_comparison"
DATA_PATH = Path.home() / "S175_sample_5pct_stratified.csv"
INPUT_FEATURES = ["draft", "trim", "rpm", "PD", "Hs", "Tp", "Chi", "Vwind", "Theta_wind"]
TARGET = "speed"
SPLIT_STATE = 42                                   # partition of script 17 (fixed)
SEEDS = (0, 1, 2)
FAMILIES = ("RF", "XGB", "MLP")


def load_partition(limit: int | None):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA_PATH, sep=";", usecols=INPUT_FEATURES + [TARGET])
    X = df[INPUT_FEATURES].to_numpy(np.float64)
    y = df[TARGET].to_numpy(np.float64)
    # identical to script 17 (lines 92-103)
    hs_bins = np.digitize(df["Hs"].to_numpy(), bins=[2.01, 5.01, 8.01])
    vw_bins = np.digitize(df["Vwind"].to_numpy(), bins=[7.5, 17.5])
    strata = hs_bins * 100 + vw_bins * 10 + (y == -1).astype(int)
    idx = np.arange(len(df))
    tr, tmp, _, st_tmp = train_test_split(idx, strata, test_size=0.2,
                                          random_state=SPLIT_STATE, stratify=strata)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SPLIT_STATE, stratify=st_tmp)
    if limit:                                      # smoke tests only
        rng = np.random.default_rng(0)
        tr, va, te = (np.sort(rng.choice(p, min(limit, len(p)), replace=False)) for p in (tr, va, te))
    h = lambda a: hashlib.sha256(np.sort(a).astype(np.int64).tobytes()).hexdigest()
    return X, y, tr, va, te, {"train": h(tr), "val": h(va), "test": h(te)}


def sha_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            d.update(chunk)
    return d.hexdigest()


def make_model(family: str, kind: str, seed: int, threads: int):
    if family == "RF":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if kind == "clf" else RandomForestRegressor
        return cls(n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=threads, random_state=seed)
    if family == "XGB":
        import xgboost as xgb
        kw = dict(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, n_jobs=threads, random_state=seed,
                  tree_method="hist", verbosity=0)
        return xgb.XGBClassifier(eval_metric="logloss", **kw) if kind == "clf" else xgb.XGBRegressor(**kw)
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    cls = MLPClassifier if kind == "clf" else MLPRegressor
    return cls(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
               learning_rate="adaptive", learning_rate_init=0.001, max_iter=300,
               early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
               batch_size=1024, random_state=seed, verbose=False)


def fit(model, family: str, X, y, threads: int):
    """MLPs see inputs standardised on their own training rows; trees see raw inputs.
    Returns (input transform, fit seconds)."""
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    scaler = StandardScaler().fit(X) if family == "MLP" else None
    t0 = time.perf_counter()
    with threadpool_limits(limits=threads):
        model.fit(scaler.transform(X) if scaler else X, y)
    return (lambda Z: scaler.transform(Z) if scaler else Z), time.perf_counter() - t0


def predict(model, Z, threads: int, proba: bool = False):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        return model.predict_proba(Z)[:, 1] if proba else model.predict(Z)


def detection(is_inf_true, is_inf_pred):
    tp = int(np.sum(is_inf_true & is_inf_pred)); fn = int(np.sum(is_inf_true & ~is_inf_pred))
    fp = int(np.sum(~is_inf_true & is_inf_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "missed_infeasible": fn, "false_alarms": fp}


def best_threshold(is_inf, score):
    """Exact validation-maximum-F1 threshold on a score where larger = more infeasible;
    a row is flagged infeasible when score >= threshold. Ties: the first (lowest) threshold."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(is_inf.astype(int), score)
    prec, rec = prec[:-1], rec[:-1]
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(den), where=den > 0)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def regression(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    n = int(len(y_true))
    if n == 0:
        return {"n_rows": 0, "MAE": None, "RMSE": None, "R2": None}
    return {"n_rows": n, "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)) if n > 1 else None}


def run(approach: str, family: str, seed: int, threads: int, limit: int | None) -> dict:
    t_all = time.time()
    X, y, tr, va, te, digests = load_partition(limit)
    inf = y == -1
    res = {"approach": approach, "family": family, "seed": seed, "split_state": SPLIT_STATE,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "index_sha256": digests, "data_sha256": sha_file(DATA_PATH),
           "script_sha256": sha_file(Path(__file__)), "limit": limit}
    if approach == "A":
        reg = make_model(family, "reg", seed, threads)
        tf, t_fit = fit(reg, family, X[tr], y[tr], threads)
        res["fit_seconds"] = t_fit
        res["params"] = {"regressor": {k: repr(v) for k, v in reg.get_params().items()}}
        p_va = predict(reg, tf(X[va]), threads); p_te = predict(reg, tf(X[te]), threads)
        s, f1_va = best_threshold(inf[va], -p_va)          # infeasible when prediction <= -s
        pred_inf = p_te <= -s
        res["detection"] = {"threshold_on_prediction": -s, "val_f1": f1_va,
                            **detection(inf[te], pred_inf)}
        speed_pred = p_te
    else:
        clf = make_model(family, "clf", seed, threads)
        tf, t_clf = fit(clf, family, X[tr], inf[tr].astype(int), threads)
        pr_va = predict(clf, tf(X[va]), threads, proba=True)
        pr_te = predict(clf, tf(X[te]), threads, proba=True)
        s, f1_va = best_threshold(inf[va], pr_va)           # infeasible when probability >= s
        res["detection_default_0p5"] = detection(inf[te], pr_te >= 0.5)
        pred_inf = pr_te >= s
        res["detection"] = {"threshold_on_probability": s, "val_f1": f1_va,
                            **detection(inf[te], pred_inf)}
        valid_tr = tr[~inf[tr]]
        reg = make_model(family, "reg", seed, threads)
        tfr, t_reg = fit(reg, family, X[valid_tr], y[valid_tr], threads)
        res["fit_seconds"] = t_clf + t_reg
        res["fit_seconds_parts"] = {"classifier": t_clf, "regressor": t_reg}
        res["params"] = {"classifier": {k: repr(v) for k, v in clf.get_params().items()},
                         "regressor": {k: repr(v) for k, v in reg.get_params().items()}}
        speed_pred = predict(reg, tfr(X[te]), threads)
    valid_te = ~inf[te]
    res["speed_all_valid_rows"] = regression(y[te][valid_te], speed_pred[valid_te])
    both = valid_te & ~pred_inf
    res["speed_valid_and_predicted_valid"] = regression(y[te][both], speed_pred[both])
    res["seconds_total"] = time.time() - t_all
    import sklearn
    res["env"] = {"python": platform.python_version(), "numpy": np.__version__,
                  "sklearn": sklearn.__version__, "host": platform.node(), "threads": threads}
    if family == "XGB":
        import xgboost
        res["env"]["xgboost"] = xgboost.__version__
    return res


def aggregate() -> None:
    rows, seen = {}, set()
    for f in sorted(OUT.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        if r.get("limit"):
            continue
        key = (r["approach"], r["family"], r["seed"])
        if key in seen:
            raise SystemExit(f"duplicate result for {key}")
        seen.add(key)
        rows.setdefault((r["approach"], r["family"]), []).append(r)
    expected = {(a, f, s) for a in ("A", "B") for f in FAMILIES for s in SEEDS}
    if seen != expected:
        raise SystemExit(f"incomplete: missing {sorted(expected - seen)}")
    fps = {(json.dumps(x["index_sha256"], sort_keys=True), x["data_sha256"], x["script_sha256"])
           for v in rows.values() for x in v}
    if len(fps) != 1:
        raise SystemExit("results come from different data, partitions or script versions")
    fp = next(iter(fps))
    summary = {"n_results": len(seen), "index_sha256": json.loads(fp[0]), "data_sha256": fp[1],
               "script_sha256": fp[2], "cells": {}}
    ms = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "values": [float(x) for x in v]}
    for (a, fam), v in sorted(rows.items()):
        v = sorted(v, key=lambda r: r["seed"])
        cell = {"seeds": [r["seed"] for r in v],
                "f1": ms([r["detection"]["f1"] for r in v]),
                "missed_infeasible": ms([r["detection"]["missed_infeasible"] for r in v]),
                "false_alarms": ms([r["detection"]["false_alarms"] for r in v]),
                "speed_MAE": ms([r["speed_all_valid_rows"]["MAE"] for r in v]),
                "speed_R2": ms([r["speed_all_valid_rows"]["R2"] for r in v]),
                "speed_MAE_predicted_valid": ms([r["speed_valid_and_predicted_valid"]["MAE"] for r in v]),
                "speed_R2_predicted_valid": ms([r["speed_valid_and_predicted_valid"]["R2"] for r in v]),
                "fit_seconds": ms([r["fit_seconds"] for r in v])}
        if a == "B":
            d5 = [r["detection_default_0p5"] for r in v]
            cell["default_0p5"] = {"f1": ms([d["f1"] for d in d5]),
                                   "missed_infeasible": ms([d["missed_infeasible"] for d in d5]),
                                   "false_alarms": ms([d["false_alarms"] for d in d5])}
        summary["cells"][f"{a}_{fam}"] = cell
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", choices=("A", "B"))
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="subsample each partition (smoke test)")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.aggregate:
        aggregate()
        return 0
    if not (a.approach and a.family and a.seed is not None):
        ap.error("--approach, --family and --seed are required")
    res = run(a.approach, a.family, a.seed, a.threads, a.limit)
    tag = f"{a.approach}_{a.family}_seed{a.seed}" + (f"_limit{a.limit}" if a.limit else "")
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    sp = res["speed_all_valid_rows"]
    print(f"{tag}: F1 {res['detection']['f1']:.4f}  speed MAE {sp['MAE']}  R2 {sp['R2']}"
          f"  ({res['seconds_total']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
