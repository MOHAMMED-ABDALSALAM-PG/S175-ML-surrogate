"""Evaluate the ORIGINAL (v1) trained checkpoints under the v2 protocol.

  python scripts/07_eval_v1_checkpoint.py

Loads the step7_8_full.py artifacts (uniform alpha=1.5, trained May 2026,
103.1 h) from ~/S175_models/ and scores them on the SAME S1_random test rows
and with the SAME metric code as the v3 runs, so the v1-vs-v3 comparison in
the paper is apples to apples: identical test rows, identical clamping,
identical masking discipline.

Honesty caveat, recorded in the output: v1 was trained on its own 80% random
split drawn with a different RNG, so roughly 80% of the v2 test rows were in
v1's TRAINING set. The comparison is therefore biased IN v1's FAVOUR; any v3
advantage shown is a lower bound.

Outputs:
  results/evaluation/v1_checkpoint_eval.json
  results/evaluation/v1_vs_v3_table.csv
  figures/fig_v1_vs_v3_safety.(png|pdf)
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from s175 import data as sdata
from s175.columns import (OUTPUT_COLS, SAFE_DIRECTION, CLASS_VALID,
                          CLASS_FUEL_ONLY, CLASS_ALL_NEG, CLASS_NAMES)
from s175.metrics import binary_classifier, clamp_physical, wilson_interval
from s175.models import MLP
from s175.splits import load_split

ev = __import__("03_evaluate")

V1_MODELS = Path.home() / "S175_models"
RAW = ev.RAW
CHUNK = 2_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SkScaler:
    """StandardScaler statistics without needing the sklearn object at runtime."""

    def __init__(self, pkl_path: Path):
        with open(pkl_path, "rb") as f:
            sk = pickle.load(f)
        self.mean = np.asarray(sk.mean_, dtype=np.float32)
        self.std = np.asarray(sk.scale_, dtype=np.float32)

    def transform(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    def inverse(self, a):
        return (a * self.std + self.mean).astype(np.float32)


def load_v1(device):
    reg = MLP(9, 10, hidden=(512, 256, 128), batchnorm=True, bn_after_act=True)
    reg.load_state_dict(torch.load(V1_MODELS / "regressor_full.pth",
                                   map_location="cpu", weights_only=True))
    c1 = MLP(9, 2, hidden=(512, 256, 128), batchnorm=True, bn_after_act=True)
    c1.load_state_dict(torch.load(V1_MODELS / "clf1_full.pth",
                                  map_location="cpu", weights_only=True))
    c2 = MLP(9, 2, hidden=(512, 256, 128, 64), batchnorm=True, bn_after_act=True)
    c2.load_state_dict(torch.load(V1_MODELS / "clf2_full.pth",
                                  map_location="cpu", weights_only=True))
    for m in (reg, c1, c2):
        m.to(device).eval()
    x_scaler = SkScaler(V1_MODELS / "input_scaler.pkl")
    y_scaler = SkScaler(V1_MODELS / "output_scaler.pkl")
    return (reg, c1, c2), x_scaler, y_scaler


@torch.no_grad()
def predict_v1(models, X, idx, x_scaler, device):
    reg, c1, c2 = models
    n = len(idx)
    y_scaled = np.empty((n, 10), dtype=np.float32)
    inf1 = np.empty(n, dtype=bool)   # argmax==1  <=>  logit1 > logit0
    inf2 = np.empty(n, dtype=bool)
    for s in range(0, n, CHUNK):
        xb = torch.from_numpy(x_scaler.transform(np.asarray(X[idx[s:s + CHUNK]]))).to(device)
        y_scaled[s:s + CHUNK] = reg(xb).float().cpu().numpy()
        l1 = c1(xb).float().cpu().numpy()
        l2 = c2(xb).float().cpu().numpy()
        inf1[s:s + CHUNK] = l1[:, 1] > l1[:, 0]
        inf2[s:s + CHUNK] = l2[:, 1] > l2[:, 0]
    return y_scaled, inf1, inf2


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, x_scaler, y_scaler = load_v1(device)
    log(f"v1 checkpoints loaded from {V1_MODELS} on {device}")

    arr, _ = sdata.load(RAW, cache_dir=ev.CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    sp = load_split(ev.SPLIT_DIR, "S1_random")
    te = sp.test
    log(f"{len(te):,} S1_random test rows")

    t0 = time.perf_counter()
    y_scaled, inf1, inf2 = predict_v1(models, X, te, x_scaler, device)
    log(f"inference {time.perf_counter() - t0:.1f}s")

    y_pred = clamp_physical(y_scaler.inverse(y_scaled))
    y_true = np.asarray(Y[te])
    cls_te = cls[te]

    reg_out, reg_meta = ev.regression_metrics(y_true, y_pred, cls_te)
    lab1 = cls_te == CLASS_ALL_NEG
    m1 = binary_classifier(lab1, inf1)
    sub2 = cls_te != CLASS_ALL_NEG
    m2 = binary_classifier(cls_te[sub2] == CLASS_FUEL_ONLY, inf2[sub2])

    pred_cls = np.full(len(te), CLASS_VALID, dtype=np.int8)
    pred_cls[inf2] = CLASS_FUEL_ONLY
    pred_cls[inf1] = CLASS_ALL_NEG
    conf = np.zeros((3, 3), dtype=np.int64)
    for t in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG):
        for p in (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG):
            conf[t, p] = int(np.sum((cls_te == t) & (pred_cls == p)))
    allneg_missed = conf[CLASS_ALL_NEG, CLASS_VALID] + conf[CLASS_ALL_NEG, CLASS_FUEL_ONLY]
    n_allneg = conf[CLASS_ALL_NEG].sum()

    result = {
        "model": "v1_original_checkpoint (step7_8_full.py, uniform alpha=1.5)",
        "checkpoint_dir": str(V1_MODELS),
        "caveat": "v1 trained on its own random 80% split; ~80% of these test "
                  "rows were in v1's training set, biasing this comparison in "
                  "v1's favour.",
        "split": "S1_random", "n_test_rows": int(len(te)),
        "regressor": {"meta": reg_meta, "per_output": reg_out},
        "clf1": m1, "clf2": m2,
        "end_to_end": {
            "confusion_matrix_true_x_pred": conf.tolist(),
            "class_order": [CLASS_NAMES[c] for c in
                            (CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG)],
            "accuracy": float(np.trace(conf) / conf.sum()),
            "dangerous_allneg_passed": int(allneg_missed),
            "dangerous_allneg_rate": float(allneg_missed / n_allneg),
            "dangerous_allneg_rate_ci95": list(wilson_interval(int(allneg_missed), int(n_allneg))),
        },
    }
    out = ROOT / "results" / "evaluation"
    (out / "v1_checkpoint_eval.json").write_text(json.dumps(result, indent=2))
    log("wrote v1_checkpoint_eval.json")

    # ------- comparison table + figure against v3 (same rows, same code) ----
    v3 = json.loads((ROOT / "runs" / "v3_masked_peroutput" / "S1_random" /
                     "seed_0" / "eval.json").read_text())
    with open(out / "v1_vs_v3_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["output", "safe_direction",
                    "v1_R2", "v3_R2", "v1_MAE", "v3_MAE",
                    "v1_overpred_pct_positive", "v3_overpred_pct_positive",
                    "v1_safe_direction_met", "v3_safe_direction_met"])
        for c in OUTPUT_COLS:
            a = reg_out[c]
            b = v3["regressor"]["per_output"][c]
            w.writerow([c, SAFE_DIRECTION[c], a["R2"], b["R2"], a["MAE"], b["MAE"],
                        a["overpred_pct_positive"], b["overpred_pct_positive"],
                        a["safe_direction_met"], b["safe_direction_met"]])
    log("wrote v1_vs_v3_table.csv")

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    xs = np.arange(len(OUTPUT_COLS))
    wd = 0.38
    v1_rates = [reg_out[c]["overpred_pct_positive"] for c in OUTPUT_COLS]
    v3_rates = [v3["regressor"]["per_output"][c]["overpred_pct_positive"]
                for c in OUTPUT_COLS]
    ax.bar(xs - wd / 2, v1_rates, wd, label="v1: uniform α=1.5, no mask",
           color="#b03a2e")
    ax.bar(xs + wd / 2, v3_rates, wd, label="v3: per-output α, masked",
           color="#31538f")
    ax.axhline(50, color="k", lw=0.8, ls=":")
    for i, c in enumerate(OUTPUT_COLS):
        want_over = SAFE_DIRECTION[c] == "over"
        ax.annotate("↑ safe" if want_over else "↓ safe", (i, 101),
                    ha="center", fontsize=7.5, color="#1e8449")
    ax.set_xticks(xs, OUTPUT_COLS, rotation=30, ha="right")
    ax.set_ylabel("over-prediction rate on positive-truth rows [%]")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("Which side does each output err on? v1 vs v3, same test rows "
                 "(above 50% = biased toward over-prediction)")
    fig.savefig(ROOT / "figures" / "fig_v1_vs_v3_safety.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_v1_vs_v3_safety.pdf", bbox_inches="tight")
    log("wrote figures/fig_v1_vs_v3_safety.png")

    log(f"v1 under v2 protocol: e2e acc {result['end_to_end']['accuracy']:.4f} "
        f"| clf1 FNR {m1['false_negative_rate']:.2e} | safe-direction "
        f"{reg_meta['safe_direction_met']}/{reg_meta['safe_direction_total']} "
        f"(v3: {v3['regressor']['meta']['safe_direction_met']}/10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
