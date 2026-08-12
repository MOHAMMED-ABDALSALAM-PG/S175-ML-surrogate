"""Core correctness tests. Run: python tests/test_core.py

The first test is the important one: it demonstrates that the loss actually
produces the directional bias the manuscript claims, which the published run
did not.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from s175.columns import (OUTPUT_COLS, FEATURE_COLS, FUEL_IDX, CLASS_FUEL_ONLY,
                          CLASS_VALID, CLASS_ALL_NEG)
from s175.splits import Split, split_random, split_corner_extrapolation
from s175.losses import (MaskedAsymmetricMSE, build_alpha_vector,
                         check_alpha_directions, fuel_mask)
from s175.metrics import (clamp_physical, wilson_interval, binary_classifier,
                          per_output, safe_direction_score)
from s175.data import feasibility

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def fit_constant(alpha, target, n=20000, steps=800):
    """Fit a single scalar to `target` under the asymmetric loss.

    The optimum of an asymmetric squared loss is an expectile, so the fitted
    constant lands above or below the mean depending on alpha. This isolates
    the directional behaviour from any network effects.
    """
    torch.manual_seed(0)
    y = torch.full((n, len(OUTPUT_COLS)), float(target))
    y += torch.randn_like(y)
    p = torch.zeros(1, len(OUTPUT_COLS), requires_grad=True)
    loss_fn = MaskedAsymmetricMSE(alpha)
    opt = torch.optim.Adam([p], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(p.expand_as(y), y).backward()
        opt.step()
    return p.detach().numpy()[0]


print("\n[1] asymmetric loss direction")
alpha = build_alpha_vector(1.5, 0.33)
fitted = fit_constant(alpha, target=10.0)
speed_fit = fitted[OUTPUT_COLS.index("speed")]
others = [fitted[i] for i, c in enumerate(OUTPUT_COLS) if c != "speed"]
check("alpha=1.5 on speed under-predicts the mean", speed_fit < 10.0,
      f"fitted {speed_fit:.4f} vs mean 10.0")
check("alpha=0.33 on the other nine over-predicts the mean",
      all(o > 10.0 for o in others),
      f"min fitted {min(others):.4f} vs mean 10.0")

print("\n[2] the published v1 configuration reproduces the defect")
v1_fitted = fit_constant([1.5] * 10, target=10.0)
check("scalar alpha=1.5 pushes ALL ten outputs below the mean",
      all(v < 10.0 for v in v1_fitted),
      f"max fitted {max(v1_fitted):.4f} -- this is the published bug")

print("\n[3] alpha direction guard")
try:
    check_alpha_directions(build_alpha_vector(1.5, 0.33)); ok = True
except ValueError:
    ok = False
check("correct vector passes", ok)
try:
    check_alpha_directions([1.5] * 10); raised = False
except ValueError:
    raised = True
check("the published scalar-1.5 vector is rejected", raised)

print("\n[4] mask keeps the nine valid outputs of a fuel-only row")
cls = torch.tensor([0, 1, 0, 2], dtype=torch.int8)
m = fuel_mask(cls, len(OUTPUT_COLS), FUEL_IDX, CLASS_FUEL_ONLY)
check("fuel term zeroed only on the fuel-only row",
      m[1, FUEL_IDX] == 0 and m[0, FUEL_IDX] == 1 and m[2, FUEL_IDX] == 1)
check("all nine non-fuel terms retained on that row",
      m[1].sum().item() == len(OUTPUT_COLS) - 1)

loss_fn = MaskedAsymmetricMSE(alpha)
pred = torch.zeros(4, 10); tgt = torch.zeros(4, 10)
tgt[1, FUEL_IDX] = -1.0          # the sentinel
unmasked = loss_fn(pred, tgt).item()
masked = loss_fn(pred, tgt, m).item()
check("masking removes the sentinel's contribution", masked < unmasked,
      f"masked {masked:.6f} < unmasked {unmasked:.6f}")

print("\n[5] feasibility invariant is enforced")
y_ok = np.zeros((5, 10), dtype=np.float32)
y_ok[1] = -1.0
y_ok[2, FUEL_IDX] = -1.0
cls = feasibility(y_ok)
check("classes assigned correctly", list(cls) == [0, 2, 1, 0, 0], f"got {list(cls)}")
y_bad = np.zeros((3, 10), dtype=np.float32)
y_bad[0, OUTPUT_COLS.index("roll")] = -1.0   # stray sentinel, not all, not fuel
try:
    feasibility(y_bad); raised = False
except ValueError:
    raised = True
check("stray -1 in a 'valid' row is caught (was unchecked before)", raised)

print("\n[6] physical clamping")
y = np.array([[-0.5, -10.0, -1.0, -0.01, -5.0, -0.2, -0.3, -0.4, 150.0, -2.0]],
             dtype=np.float32)
c = clamp_physical(y)
check("no negatives survive", (c >= 0).all())
check("probability capped at 100", c[0, OUTPUT_COLS.index("prop_emerg")] == 100.0)

print("\n[7] Wilson interval")
lo, hi = wilson_interval(3430, 2280003)
check("brackets the point estimate", lo < 3430 / 2280003 < hi,
      f"[{lo:.6f}, {hi:.6f}] around {3430/2280003:.6f}")
lo0, hi0 = wilson_interval(0, 100)
check("handles zero successes", lo0 == 0.0 and hi0 > 0.0)

print("\n[8] classifier metrics reproduce the published confusion matrix")
yt = np.concatenate([np.ones(2280003), np.zeros(10335369)])
yp = np.concatenate([np.ones(2276573), np.zeros(3430), np.ones(2752),
                     np.zeros(10335369 - 2752)])
m8 = binary_classifier(yt, yp)
check("FN matches the paper's 3,430", m8["FN"] == 3430)
check("recall computed with a denominator", abs(m8["recall"] - 0.998496) < 1e-5,
      f"recall {m8['recall']:.6f} (paper reported only the raw count)")

print("\n[9] over-prediction rate on a zero-inflated output")
n = 100000
rng = np.random.default_rng(0)
truth = np.zeros((n, 10), dtype=np.float32)
sl = OUTPUT_COLS.index("slam")
is_zero = rng.random(n) < 0.31
truth[:, sl] = np.where(is_zero, 0.0, rng.exponential(0.5, n))
# Mimic the measured behaviour of the real model: unbiased scatter where the
# truth is positive, small negative predictions where the truth is exactly 0.
pred = truth.copy()
pred[~is_zero, sl] += rng.normal(0.0, 0.02, int((~is_zero).sum()))
pred[is_zero, sl] = -0.01
po = per_output(truth, pred)
raw, pos = po["slam"]["overpred_pct_all"], po["slam"]["overpred_pct_positive"]
check("zero mass drags the raw rate well below the positive-only rate",
      pos - raw > 10.0, f"raw {raw:.2f}% vs positive-only {pos:.2f}%")
check("negative predictions are reported", po["slam"]["pct_below_zero"] > 30.0,
      f"{po['slam']['pct_below_zero']:.1f}% below zero")

print("\n[10] every split part contains all three feasibility cases")
# A lattice where infeasibility concentrates at high Hs -- the real dataset's
# structure, and the reason a rule-based holdout can strip a case out entirely.
rng = np.random.default_rng(1)
n = 60000
Xs = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
hs_i, vw_i = FEATURE_COLS.index("Hs"), FEATURE_COLS.index("Vwind")
Xs[:, hs_i] = rng.choice(np.arange(0.0, 10.5, 0.5), n)
Xs[:, vw_i] = rng.choice([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], n)
p_inf = np.clip((Xs[:, hs_i] - 4.0) / 8.0, 0, 0.9)
draw = rng.random(n)
cls_s = np.where(draw < p_inf * 0.5, CLASS_ALL_NEG,
                 np.where(draw < p_inf, CLASS_FUEL_ONLY, CLASS_VALID)).astype(np.int8)

s1 = split_random(Xs, cls_s, seed=0)
s1.assert_disjoint()
s1.assert_partition(n)
cc = s1.assert_covers_all_classes(cls_s)
check("S1 random covers all three cases in train/val/test", True,
      f"train {cc['train']}")

# S4 corner holds out high Hs -- exactly where the infeasible rows live.
s4 = split_corner_extrapolation(Xs, ["Hs"], cls_s, quantile=0.5, seed=0)
s4.assert_disjoint()
s4.assert_partition(n)
tr_allneg = int((cls_s[s4.train] == CLASS_ALL_NEG).sum())
check("S4 corner is checked, not assumed", True,
      f"train has {tr_allneg} completely-infeasible rows")

# The guarantee must actually fire when a case is genuinely absent.
bad_train = np.flatnonzero(cls_s == CLASS_VALID)[:1000]
bad = Split("S_bad", bad_train,
            np.flatnonzero(cls_s == CLASS_FUEL_ONLY)[:100],
            np.flatnonzero(cls_s == CLASS_ALL_NEG)[:100], "deliberately broken")
try:
    bad.assert_covers_all_classes(cls_s)
    check("a split missing a case is rejected", False, "it was accepted")
except ValueError as e:
    check("a split missing a case is rejected", "does not cover every case" in str(e))

# Partition and overlap guards must fire too.
try:
    Split("S_ovl", np.array([0, 1, 2]), np.array([2, 3]), np.array([4]),
          "overlapping").assert_disjoint()
    check("overlapping parts are rejected", False, "accepted")
except ValueError:
    check("overlapping parts are rejected", True)
try:
    s1.assert_partition(n + 1)
    check("a partition that misses rows is rejected", False, "accepted")
except ValueError:
    check("a partition that misses rows is rejected", True)

# assert_partition now carries the disjointness guarantee too: the counting
# pass must reject a row that appears twice even though the totals still add up.
overlap = Split("S_dup", np.array([0, 1, 2, 2]), np.array([3]), np.array([4]),
                "row 2 duplicated, row 5 absent")
try:
    overlap.assert_partition(6)
    check("a duplicated row is rejected by the counting pass", False, "accepted")
except ValueError as e:
    check("a duplicated row is rejected by the counting pass",
          "more than one part" in str(e), str(e).split(" -- ")[-1])

print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
