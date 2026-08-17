"""Network definitions.

The architectures match the ones the original study selected in its tuning step
(regressor 512-256-128, C1 512-256-128, C2 512-256-128-64), so that the rewrite
changes the training procedure and the evaluation -- not the model capacity.
Keeping them identical is what makes the corrected results comparable to the
published ones.

Two deliberate differences from the original `step7_8_full.py`:

*  Layer order. The original placed BatchNorm *after* the activation
   (Linear -> ReLU -> BatchNorm). Both orderings train, but Linear -> BN -> ReLU
   is the form BatchNorm was proposed with and it makes the bias in the
   preceding Linear redundant. The order is exposed as a flag so the original
   can be reproduced exactly when comparing against the published run.

*  The classifiers emit a single logit rather than two, so the decision
   threshold is an explicit, tunable number instead of an implicit argmax at
   0.5. The rationale: for a safety-critical false-negative rate,
   the threshold must be selected on validation, not assumed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .columns import FEATURE_COLS, OUTPUT_COLS


def _block(in_f: int, out_f: int, batchnorm: bool, bn_after_act: bool,
           dropout: float) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Linear(in_f, out_f, bias=not (batchnorm and not bn_after_act))]
    if batchnorm and bn_after_act:
        layers += [nn.ReLU(), nn.BatchNorm1d(out_f)]
    elif batchnorm:
        layers += [nn.BatchNorm1d(out_f), nn.ReLU()]
    else:
        layers += [nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class MLP(nn.Module):
    """Plain multi-layer perceptron with a linear head."""

    def __init__(self, in_features: int, out_features: int,
                 hidden=(512, 256, 128), batchnorm: bool = True,
                 bn_after_act: bool = False, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += _block(prev, h, batchnorm, bn_after_act, dropout)
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.network = nn.Sequential(*layers)
        self.config = {
            "in_features": in_features, "out_features": out_features,
            "hidden": list(hidden), "batchnorm": batchnorm,
            "bn_after_act": bn_after_act, "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_regressor(hidden=(512, 256, 128), **kw) -> MLP:
    return MLP(len(FEATURE_COLS), len(OUTPUT_COLS), hidden=hidden, **kw)


def build_classifier(hidden=(512, 256, 128), **kw) -> MLP:
    """Single-logit binary classifier; apply a sigmoid for the probability."""
    return MLP(len(FEATURE_COLS), 1, hidden=hidden, **kw)


class Surrogate(nn.Module):
    """The assembled two-stage surrogate, for end-to-end inference and timing.

    Mirrors the manuscript's inference rule exactly:

        if C1(x) -> every output is infeasible
        else     -> y = R(x), and if C2(x) the fuel output alone is infeasible

    Both thresholds are explicit so they can be selected on the validation
    split rather than left at an implicit 0.5.
    """

    def __init__(self, clf1: nn.Module, clf2: nn.Module, regressor: nn.Module,
                 thr1: float = 0.5, thr2: float = 0.5,
                 infeasible_value: float = -1.0, fuel_idx: int | None = None):
        super().__init__()
        self.clf1, self.clf2, self.regressor = clf1, clf2, regressor
        self.thr1, self.thr2 = thr1, thr2
        self.infeasible_value = infeasible_value
        self.fuel_idx = OUTPUT_COLS.index("fuel") if fuel_idx is None else fuel_idx

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.regressor(x)
        p1 = torch.sigmoid(self.clf1(x)).squeeze(-1)
        p2 = torch.sigmoid(self.clf2(x)).squeeze(-1)
        fuel_only = p2 > self.thr2
        y[fuel_only, self.fuel_idx] = self.infeasible_value
        all_neg = p1 > self.thr1
        y[all_neg, :] = self.infeasible_value
        return y
