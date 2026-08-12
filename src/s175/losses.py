"""Masked, per-output asymmetric squared-error loss.

This is the loss the manuscript describes and that the final published run did
not use. The published `step7_8_full.py` had:

    weights = torch.where(error > 0, self.alpha, 1.0)   # one scalar, all outputs
    return (weights * error ** 2).mean()                # no mask

which applied alpha=1.5 uniformly -- biasing every output toward
under-prediction, including the nine that should have been biased the other
way -- and dropped the fuel-only-infeasible rows entirely rather than masking
the fuel term.

Convention (Newey & Powell 1987 asymmetric least squares), with e = pred - y:

    alpha > 1  penalises over-prediction   -> model under-predicts
    alpha < 1  penalises under-prediction  -> model over-predicts

so ship speed takes alpha > 1 (never promise more speed than the simulator)
and the propulsion, fuel and seakeeping outputs take alpha < 1 (never
under-state required power or operational risk).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .columns import OUTPUT_COLS, SAFE_DIRECTION


def build_alpha_vector(alpha_speed: float = 1.5,
                       alpha_others: float = 0.33) -> list[float]:
    """Per-output alpha, ordered to match OUTPUT_COLS."""
    return [alpha_speed if c == "speed" else alpha_others for c in OUTPUT_COLS]


def check_alpha_directions(alpha_vec) -> None:
    """Assert each alpha points the way SAFE_DIRECTION says it should.

    Guards against the failure that produced the published table: an alpha
    vector whose values silently contradict the stated safety intent.
    """
    if len(alpha_vec) != len(OUTPUT_COLS):
        raise ValueError(f"alpha vector has {len(alpha_vec)} entries, expected {len(OUTPUT_COLS)}")
    for col, a in zip(OUTPUT_COLS, alpha_vec):
        want = SAFE_DIRECTION[col]
        if want == "under" and not a > 1.0:
            raise ValueError(f"{col} should be biased to under-predict, needs alpha > 1, got {a}")
        if want == "over" and not a < 1.0:
            raise ValueError(f"{col} should be biased to over-predict, needs alpha < 1, got {a}")


class MaskedAsymmetricMSE(nn.Module):
    """loss = sum_ij m_ij * w(e_ij) * e_ij^2 / sum_ij m_ij

    Parameters
    ----------
    alpha_per_output : sequence of float, length = number of outputs
    reduction_eps : guards a batch in which every element is masked
    """

    def __init__(self, alpha_per_output, reduction_eps: float = 1.0):
        super().__init__()
        alpha = torch.as_tensor(list(alpha_per_output), dtype=torch.float32)
        self.register_buffer("alpha", alpha)
        self.reduction_eps = reduction_eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        error = pred - target
        w = torch.where(error > 0, self.alpha.expand_as(error),
                        torch.ones_like(error))
        sq = w * error.pow(2)
        if mask is None:
            return sq.mean()
        sq = sq * mask
        denom = mask.sum()
        return sq.sum() / torch.clamp(denom, min=self.reduction_eps)


def fuel_mask(feasibility_cls: torch.Tensor, n_outputs: int,
              fuel_idx: int, class_fuel_only: int = 1) -> torch.Tensor:
    """(batch, n_outputs) mask: 0 on the fuel term of fuel-only-infeasible rows.

    Every other element is 1, so those rows still contribute their nine valid
    outputs to the loss. This is what lets the regressor train on
    valid + partial rows (82.7M) instead of valid only (67.8M) -- and it
    matters most at high sea states, where the fuel-only share rises to 25%.
    """
    m = torch.ones((len(feasibility_cls), n_outputs), dtype=torch.float32,
                   device=feasibility_cls.device)
    m[feasibility_cls == class_fuel_only, fuel_idx] = 0.0
    return m
