"""Reader-facing names for outputs and holdout regimes, shared by every figure.

Keys are the internal identifiers of columns.py and splits.py; values are the
names printed on axes, panel titles and legends. Matplotlib mathtext is used
for the input symbols.
"""
from __future__ import annotations

from s175.columns import UNITS

OUTPUT_LABELS = {
    "speed": "Ship speed",
    "power": "Brake power",
    "torque": "Brake torque",
    "lat_acc": "Lateral acc. RMS",
    "MSI": "Motion sickness incidence",
    "roll": "Roll RMS",
    "slam": "Slamming prob.",
    "green_water": "Green water prob.",
    "prop_emerg": "Prop. emergence prob.",
    "fuel": "Fuel consumption",
}

REGIME_LABELS = {
    "S1_random": "Random split",
    "S2_level_Hs": "Unseen $H_s$ levels",
    "S2_level_Chi": r"Unseen $\chi$ levels",
    "S2_level_draft": "Unseen draft level",
    "S2_level_Vwind": "Unseen $V_{wind}$ levels",
    "S3_block": "Block holdout",
    "S4_corner": "Corner extrapolation",
}


def output_label(c: str, units: bool = False) -> str:
    """Display name of output `c`, optionally followed by its unit."""
    return f"{OUTPUT_LABELS[c]} [{UNITS[c]}]" if units else OUTPUT_LABELS[c]


def regime_label(split: str) -> str:
    return REGIME_LABELS[split]
