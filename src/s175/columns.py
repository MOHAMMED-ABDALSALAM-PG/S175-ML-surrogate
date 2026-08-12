"""Single source of truth for the S175 dataset schema.

Every other module imports from here. The raw file is positional (the header
line is prose, not machine-parseable), so the ordering below is load-bearing
and is asserted against the file header on every load -- see data.py.
"""

# Raw file column order, verified against the header line of
# S175_shaft_gen_off.txt on 2026-08-09.
INPUT_COLS_ALL = [
    "draft",       # mean draft [m]
    "trim",        # trim [m]
    "GMt",         # transverse metacentric height [m]  (constant 0.55)
    "EngRPM",      # engine speed [1/min]
    "PD",          # propeller P/D ratio [-]
    "shaft_gen",   # shaft gen. power [kW]              (constant 0 in this subset)
    "Hs",          # wave significant height [m]
    "Tp",          # wave peak period [s]
    "Chi",         # wave direction [rad]
    "Vwind",       # wind relative speed [m/s]
    "Theta_wind",  # wind relative direction [rad]
]

OUTPUT_COLS = [
    "speed",        # ship speed [kn]
    "power",        # brake power [kW]
    "torque",       # brake torque [kNm]
    "lat_acc",      # lateral acceleration RMS [g]
    "MSI",          # motion sickness incidence [%]
    "roll",         # rolling amplitude RMS [deg]
    "slam",         # slamming probability [%]
    "green_water",  # green water probability [%]
    "prop_emerg",   # propeller emergence probability [%]
    "fuel",         # fuel consumption [kg/nm]
]

ALL_COLS = INPUT_COLS_ALL + OUTPUT_COLS

# The nine inputs that actually vary once GMt and shaft_gen are held constant.
FEATURE_COLS = [c for c in INPUT_COLS_ALL if c not in ("GMt", "shaft_gen")]

CONSTANT_INPUTS = {"GMt": 0.55, "shaft_gen": 0.0}

FUEL_IDX = OUTPUT_COLS.index("fuel")

UNITS = {
    "speed": "kn", "power": "kW", "torque": "kNm", "lat_acc": "g",
    "MSI": "%", "roll": "deg", "slam": "%", "green_water": "%",
    "prop_emerg": "%", "fuel": "kg/nm",
}

# Physical bounds. None means unbounded on that side.
# All ten outputs are non-negative; the three probabilities are percentages.
# Used to clamp predictions -- the unconstrained network can otherwise emit
# negative probabilities (measured at 86.7% of slamming predictions on the
# original v1 model).
PHYSICAL_BOUNDS = {
    "speed":       (0.0, None),
    "power":       (0.0, None),
    "torque":      (0.0, None),
    "lat_acc":     (0.0, None),
    "MSI":         (0.0, 100.0),
    "roll":        (0.0, None),
    "slam":        (0.0, 100.0),
    "green_water": (0.0, 100.0),
    "prop_emerg":  (0.0, 100.0),
    "fuel":        (0.0, None),
}

# Direction each output should err in, for the safety-aware loss.
# "under" -> prefer under-prediction (alpha > 1)
# "over"  -> prefer over-prediction  (alpha < 1)
SAFE_DIRECTION = {c: ("under" if c == "speed" else "over") for c in OUTPUT_COLS}

# Sentinel the simulator writes for an infeasible operating point.
INFEASIBLE = -1.0

# Feasibility classes.
CLASS_VALID = 0    # all ten outputs defined
CLASS_FUEL_ONLY = 1  # only fuel infeasible
CLASS_ALL_NEG = 2    # every output infeasible

CLASS_NAMES = {
    CLASS_VALID: "fully valid",
    CLASS_FUEL_ONLY: "fuel-only infeasible",
    CLASS_ALL_NEG: "completely infeasible",
}

# Tokens expected in the raw header line, in order. Asserted on load so a
# silently reordered or regenerated file cannot be read with the wrong mapping.
HEADER_TOKENS = [
    "mean draft", "trim", "transverse metacentric height", "engine speed",
    "propeller p/d", "shaft gen", "wave significant height", "wave peak period",
    "wave direction", "wind relative speed", "wind relative direction",
    "ship speed", "brake power", "brake torque", "lateral acceleration",
    "motion sickness", "rolling amplitude", "slamming probability",
    "green water", "propeller emergence", "fuel consumption",
]

N_PREAMBLE_LINES = 4  # blank, title, blank, header -- data starts on line 5
