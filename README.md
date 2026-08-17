# S175 ML Surrogate — data-driven surrogate of the S175 ship performance model

Machine-learning surrogate of the S175 WeatherRouting ship-performance
simulation: nine operating and environmental inputs → ten performance and
seakeeping outputs, with explicit modelling of the simulator's two
infeasibility modes and a safety-aware asymmetric loss.

**Paper (in preparation):** *Data-driven machine learning surrogate model for
the S175 ship performance prediction* — Mohammed Abdalsalam¹, Aleksander
Kniat², Przemysław Krata², Joanna Szłapczyńska¹\*
¹ Gdańsk University of Technology, Faculty of Electronics, Telecommunications
and Informatics, Department of Computer Architecture
² Gdańsk University of Technology, Faculty of Mechanical Engineering and Ship
Technology  \* corresponding author

**Physical model:** the training data and all reference outputs come from the
*WeatherRouting* computations of ship speed, fuel consumption & safety indexes
(ver. 0.993) — calculation procedures by Roberto Vettor & Przemysław Krata,
C++ programming by Aleksander Kniat. The simulator itself is not part of this
repository.

## What is in this repository

Code, configurations, **trained models**, per-run records, metrics tables and
figures — everything needed to verify or reuse the reported results.
**The dataset is not included**: the source table is 15.9 GB of simulator
output (126,153,720 operating points on a full factorial grid) and is
regenerated with the simulator; see `data/README.md` for the schema and
`results/evaluation/dataset_splits_table.csv` for the split composition.

```
configs/            experiment definitions (YAML); v1_original.yaml is the
                    uniform-alpha baseline reproduced for the ablation
src/s175/           library: schema, loader, splits, losses, models, metrics
scripts/
  01_prepare.py     cache build + leakage-safe split generation (seeded)
  02_train.py       train one (config, split, seed); timing pilot included
  03_evaluate.py    test-set evaluation of every run (oracle masking, CIs)
  04_report.py      aggregate tables + core figures
  05_paper_figures.py  pred-vs-true, per-input error, response slices, timing
  06_timing_simulator.py  simulator wall-clock measurement
  07_eval_v1_checkpoint.py  original v1 model under the identical protocol
  08_interpolation_test.py  surrogate vs fresh simulator runs at OFF-GRID inputs
runs/               trained checkpoints (~700 KB each) + full provenance:
                    config, seed, split sha256, environment, history, metrics
splits/             split manifests with sha256 (index arrays regenerable)
results/evaluation/ all metric tables (CSV/JSON)
figures/            all paper figures, 300 dpi PNG + vector PDF
tests/              executable assertions, incl. the v1 defect reproduction
```

## The model

Three MLPs sharing one input standardisation (fitted on train only):
a 512-256-128 regressor with a **masked per-output asymmetric loss**
(α=1.5 for ship speed → biased to under-predict; α=0.33 for the other nine →
biased to over-predict, so power/fuel/risk are never under-stated), and two
feasibility classifiers (C1: completely infeasible; C2: fuel-only infeasible)
whose decision thresholds are selected on validation for recall ≥ 0.99, never
an implicit 0.5.

Load a trained run:

```python
import torch
from s175.models import MLP
ck = torch.load("runs/v3_masked_peroutput/S1_random/seed_0/regressor.pt",
                map_location="cpu", weights_only=True)
model = MLP(**ck["model_config"]); model.load_state_dict(ck["model"]).eval()
# ck["x_scaler"] / ck["y_scaler"] hold the standardisation statistics
```

## Headline results (details in results/evaluation/)

- **Accuracy:** R² ≥ 0.999 for all ten outputs on the held-out random test
  split (12.6M rows, 3 seeds); R² ≥ 0.98 under every extrapolation regime
  tested (unseen input levels, held-out operating regions, corner
  extrapolation).
- **Off-grid validation against the physical model:** fresh simulator runs at
  inputs strictly *between* training grid levels — R² ≥ 0.996 (midpoints,
  4,608 points) and ≥ 0.997 (random positions, 54,675 points).
- **Safety bias:** the per-output asymmetric loss puts all 10 outputs on their
  safe error side in-distribution (the uniform-α v1 baseline: 2 of 10 on the
  same test rows) and largely holds off-grid (18/20 cells across the two
  off-grid designs).
- **Honest limits:** at corner extrapolation the completely-infeasible
  classifier misses 11.2% of positives; off-grid random inputs reduce its
  recall from 0.99 to 0.91 — quantified with Wilson CIs in the tables.
- **Speed:** 0.25 µs per point batched on one GPU — ~3.6·10⁷× the single-point
  simulator wall time (measured, `results/evaluation/timing_table.csv`).

## Reproducing

```bash
pip install -r requirements.lock
python scripts/01_prepare.py            # needs the raw simulator table
python scripts/02_train.py --config configs/v3_masked_peroutput.yaml --seed 0
python scripts/03_evaluate.py
python scripts/04_report.py
```

Split manifests carry SHA-256 checksums, every `run.json` records config,
seed, dataset and split hashes, library versions and git commit, and
`tests/test_core.py` contains the executable assertions — including a
reproduction of the v1 uniform-α defect that motivated the corrected loss.

## Data availability

Neither the dataset (15.9 GB of simulator output; 126,153,720 operating
points) nor the S175 WeatherRouting simulator that generates it is
distributed in this repository. The dataset is available **upon request to
the corresponding author, Prof. Joanna Szłapczyńska** (Gdańsk University of
Technology).

## Acknowledgements

The authors gratefully acknowledge Prof. Roberto Vettor for developing the
theoretical general ship model that underlies the S175 WeatherRouting
performance simulator used in this study. His foundational contribution to
the formulation of the ship-performance model is sincerely appreciated.

## License

To be decided by the authors before the repository is made public.
