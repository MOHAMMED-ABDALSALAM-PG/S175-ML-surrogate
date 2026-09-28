# S175 ML Surrogate — data-driven surrogate of the S175 ship performance model

Machine-learning surrogate of the S175 WeatherRouting ship-performance
simulation: nine operating and environmental inputs → ten performance and
seakeeping outputs, with explicit modelling of the simulator's two
infeasibility modes and a safety-aware asymmetric loss.

**Paper (manuscript sources in [`paper/`](paper/)):** *Data-driven machine learning surrogate model for
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
  00_screening_subsample.py  draws the 5% screening subsample (6,306,419 rows):
                    5% of every Hs-band x wave-direction x wind-speed x
                    feasibility stratum, random_state 42 (copied unchanged
                    from the original study; needs the full dataset, which
                    is not distributed)
  01_prepare.py     cache build + leakage-safe split generation (seeded)
  02_train.py       train one (config, split, seed); timing pilot included
  03_evaluate.py    test-set evaluation of every run (oracle masking, CIs)
  04_report.py      aggregate tables + core figures
  05_paper_figures.py  pred-vs-true, per-input error, response slices, timing
  06_timing_simulator.py  simulator wall-clock measurement
  08_interpolation_test.py  surrogate vs fresh simulator runs at OFF-GRID inputs
  17a_architecture_search.py  architecture search behind the network sizes:
                    regressor and classifier configurations compared on one
                    80/10/10 split of the screening subsample (copied
                    unchanged from the original study; results in
                    results/evaluation/architecture_search.json)
  17_algorithm_screening.py  original model-family screening, step 1: single
                      speed regressor with -1 infeasibility target (copied
                      unchanged from the original study; writes to its paths)
  17b_two_stage_screening.py original model-family screening, step 2: RF/XGB/MLP
                      infeasibility classifiers and valid-only speed regressors
                      (copied unchanged; results in two_stage_screening.json --
                      note: in its combined_results block the key
                      "false_negatives_dangerous" holds the false positives;
                      the correct missed-infeasible counts are the
                      classifier_results "FN_dangerous" values)
  17c_screening_comparison.py  model-family comparison behind Table 4: single
                      -1-target regressor vs classifier + valid-row regressor
                      for RF/XGB/MLP on one shared partition, 3 seeds
                      (run_screening_comparison.sh runs all 18 jobs; results in
                      results/evaluation/screening_comparison/)
  verify_tables.py / verify_manuscript.py  check paper/main.tex against results
  run_alpha_grid.sh   joint loss-weight grid: screening split → 30 pairs ×
                      5 seeds → prespecified selection rule → grid table and
                      figure (scripts 19–24, in order)
runs/               trained checkpoints (~700 KB each) + full provenance:
                    config, seed, split sha256, environment, history, metrics
splits/             split manifests with sha256 (index arrays regenerable)
results/evaluation/ all metric tables (CSV/JSON)
figures/            all paper figures, 300 dpi PNG + vector PDF
paper/              manuscript sources (main.tex, sample.bib, included figures)
tests/              executable assertions, incl. the asymmetric-loss direction contrast
```

## The model

Three MLPs sharing one input standardisation (fitted on train only):
a 512-256-128 regressor with a **masked per-output asymmetric loss**
(α=1.5 for ship speed → biased to under-predict; α=0.33 for the other nine →
biased to over-predict, so propulsion, fuel and seakeeping outputs err
preferentially on their conservative side -- a distributional tendency, not a
per-prediction guarantee), and two
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

- **Accuracy:** R² ≥ 0.9994 for all ten outputs on the screening-disjoint
  random test split (11,984,559 rows, seed 0; three-seed spread in
  `s1_seed_spread.json`). Under the structured holdouts the minimum output R²
  ranges from 0.921 (unseen draft level, slamming probability) to 0.9993
  (unseen wave-direction levels).
- **Off-grid validation against the physical model:** fresh simulator runs at
  inputs strictly *between* training grid levels — R² ≥ 0.9959 (midpoint
  design, 4,608 combinations) and ≥ 0.9967 (irregular design, 54,675
  combinations).
- **Conservative error direction:** with the per-output asymmetric loss all 10
  outputs err preferentially on their conservative side for seeds 0 and 1 and
  9 of 10 for seed 2 (the uniform-α baseline: 3 of 10 on the same test rows);
  off-grid, 10 of 10 at the midpoints and 8 of 10 at the irregular inputs.
- **Speed (same workstation, simulator on CPU, surrogate on GPU):** single
  point 2.65 ms vs 8.94 s (×3,378); identical 864-point batch 2.74 ms vs
  14.34 s (×5,226); large-batch throughput about 4.03·10⁶ predictions/s
  (0.25 µs per point, amortised) — `results/simulator_timing.json`,
  `results/evaluation/same_inputs_timing.json`, `figures/inference_speed.json`.

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
demonstration of how a uniform-α weighting differs from the selected
per-output loss.

## Data availability

The dataset used in this project consists of 126,153,720 evaluations of the
S175 WeatherRouting ship-performance simulator (ver. 0.993) over the full
factorial design described in the paper (15.9 GB of simulator output, shaft
generator off). It was generated specifically for this project and is **not
included in this repository**, and neither is the simulator itself.

The dataset is available on reasonable request from the corresponding author:
**Prof. Joanna Szłapczyńska**, Gdańsk University of Technology, Faculty of
Electronics, Telecommunications and Informatics
(joanna.szlapczynska@pg.edu.pl).

What *is* included: the split definitions with SHA-256 checksums (the index
arrays are regenerated from the recorded seeds), the trained checkpoints of
every reported run with their provenance records, all evaluation tables, and
the input values of the two off-grid validation designs
(`results/evaluation/offgrid_design_levels.json`).

## Acknowledgements

The authors gratefully acknowledge **Prof. Roberto Vettor** for developing the
theoretical general ship model that underlies the S175 WeatherRouting
performance simulator used in this study. His foundational contribution to
the formulation of the ship-performance model is sincerely appreciated.

## License

To be decided by the authors before the repository is made public.
