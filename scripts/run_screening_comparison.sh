#!/usr/bin/env bash
# Runs all 18 jobs of scripts/17c_screening_comparison.py (2 approaches x 3 families x 3 seeds)
# in parallel on one host, then aggregates. Thread budget per job keeps the total below the
# host's core count. XGBoost is taken from $XGB_PATH if it is not installed in the interpreter.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/env314/bin/python}
export PYTHONPATH=${XGB_PATH:-$HOME/pylibs_xgb}${PYTHONPATH:+:$PYTHONPATH}
THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
LOG=logs/screening_comparison; mkdir -p "$LOG"
for seed in 0 1 2; do
  for fam in MLP RF XGB; do
    for app in A B; do
      out=results/evaluation/screening_comparison/${app}_${fam}_seed${seed}.json
      [ -e "$out" ] && continue
      "$PY" scripts/17c_screening_comparison.py --approach $app --family $fam --seed $seed \
        --threads "$THREADS" > "$LOG/${app}_${fam}_seed${seed}.log" 2>&1 &
    done
  done
done
wait
"$PY" scripts/17c_screening_comparison.py --aggregate > "$LOG/aggregate.log" 2>&1
echo "finished $(date)"
