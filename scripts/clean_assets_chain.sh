#!/usr/bin/env bash
# One-shot chain for the clean-evaluation assets. Runs GPU steps sequentially
# to avoid contention: waits for any ablation clean evals and
# abl_unmasked_fixedbn training already in flight, then regenerates every
# remaining derived asset on the screening-disjoint rows.
set -u
PY=$HOME/env314/bin/python
cd "$(dirname "$0")/.."

echo "chain started $(date '+%F %T')"

# 1. wait for the in-flight ablation clean evals (three configs)
while pgrep -f "03_evaluate.py --config-name (v1_original|abl_uniform_masked|abl_symmetric_masked)" > /dev/null; do
  sleep 30
done
echo "step1 done: existing ablation arms clean-evaluated $(date '+%F %T')"

# 2. wait for the corrected unmasked arm to finish training
while ! grep -q "wrote regressor.pt" logs/abl_unmasked_fixedbn.log; do
  if grep -qE "Traceback|CUDA out of memory" logs/abl_unmasked_fixedbn.log; then
    echo "FATAL: abl_unmasked_fixedbn training failed"; exit 1
  fi
  sleep 60
done
echo "step2 done: abl_unmasked_fixedbn trained $(date '+%F %T')"

# 3. evaluate the new arm, full and clean
$PY scripts/03_evaluate.py --config-name abl_unmasked_fixedbn --chunk 1000000 || exit 1
$PY scripts/03_evaluate.py --config-name abl_unmasked_fixedbn --clean --chunk 1000000 || exit 1
echo "step3 done: new arm evaluated $(date '+%F %T')"

# 4. five-arm loss-ablation table + figure on clean rows
$PY scripts/12_loss_ablation.py --clean || exit 1
echo "step4 done: loss ablation regenerated $(date '+%F %T')"

# 5. paper figures on clean rows
$PY scripts/05_paper_figures.py --clean || exit 1
echo "step5 done: paper figures $(date '+%F %T')"
$PY scripts/10_legacy_style_figures.py --clean --only pipeline regression confusion residuals seaheat || exit 1
echo "step6 done: legacy-style figures $(date '+%F %T')"

echo "chain finished OK $(date '+%F %T')"
