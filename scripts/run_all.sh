#!/usr/bin/env bash
# The recorded run matrix: sequential (one GPU), resumable, disconnect-proof.
#
# Start it detached so a dropped SSH connection cannot kill it:
#
#   setsid nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
#
# A run whose run.json already exists is skipped, so re-invoking after a crash
# or reboot continues from the first unfinished run. A failed run is reported
# and does NOT stop the matrix -- the remaining runs are independent of it.
set -u
PY="${PY:-$HOME/env314/bin/python}"
cd "$(dirname "$0")/.."

# Single-instance lock: a second invocation (an accidental double launch, or
# the optional cron watchdog firing while the matrix is mid-run) exits
# immediately instead of contending for the GPU. The lock dies with the
# process, so a crash or reboot never leaves it stuck.
LOCK=/tmp/s175_run_all.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another run_all.sh already holds $LOCK -- exiting"
  exit 0
fi
CONFIG=configs/v3_masked_peroutput.yaml
NAME=v3_masked_peroutput

# Order: the S1 seeds first (the headline comparability result and its
# variance), then one seed per generalisation regime.
RUNS=(
  "S1_random 0"
  "S1_random 1"
  "S1_random 2"
  "S2_level_Hs 0"
  "S2_level_Chi 0"
  "S2_level_draft 0"
  "S2_level_Vwind 0"
  "S3_block 0"
  "S4_corner 0"
)

echo "run matrix: ${#RUNS[@]} runs | host $(hostname) | started $(date '+%F %T')"
failures=0
for r in "${RUNS[@]}"; do
  read -r split seed <<< "$r"
  out="runs/$NAME/$split/seed_$seed"
  if [ -f "$out/run.json" ]; then
    echo "[$(date '+%T')] skip  $split seed $seed -- $out/run.json exists"
    continue
  fi
  log="logs/03_full_${split}_seed${seed}.log"
  echo "[$(date '+%T')] start $split seed $seed -> $log"
  "$PY" scripts/02_train.py --config "$CONFIG" --split "$split" --seed "$seed" \
      > "$log" 2>&1
  st=$?
  if [ $st -eq 0 ]; then
    echo "[$(date '+%T')] done  $split seed $seed"
  else
    failures=$((failures + 1))
    echo "[$(date '+%T')] FAIL  $split seed $seed (exit $st) -- see $log; continuing"
  fi
done
echo "matrix finished $(date '+%F %T') with $failures failure(s)"
exit $((failures > 0))
