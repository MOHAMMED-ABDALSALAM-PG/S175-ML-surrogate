#!/usr/bin/env bash
# The loss-weight grid, end to end: screening split -> 30x5 grid -> selection
# scoring -> selection rule. Sequential (one GPU), resumable, disconnect-proof.
#
# Start it detached so a dropped SSH connection cannot kill it:
#
#   setsid nohup bash scripts/run_alpha_grid.sh > logs/alpha_grid.log 2>&1 &
#
# Every stage skips work that is already on disk, so re-invoking after a crash
# or reboot continues from the first unfinished run.
set -u
PY="${PY:-$HOME/env314/bin/python}"
cd "$(dirname "$0")/.."

# Same single-instance lock as run_all.sh: the two matrices share the GPU and
# must never run concurrently with themselves or each other.
LOCK=/tmp/s175_run_all.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another run matrix already holds $LOCK -- exiting"
  exit 0
fi

echo "=== $(date) alpha grid start ==="

"$PY" scripts/19_screening_split.py || { echo "screening split failed"; exit 1; }
"$PY" scripts/20_alpha_grid.py
grid_rc=$?
"$PY" scripts/21_alpha_grid_eval.py || { echo "selection scoring failed"; exit 1; }
"$PY" scripts/22_alpha_grid_select.py
select_rc=$?

echo "=== $(date) alpha grid end (grid rc=$grid_rc, select rc=$select_rc) ==="
exit $(( grid_rc != 0 ? grid_rc : select_rc ))
