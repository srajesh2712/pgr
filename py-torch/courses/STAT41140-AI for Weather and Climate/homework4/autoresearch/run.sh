#!/usr/bin/env bash
# One time-boxed experiment: ./run.sh "short description of the idea"
# Trains train.py for TIME_BUDGET minutes (default 5) on the GPU, appends a row to
# results.tsv and prints KEEP (beats best so far) or DISCARD.
set -uo pipefail
cd "$(dirname "$0")"

# Optional flags (flags, not env vars, so the command always starts with ./run.sh):
#   --seed N      random seed (default 0)      --status S   force status label (e.g. baseline, noise)
#   --budget M    minutes (default 5)
SEED=0; STATUS_OVERRIDE=""; BUDGET="${TIME_BUDGET:-5}"
while [[ $# -gt 0 && "$1" == --* ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --status) STATUS_OVERRIDE="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        *) echo "unknown flag $1"; exit 2 ;;
    esac
done
DESC="${1:-no description}"
# A change is KEEP only if it beats the best by more than run-to-run noise (seed spread).
MIN_DELTA=0
DEFAULT_PY=/home/james/project/AI_weather/.venv/bin/python
[ -x "$DEFAULT_PY" ] || DEFAULT_PY=python
PY="${PYTHON:-$DEFAULT_PY}"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p logs
LOG="logs/$(date +%Y%m%d_%H%M%S)_${COMMIT}.log"
HARD_LIMIT=$(awk "BEGIN{print int($BUDGET*60 + 240)}")    # budget + data load/eval slack

[ -f results.tsv ] || printf "commit\tval_score\trmse_temp\trmse_rhum\trmse_msl\trmse_wdsp\tparams\tstatus\tdescription\n" > results.tsv

timeout "${HARD_LIMIT}s" env TIME_BUDGET="$BUDGET" SEED="$SEED" MPLCONFIGDIR="${TMPDIR:-/tmp}" "$PY" train.py > "$LOG" 2>&1
rc=$?

get() { grep "^$1:" "$LOG" | tail -1 | awk '{print $2}'; }
SCORE=$(get val_score)
if [ $rc -ne 0 ] || [ -z "$SCORE" ]; then
    STATUS="crash"; SCORE="nan"
    echo "CRASH (exit $rc). Last lines of $LOG:"; tail -20 "$LOG"
else
    BEST=$(awk -F'\t' 'NR>1 && ($8=="keep" || $8=="baseline") && $2!="nan" {print $2}' results.tsv | sort -g | head -1)
    if [ -z "$BEST" ] || awk "BEGIN{exit !($SCORE < $BEST - $MIN_DELTA)}"; then STATUS="keep"; else STATUS="discard"; fi
    echo "$(echo $STATUS | tr a-z A-Z): val_score=$SCORE (best before: ${BEST:-none}, min_delta $MIN_DELTA)  log=$LOG"
fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$COMMIT" "$SCORE" "$(get rmse_temp)" "$(get rmse_rhum)" \
    "$(get rmse_msl)" "$(get rmse_wdsp)" "$(get params)" "${STATUS_OVERRIDE:-$STATUS}" "$DESC" >> results.tsv
