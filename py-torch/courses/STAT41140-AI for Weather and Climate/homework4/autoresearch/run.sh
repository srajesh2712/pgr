#!/usr/bin/env bash
# One time-boxed experiment: ./run.sh "short description of the idea"
# Trains train.py for TIME_BUDGET minutes (default 5) on the GPU, appends a row to
# results.tsv and prints KEEP (beats best so far) or DISCARD.
set -uo pipefail
cd "$(dirname "$0")"

DESC="${1:-no description}"
BUDGET="${TIME_BUDGET:-5}"
PY="${PYTHON:-python}"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p logs
LOG="logs/$(date +%Y%m%d_%H%M%S)_${COMMIT}.log"
HARD_LIMIT=$(awk "BEGIN{print int($BUDGET*60 + 240)}")    # budget + data load/eval slack

[ -f results.tsv ] || printf "commit\tval_score\trmse_temp\trmse_rhum\trmse_msl\trmse_wdsp\tparams\tstatus\tdescription\n" > results.tsv

timeout "${HARD_LIMIT}s" env TIME_BUDGET="$BUDGET" MPLCONFIGDIR="${TMPDIR:-/tmp}" "$PY" train.py > "$LOG" 2>&1
rc=$?

get() { grep "^$1:" "$LOG" | tail -1 | awk '{print $2}'; }
SCORE=$(get val_score)
if [ $rc -ne 0 ] || [ -z "$SCORE" ]; then
    STATUS="crash"; SCORE="nan"
    echo "CRASH (exit $rc). Last lines of $LOG:"; tail -20 "$LOG"
else
    BEST=$(awk -F'\t' 'NR>1 && ($8=="keep" || $8=="baseline") && $2!="nan" {print $2}' results.tsv | sort -g | head -1)
    if [ -z "$BEST" ] || awk "BEGIN{exit !($SCORE < $BEST)}"; then STATUS="keep"; else STATUS="discard"; fi
    echo "$(echo $STATUS | tr a-z A-Z): val_score=$SCORE (best before: ${BEST:-none})  log=$LOG"
fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$COMMIT" "$SCORE" "$(get rmse_temp)" "$(get rmse_rhum)" \
    "$(get rmse_msl)" "$(get rmse_wdsp)" "$(get params)" "${STATUS_OVERRIDE:-$STATUS}" "$DESC" >> results.tsv
