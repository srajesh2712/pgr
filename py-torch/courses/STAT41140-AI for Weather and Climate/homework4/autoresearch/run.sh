#!/usr/bin/env bash
# One time-boxed experiment: ./run.sh "short description of the idea"
# Trains train.py for TIME_BUDGET minutes (default 5) on the GPU, appends a row to
# results.tsv and prints KEEP (beats best so far) or DISCARD.
set -uo pipefail
cd "$(dirname "$0")"

# Optional flags (flags, not env vars, so the command always starts with ./run.sh):
#   --seed N      random seed (default 0)      --status S   force status label (e.g. baseline, noise)
#   --budget M    minutes (default 5)        --horizon H  forecast hours (default 6; H!=6 -> results_hH.tsv)
SEED=0; STATUS_OVERRIDE=""; BUDGET="${TIME_BUDGET:-5}"; HORIZON=6
while [[ $# -gt 0 && "$1" == --* ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --status) STATUS_OVERRIDE="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --horizon) HORIZON="$2"; shift 2 ;;
        *) echo "unknown flag $1"; exit 2 ;;
    esac
done
DESC="${1:-no description}"
RESULTS=results.tsv; [ "$HORIZON" = "6" ] || RESULTS="results_h${HORIZON}.tsv"   # one table per horizon
# A change is KEEP only if it beats the best by more than run-to-run noise (seed spread).
MIN_DELTA=0.002   # attn-arch seed std ~0.0005 (6h seeds 0.4544/0.4553/0.4548); was 0.005 for the noisier teammate arch
DEFAULT_PY=/home/james/project/AI_weather/.venv/bin/python
[ -x "$DEFAULT_PY" ] || DEFAULT_PY=python
PY="${PYTHON:-$DEFAULT_PY}"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p logs
LOG="logs/$(date +%Y%m%d_%H%M%S)_${COMMIT}.log"
HARD_LIMIT=$(awk "BEGIN{print int($BUDGET*60 + 240)}")    # budget + data load/eval slack

[ -f "$RESULTS" ] || printf "commit\tval_score\trmse_temp\trmse_rhum\trmse_msl\trmse_wdsp\tparams\tstatus\tdescription\n" > "$RESULTS"

timeout "${HARD_LIMIT}s" env TIME_BUDGET="$BUDGET" SEED="$SEED" HORIZON="$HORIZON" MPLCONFIGDIR="${TMPDIR:-/tmp}" "$PY" train.py > "$LOG" 2>&1
rc=$?

get() { grep "^$1:" "$LOG" | tail -1 | awk '{print $2}'; }
SCORE=$(get val_score)
if ! grep -q "^device=cuda" "$LOG"; then
    STATUS="crash"; SCORE="nan"
    echo "CRASH: ran on CPU, not GPU. Run ./run.sh as a standalone command (not chained) so the sandbox exemption applies."
elif [ $rc -ne 0 ] || [ -z "$SCORE" ]; then
    STATUS="crash"; SCORE="nan"
    echo "CRASH (exit $rc). Last lines of $LOG:"; tail -20 "$LOG"
else
    BEST=$(awk -F'\t' 'NR>1 && ($8=="keep" || $8=="baseline") && $2!="nan" {print $2}' "$RESULTS" | sort -g | head -1)
    if [ -z "$BEST" ] || awk "BEGIN{exit !($SCORE < $BEST - $MIN_DELTA)}"; then STATUS="keep"; else STATUS="discard"; fi
    echo "$(echo $STATUS | tr a-z A-Z): val_score=$SCORE (best before: ${BEST:-none}, min_delta $MIN_DELTA)  log=$LOG"
fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$COMMIT" "$SCORE" "$(get rmse_temp)" "$(get rmse_rhum)" \
    "$(get rmse_msl)" "$(get rmse_wdsp)" "$(get params)" "${STATUS_OVERRIDE:-$STATUS}" "$DESC" >> "$RESULTS"
