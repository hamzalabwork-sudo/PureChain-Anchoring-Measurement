#!/usr/bin/env bash
# B8 rounds: repeat the OpenTimestamps baseline every GAP seconds so that latency covers different calendar commit
# times and Bitcoin blocks. Each round stamps, then polls its proofs in the background until they are attested.
# usage: experiments/ots_rounds.sh ROUNDS GAP_S RECORDS BATCHES
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for i in $(seq 1 "$1"); do
  run=$($PY -m experiments.ots_baseline stamp --records "$3" --batches "$4" --seed $((100 + i)) | tail -1)
  echo "$(date '+%F %T') round $i: $run"
  $PY -m experiments.ots_baseline upgrade "$run" --every 600 --hours 30 > "$run/upgrade.log" 2>&1 &
  [ "$i" -lt "$1" ] && sleep "$2"
done
wait
echo "$(date '+%F %T') all rounds attested or timed out"
