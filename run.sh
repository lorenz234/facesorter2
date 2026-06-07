#!/usr/bin/env bash
# One command, start to finish: set up the environment (first run only), then
# detect → cluster → export one folder per person → build the visual keys.
#
# Usage:
#   ./run.sh                       # uses all_pictures/ with sensible defaults
#   DATA=my_photos ./run.sh        # point at a different folder
#   MIN_CLUSTER_SIZE=5 ./run.sh    # override any setting (see below)
set -euo pipefail
cd "$(dirname "$0")"

# ---- settings (override via environment variables) ----
DATA="${DATA:-all_pictures}"          # input photo folder
OUT="${OUT:-out}"                     # output folder
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-8}"   # smallest group counted as a person
MIN_DET_SCORE="${MIN_DET_SCORE:-0.7}"       # drop weak detections (backs of heads)
MIN_SIZE="${MIN_SIZE:-3}"             # skip people seen in fewer than N photos

# ---- 1. environment (only the first time) ----
if [ ! -d .venv ]; then
  PY=""
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  [ -z "$PY" ] && { echo "No suitable python found (need 3.10-3.12)." >&2; exit 1; }
  echo ">> Creating virtualenv with $("$PY" --version)"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
PY=.venv/bin/python

if [ ! -d "$DATA" ]; then
  echo "Input folder '$DATA' not found. Put your photos there, or set DATA=..." >&2
  exit 1
fi

# ---- 2. cluster  →  3. export  →  4. visualize ----
echo ">> [1/3] Detecting + clustering faces in '$DATA'"
"$PY" cluster.py --data "$DATA" --out "$OUT" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" --min-det-score "$MIN_DET_SCORE" --resume

echo ">> [2/3] Exporting one folder per person"
"$PY" export_clusters.py --data "$DATA" --clusters "$OUT/clusters.csv" \
  --out "$OUT" --min-size "$MIN_SIZE" --clear

echo ">> [3/3] Building the person key + annotated group photos"
"$PY" visualize.py key --data "$DATA" --out "$OUT"
"$PY" visualize.py annotate --data "$DATA" --out "$OUT" --top 3

echo
echo "Done. Open:"
echo "  $OUT/clusters/            one folder per person"
echo "  $OUT/people_overview.jpg  labeled face key (who is who)"
echo "  $OUT/annotated/           group photos with labeled boxes"
