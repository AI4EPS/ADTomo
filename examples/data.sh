#!/usr/bin/env bash
# Edit these values to control the noisy starting catalog, then run: bash data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HORIZONTAL_NOISE_KM=2.0
DEPTH_NOISE_KM=2.0
TIME_NOISE_S=0.5

python "$SCRIPT_DIR/00_gen_velocity.py"
python "$SCRIPT_DIR/01_gen_stations.py"
python "$SCRIPT_DIR/02_gen_events.py" \
    --horizontal-noise-km "$HORIZONTAL_NOISE_KM" \
    --depth-noise-km "$DEPTH_NOISE_KM" \
    --time-noise-s "$TIME_NOISE_S"
python "$SCRIPT_DIR/03_gen_picks.py"
