#!/usr/bin/env bash
# Edit these values to control the noisy starting catalog, then run: bash data.sh
set -euo pipefail

python 00_gen_velocity.py
python 01_gen_stations.py
python 02_gen_events.py
python 03_gen_picks.py
