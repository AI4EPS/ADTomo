#!/usr/bin/env bash
set -e

# TRAINABLE names: vp, vs, event_loc, event_time.
INPUT_DIR="data"
OUTPUT_DIR="results/relocation"
TRAINABLE="event_loc,event_time"

# INPUT_DIR="results/relocation"
# OUTPUT_DIR="results/tomography"
# TRAINABLE="vp,vs,event_loc,event_time"

# Use NPROC=1 for a simple local run; increase only when using torchrun.
NPROC=32
ITERATIONS=200
# Separate Adam learning rates for vp, vs, horizontal location, vertical
# location, and origin time.
LR_VP=0.01
LR_VS=0.01
LR_LOC_HORI=0.005
LR_LOC_VERT=0.05
LR_ORIGIN_TIME=0.01
# Velocity regularization (zero means no regularization).
ALPHA_VP=0.0
ALPHA_VS=0.0
BETA_VP=0.0
BETA_VS=0.0
# Forward-grid spacing and numerical padding, in km.
SPACING_KM=2.0
GRID_PADDING_KM=4.0

export TRAINABLE LR_VP LR_VS LR_LOC_HORI LR_LOC_VERT LR_ORIGIN_TIME
export SPACING_KM GRID_PADDING_KM ITERATIONS
export ALPHA_VP ALPHA_VS BETA_VP BETA_VS
export INPUT_DIR OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"
if [ "$NPROC" -eq 1 ]; then
    python inversion.py
else
    torchrun --standalone --nproc_per_node="$NPROC" inversion.py
fi
