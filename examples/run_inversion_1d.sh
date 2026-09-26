#!/usr/bin/env bash
# Edit these values, then run: bash run_inversion_1d.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=1
SPACING_KM=1.0
GRID_PADDING_KM=20.0
ITERATIONS=20
VP_INITIAL_KM_S=5.5
VS_INITIAL_KM_S=3.2
ALPHA_VP=0.0
ALPHA_VS=0.0
BETA_VP=0.0
BETA_VS=0.0

ARGS=(
    --spacing-km "$SPACING_KM"
    --grid-padding-km "$GRID_PADDING_KM"
    --iterations "$ITERATIONS"
    --vp-initial-km-s "$VP_INITIAL_KM_S"
    --vs-initial-km-s "$VS_INITIAL_KM_S"
    --alpha-vp "$ALPHA_VP"
    --alpha-vs "$ALPHA_VS"
    --beta-vp "$BETA_VP"
    --beta-vs "$BETA_VS"
)
if [[ "$NPROC" -eq 1 ]]; then
    python "$SCRIPT_DIR/inversion_1d.py" "${ARGS[@]}"
else
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/inversion_1d.py" "${ARGS[@]}"
fi
