#!/usr/bin/env bash
# Edit these values, then run: bash run_relocation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=1
SPACING_KM=1.0
GRID_PADDING_KM=20.0
ITERATIONS=20

ARGS=(
    --spacing-km "$SPACING_KM"
    --grid-padding-km "$GRID_PADDING_KM"
    --iterations "$ITERATIONS"
)
if [[ "$NPROC" -eq 1 ]]; then
    python "$SCRIPT_DIR/relocation.py" "${ARGS[@]}"
else
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/relocation.py" "${ARGS[@]}"
fi
