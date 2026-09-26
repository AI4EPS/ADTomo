#!/usr/bin/env bash
# Edit these values, then run: bash run_inversion.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC=1
SPACING_1D=1.0
SPACING_3D=2.0
GRID_PADDING=20.0
ITERATIONS_1D=20
ITERATIONS_RELOCATION=20
ITERATIONS_3D=30
ALPHA_VP_1D=0.0
ALPHA_VS_1D=0.0
BETA_VP_1D=0.0
BETA_VS_1D=0.0
ALPHA_VP_3D=0.0
ALPHA_VS_3D=0.0
BETA_VP_3D=0.0
BETA_VS_3D=0.0

ARGS=(
    --spacing-1d "$SPACING_1D"
    --spacing-3d "$SPACING_3D"
    --grid-padding "$GRID_PADDING"
    --iterations-1d "$ITERATIONS_1D"
    --iterations-relocation "$ITERATIONS_RELOCATION"
    --iterations-3d "$ITERATIONS_3D"
    --alpha-vp-1d "$ALPHA_VP_1D"
    --alpha-vs-1d "$ALPHA_VS_1D"
    --beta-vp-1d "$BETA_VP_1D"
    --beta-vs-1d "$BETA_VS_1D"
    --alpha-vp-3d "$ALPHA_VP_3D"
    --alpha-vs-3d "$ALPHA_VS_3D"
    --beta-vp-3d "$BETA_VP_3D"
    --beta-vs-3d "$BETA_VS_3D"
)

if [[ "$NPROC" -eq 1 ]]; then
    python "$SCRIPT_DIR/inversion.py" "${ARGS[@]}"
else
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/inversion.py" "${ARGS[@]}"
fi
