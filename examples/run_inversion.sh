#!/usr/bin/env bash
# Invert the dataset in data/ with a 1-D (MODEL=1d) or 3-D (MODEL=3d) model, serially or under torchrun.
#
#   MODEL=3d bash run_inversion.sh                                        # invert vp,vs
#   MODEL=1d TRAINABLE=event_loc,event_time bash run_inversion.sh         # relocation only
#   MODEL=1d TRAINABLE=vp,vs,event_loc,event_time NPROC=4 bash run_inversion.sh
#
# Extra command-line arguments are passed to inversion.py (e.g. --grid-padding 15).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-3d}"                       # 3d | 1d
NPROC="${NPROC:-1}"
TRAINABLE="${TRAINABLE:-vp,vs}"            # subset of vp,vs,event_loc,event_time
OPTIMIZER="${OPTIMIZER:-lbfgs}"            # lbfgs | adam
ITERATIONS="${ITERATIONS:-30}"
BETA_VP="${BETA_VP:-0.0}"
BETA_VS="${BETA_VS:-0.0}"
ALPHA_VP="${ALPHA_VP:-0.0}"
ALPHA_VS="${ALPHA_VS:-0.0}"

ARGS=(
    --model "$MODEL"
    --trainable "$TRAINABLE"
    --optimizer "$OPTIMIZER"
    --iterations "$ITERATIONS"
    --beta-vp "$BETA_VP"
    --beta-vs "$BETA_VS"
    --alpha-vp "$ALPHA_VP"
    --alpha-vs "$ALPHA_VS"
)
[[ -n "${LEARNING_RATE:-}" ]] && ARGS+=(--learning-rate "$LEARNING_RATE")
[[ -n "${SPACING:-}" ]] && ARGS+=(--spacing "$SPACING")
[[ -n "${GRID_PADDING:-}" ]] && ARGS+=(--grid-padding "$GRID_PADDING")

# Extra command-line arguments come last, so argparse treats them as overrides.
if [[ "$NPROC" -eq 1 ]]; then
    python "$SCRIPT_DIR/inversion.py" "${ARGS[@]}" "$@"
else
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/inversion.py" "${ARGS[@]}" "$@"
fi
