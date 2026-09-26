# ADTomo

A differentiable travel-time tomography framework based on the eikonal equation.

## Install

```bash
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

## Package

```text
adtomo/grid.py          VelocityModel (3-D), VelocityModel1D (+ from_3d), ForwardGrid, ForwardGrid2D, coordinate transforms
adtomo/tomography3d.py  3-D eikonal autograd op, predict_travel_times, smoothness, Tomography
adtomo/tomography2d.py  2-D eikonal autograd op, predict_travel_times_2d, smoothness_1d, Tomography2D
adtomo/data.py          build_station_groups: stations + events + picks -> station/phase groups
adtomo/optimize.py      init_distributed, set_trainable, optimize (L-BFGS / Adam, serial or torchrun)
```

A 1-D model is inverted on station-centered Cartesian vertical sections
(`ForwardGrid2D`): each node samples `V(depth)` at its true spherical depth, so
layers stay curved, and events map to `(sqrt(E^2 + N^2), D)` of the same
East/North/Down frame the 3-D `ForwardGrid` uses. Event locations and
origin-time corrections are trainable in both modes and are re-mapped live on
every forward pass.

## Synthetic example

```bash
bash examples/run_pipeline.sh                                        # 3-D checkerboard dataset, then invert vp,vs in 3-D
MODEL=1d bash examples/run_inversion.sh                              # 1-D inversion of the same dataset
MODEL=1d TRAINABLE=event_loc,event_time bash examples/run_inversion.sh
MODEL=3d TRAINABLE=vp,vs,event_loc,event_time NPROC=4 bash examples/run_inversion.sh
```

`run_pipeline.sh` runs `00_gen_velocity.py` (3-D initial/true checkerboard),
`01_gen_stations.py`, `02_gen_events.py` (true `events.csv` plus, with
`LOCATION_NOISE_KM`/`TIME_NOISE_S`, a noisy `events_initial.csv` the inversion
starts from), `03_gen_picks.py` (3-D forward modelling), then `run_inversion.sh`.
Both `MODEL=1d` and `MODEL=3d` read exactly the same files; the 1-D mode starts
from the horizontal mean of the 3-D initial model (`VelocityModel1D.from_3d`).

Inversion knobs: `TRAINABLE` (subset of `vp,vs,event_loc,event_time`, toggled
through `requires_grad`), `OPTIMIZER` (`lbfgs` or `adam`), `ITERATIONS`,
`LEARNING_RATE`, `SPACING`, `GRID_PADDING`, `BETA_VP/VS` (smoothness),
`ALPHA_VP/VS` (damping), `NPROC`; extra arguments go to `inversion.py`. Results
land in `examples/results/` (`model_inverted.pt`, `events_inverted.csv`) and
figures in `examples/figures/` (`checkerboard.png`, `geometry.png`,
`inversion.png`, `events.png`).
