# ADTomo

Differentiable spherical travel-time tomography.

## Create the environment

```bash
conda create -n adtomo python=3.11 -y
conda activate adtomo
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

The build creates the two extensions used by the examples and tests:
`eikonal2d_op` and `eikonal3d_op`.

## Generate the common synthetic data

The numbered scripts create the same input format consumed by all three
independent examples. The inversion inputs are `model_initial.pt`,
`stations.csv`, `events_initial.csv`, and `picks.csv`. The true model and true
catalog are used only for synthetic comparison figures.

```bash
cd examples
python 00_gen_velocity.py
python 01_gen_stations.py
python 02_gen_events.py \
    --horizontal-noise-km 2.0 \
    --depth-noise-km 2.0 \
    --time-noise-s 0.5
python 03_gen_picks.py --spacing-km 2.0
```

The generated tables use `station_id`, `longitude`, `latitude`, `depth_km` for
stations and `event_id`, `event_time`, `longitude`, `latitude`, `depth_km` for
events. Picks use `event_id`, `station_id`, `phase_type`, and `phase_time`.

## Run independent examples

Each script reads the same four inversion inputs and can be run independently:

```bash
bash run_inversion_1d.sh
bash run_relocation.sh
bash run_inversion_3d.sh
```

`run_inversion_1d.sh` inverts 1-D Vp/Vs, `run_relocation.sh` relocates events
with a fixed 1-D model, and `run_inversion_3d.sh` jointly inverts 3-D Vp/Vs,
event locations, and origin times. Edit each shell file's `SPACING_KM` and
`GRID_PADDING_KM` values (km), iteration count, and local alpha/beta
regularization settings before running.

## Run the gradient validation

Run the retained Eikonal backward check from the `tests` directory:

```bash
cd ../tests
python test_gradient.py
```

The script saves `tests/figures/gradient_test.png` and displays the two
Taylor-convergence panels when an interactive Matplotlib backend is available.
