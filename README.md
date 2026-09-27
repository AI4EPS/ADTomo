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

## Run the synthetic 3-D example

The example has one workflow:

```text
00_gen_velocity.py -> 01_gen_stations.py -> 02_gen_events.py
                   -> 03_gen_picks.py -> run_inversion.sh -> inversion.py
```

`model.pt` is horizontally homogeneous. `model_true.pt` has the same
background and grid plus the 3-D checkerboard perturbation. Checkerboard
wavelengths are expressed directly in degrees longitude, degrees latitude,
and km depth. With the default wavelengths, the model spans 0.75° in
longitude, 0.75° in latitude, and -2 to 16 km in depth. The depth grid spacing
is 2 km, and the checkerboard depth wavelength is 5 km.

```bash
cd examples
bash data.sh
```

The event script uses 0.025° horizontal noise, 0.5 km depth noise, and 0.1 s
origin-time noise by default. Events are generated from 0 to 13 km depth.
Longitude and latitude noise is added directly in degrees; there is no
conversion from km. Stations are generated with random depths between -2 and
2 km.
`events_true.csv` is used to generate picks, while `events.csv` is the starting
catalog used by inversion. Depth is positive downward.

The generated tables use `station_id`, `longitude`, `latitude`, `depth_km` for
stations and `event_id`, `event_time`, `longitude`, `latitude`, `depth_km` for
events. Picks use `event_id`, `station_id`, `phase_type`, and `phase_time`.

## Run the joint inversion

There is one inversion entry point:

```bash
cd examples
bash run_inversion.sh
```

Edit `INPUT_DIR` and `OUTPUT_DIR` at the top of `run_inversion.sh` for each run.
Results are written to `OUTPUT_DIR`, with figures under `OUTPUT_DIR/figures/`.

The default is `NPROC=1`, so a direct run does not start distributed workers.
Increase `NPROC` only when the current environment has enough resources.

It always jointly inverts 3-D Vp/Vs, event longitude/latitude/depth, and event
origin times. The `ForwardGrid` keeps its existing boundary-extension behavior
when its local grid reaches beyond the global velocity-model grid. Edit the
visible values in `run_inversion.sh`. The `TRAINABLE` line controls which
parameters are optimized; valid names are `vp`, `vs`, `event_loc`, and
`event_time`:

```bash
TRAINABLE=vp,vs,event_loc,event_time bash run_inversion.sh
TRAINABLE=vp,vs bash run_inversion.sh
```

`event_loc` enables both horizontal and vertical event coordinates, while
`event_time` enables origin-time corrections. A learning-rate variable only
has an effect when its corresponding trainable group is selected.

Each run writes only `model.pt`, `events.csv`, `stations.csv`, `picks.csv`, and
`figures/` into `OUTPUT_DIR`. The shared synthetic truth files remain in
`data/`, and the output directory can therefore be used directly as the next
`INPUT_DIR`.

Adam uses separate rates for `vp`, `vs`, horizontal event location, vertical
event location, and origin time: `LR_VP`, `LR_VS`, `LR_LOC_HORI`,
`LR_LOC_VERT`, and `LR_ORIGIN_TIME`.

The generation scripts keep their experiment values in a small configuration
block at the top of each file; there are no command-line argument parsers.

## Run the gradient validation

Run the retained Eikonal backward check from the `tests` directory:

```bash
cd ../tests
python test_gradient.py
```

The script saves `tests/figures/gradient_test.png` and displays the two
Taylor-convergence panels when an interactive Matplotlib backend is available.
