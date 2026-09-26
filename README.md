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

## Generate fixed synthetic data

The numbered scripts create one fixed dataset under `examples/data/`:

```bash
cd examples
python 00_gen_velocity.py
python 01_gen_stations.py
python 02_gen_events.py
python 03_gen_picks.py
```

## Run the inversion

Edit the stage settings at the top of `examples/run_inversion.sh`, then run:

```bash
bash run_inversion.sh
```

The workflow is explicit and sequential: 1-D velocity inversion, event
relocation, then 3-D velocity inversion. It writes `inversion_1d.png`,
`relocation.png`, and `inversion_3d.png` under `examples/figures/`.

## Run the gradient validation

Run the retained Eikonal backward check from the `tests` directory:

```bash
cd ../tests
python test_gradient.py
```

The script saves `tests/figures/gradient_test.png` and displays the two
Taylor-convergence panels when an interactive Matplotlib backend is available.
