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

## Run the synthetic example

Generate the shared 3-D model, stations, events, and picks, then run a 3-D
inversion:

```bash
cd examples
bash run_pipeline.sh
```

## Run the validation scripts

Run the test scripts from the `tests` directory:

```bash
cd ../tests
python test_eikonal.py
python test_grid.py
python test_gradient.py
```
