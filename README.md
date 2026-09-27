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

## Run the example

From the repository root:

```bash
cd examples
bash data.sh
bash run_inversion.sh
```

## Run the test

From the repository root:

```bash
cd tests
python test_gradient.py
```
