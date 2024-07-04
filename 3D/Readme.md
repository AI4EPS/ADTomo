Order to run code
- setup.py
- 3D/syn _data_3d.py
- 3D/run.py

#
python setup.py \
Downloads Libtorch and Eigen libraries as well as setting up customops eik2d_cpp and eik3d_cpp

python syn_data_3d.py \
Generates pick file computed from PyTorch for same events and stations as benchmark case

python run.py \
Performs inversion

#
syn_data_3d.py can both generate checkboard pattern or anomaly by toggling 'anomaly' on or off

Path to correct pick file needs to be edited in run.py before starting inversion.
Julia picks are provided in 'data_benchmark' (anomaly case) or 'data_benchmark_checker' (checkerboard case).
