Core files:
Model,Optimize,Data

Syn_data_3d: Generates picks from PyTorch

Run: Performs inversion for specified set of picks

#
python setup.py \
Should download Libtorch and Eigen libraries as well as setting up customops eik2d_cpp and eik3d_cpp

python syn_data_3d.py \
Generates pick file computed from PyTorch for same events and stations as benchmark case

python run.py \
Performs inversion





