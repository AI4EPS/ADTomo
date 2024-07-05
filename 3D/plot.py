import h5py
import matplotlib.pyplot as plt

# read file and set title
h5file = "result.h5" 
desc = "Layer_"


with h5py.File(h5file, "r") as f:
    key = list(f.keys())[0] 
    vel0 = f[key][()]


depths = [i for i in range(17)] # depths to plot

for i in depths:
    x = vel0[:,:,i]
    ind = str(i)
    plt.figure()
    plt.pcolormesh(x,vmin=5,vmax=7)
    plt.colorbar()
    plt.title('Python ' + ind)
    plt.savefig(desc+ind+".png")
