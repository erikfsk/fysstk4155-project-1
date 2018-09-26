import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('SRTM_data_Norway_2.tif')

print(np.shape(terrain1[::10,::10]))
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='YlOrBr')
plt.colorbar(shrink=0.5, aspect=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()