import numpy as np

dipole_moment = np.load("./induced_dipoles.npy")
np.savetxt("./induced_dipoles.txt", dipole_moment)

position = np.load("./position_data.npy")
np.savetxt("./position_data.txt", position[-1, :, :])
