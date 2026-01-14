import numpy as np

data = np.load("./data/velocity_data.npy")

for i in range(3):
    print(data[5000:5050, 1, 2])
