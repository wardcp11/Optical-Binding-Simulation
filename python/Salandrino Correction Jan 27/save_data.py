import numpy as np

position_data = np.load(
    "./data/2_9_26_tests/initial_pos_to_stable_data/position_data.npy"
)

# save the last time step position data to load in "main.py"
np.save("./data/2_9_26_tests/stable_pos.npy", position_data[-1])
print("Saved data")
