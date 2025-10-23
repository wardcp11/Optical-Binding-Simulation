import matplotlib.pyplot as plt
import numpy as np

data = np.load("./data/velocity_data.npy")
print(f"{data.shape=}")

t = np.linspace(0, 100, data.shape[0])

# plt.figure(figsize=(8, 5))
plt.plot(t, data[:, :, 0], label="x-data")
plt.plot(t, data[:, :, 1], label="y-data", linestyle="--")
plt.plot(t, data[:, :, 2], label="z-data", linestyle=":")
plt.xlabel("time step percentage")
plt.ylabel("velocity")
plt.grid(True)
plt.legend(loc="lower right")
plt.savefig("./plotting/velocity_data.png", dpi=300)


# data = np.load("../data/forces_data.npy")
data = np.load("./data/forces_data.npy")
plt.figure()

plt.plot(t, data[:, :, 0], label="x-data")
plt.plot(t, data[:, :, 1], label="y-data", linestyle="--")
plt.plot(t, data[:, :, 2], label="z-data", linestyle=":")
plt.xlabel("time step percentage")
plt.ylabel("forces")
plt.legend()
plt.grid(True)


plt.show()
