import numpy as np
import matplotlib.pyplot as plt

unmodified_data = np.abs(np.load("./eigenvalues_unmodified_G.npy"))
modified_data = np.abs(np.load("./eigenvalues_modified_G.npy"))

x = np.asarray(range(1, unmodified_data.shape[0] + 1))
fig, ax = plt.subplots()
ax.stem(x, unmodified_data, label="eigvals unmodified", markerfmt="orange")
ax.stem(x, modified_data, label="eigvals w/ additional radius")
ax.set_title("Eigenvalue Magnitude Comparison")
ax.grid(True)
ax.legend()

plt.savefig("./eigenvalue_comparison.png", dpi=300)
plt.show()
