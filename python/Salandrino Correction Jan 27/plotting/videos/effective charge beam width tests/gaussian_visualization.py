import numpy as np
import matplotlib.pyplot as plt


lam = 1
L = 10 * lam
x = np.linspace(-L / 2, L / 2, 1000)

w0_arr = [4, 40, 400]
label_arr = [r"4\lambda", r"40\lambda", r"400\lambda"]


plt.figure()

for i, w0 in enumerate(w0_arr):
    gaussian = np.exp(-(x**2) / (2 * w0**2)) / (w0 * np.sqrt(np.pi))
    plt.plot(gaussian, label=label_arr[i])

plt.legend()
plt.grid(True)
plt.show()
