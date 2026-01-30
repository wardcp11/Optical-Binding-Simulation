import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

num_of_particle = 4

position_salandrino = pd.read_csv(
    "./comparing_salandrinos_simulation/positions.csv", header=None
).to_numpy()

x_data = np.zeros((4, 9999))
y_data = np.zeros((4, 9999))

for i in range(position_salandrino.shape[0]):
    for j in range(position_salandrino.shape[1]):
        xy = position_salandrino[i, j].split(",")
        x = float(xy[0].removeprefix("{"))
        y = float(xy[1].removesuffix("}"))

        x_data[i, j] = x
        y_data[i, j] = y

path = "./data/4_particles_salandrino/"

data = np.load(path + "velocity_data.npy")
length = data.shape[0] - 1

plt.figure()
plt.grid(True)
plt.title("x Position vs Time Salandrino")

for n in range(num_of_particle):
    plt.plot(x_data[n])

plt.figure()
plt.grid(True)
plt.title("y Position vs Time Salandrino")

for n in range(num_of_particle):
    plt.plot(y_data[n])


def position_data():
    data = np.load(path + "position_data.npy")
    plt.figure()
    plt.title("x Position vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.savefig(path + "x Position vs Time.png")

    plt.figure()
    plt.title("y Position vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Position vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])


if __name__ == "__main__":
    position_data()
    plt.show()
