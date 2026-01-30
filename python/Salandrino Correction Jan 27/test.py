import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det
from constants import num_of_particle, maxstep


path = "./data/"
# path = "./data/3_particles_stable/"
# path = "./data/30_particles/"

data = np.load(path + "velocity_data.npy")
length = data.shape[0] - 1


def position_data_3D():
    data = np.load(path + "position_data.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for n in range(num_of_particle):
        x = data[(maxstep // 2), n, 0]
        y = data[(maxstep // 2), n, 1]
        z = data[(maxstep // 2), n, 2]

        ax.scatter(x, y, z)
    plt.show()


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

    plt.show()


def velocity_data():
    data = np.load(path + "velocity_data.npy")
    plt.figure()
    plt.title("x Velocity vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.savefig(path + "x Velocity vs Time.png")

    plt.figure()
    plt.title("y Velocity vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Velocity vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def force_data():
    data = np.load(path + "forces_data.npy")
    plt.figure()
    plt.title("x Force vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.figure()
    plt.title("y Force vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Force vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def E_field_data():
    data = np.load(path + "E_n_data.npy")
    plt.figure()
    plt.title("x E_field_strength vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("E_field_strength")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.figure()
    plt.title("y E_field_strength vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("E_field_strength")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z E_field_strength vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("E_field_strength")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def dipole_moment_data():
    data = np.load(path + "p_i_data.npy")
    plt.figure()
    plt.title("x Dipole Moment vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dipole Moment")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.figure()
    plt.title("y Dipole Moment vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dipole Moment")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Dipole Moment vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dipole Moment")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def G_max_det():
    det_data = np.load(path + "G_det.npy")
    max_data = np.load(path + "G_max.npy")

    plt.figure()
    plt.title("G Det Data vs Time")
    plt.grid(True)
    plt.plot(det_data)

    plt.figure()
    plt.title("G Max Data vs Time")
    plt.grid(True)
    plt.plot(max_data)

    plt.show()


if __name__ == "__main__":
    # position_data_3D()
    position_data()
    velocity_data()
    force_data()
    E_field_data()
    dipole_moment_data()
    G_max_det()

    # position_data = np.load(path + "position_data.npy")
    # print(position_data.shape)
    #
    # x_data = position_data[15000, 0, 0]
    #
    # for n in range(num_of_particle):
    #     for i in range(num_of_particle):
    #         print(position_data[15000, n, i])
    #     print("\n")
