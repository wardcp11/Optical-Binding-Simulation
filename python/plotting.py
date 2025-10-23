import imageio.v2 as imageio
import glob
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from constants import L


def gen_frames():
    data = np.load("../data/position_data.npy")
    print(f"{data.shape=}")

    time_steps = data.shape[0]
    num_of_particle = data.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for t in range(time_steps):
        for n in range(num_of_particle):
            x = data[t, n, 0]
            y = data[t, n, 1]
            z = data[t, n, 2]
            ax.scatter(x, y, z)
            ax.set_xlim(-L / 2, L / 2)
            ax.set_ylim(-L / 2, L / 2)
            ax.set_zlim(-L / 2, L / 2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        plt.savefig(f"../data/frames/time_{t}.png")
        ax.cla()


def gen_video():
    images = glob.glob("../data/frames/time_*.png")
    images.sort(key=lambda f: int(re.search(r"time_(\d+)", f).group(1)))

    with imageio.get_writer("../data/video.mp4", fps=60) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("video saved")

    for image in images:
        try:
            os.remove(image)
        except OSError as e:
            print(f"Error deleting {image}: {e}")


if __name__ == "__main__":
    print("Double check plotting locations and frame generation locations")
    exit()
    gen_frames()
    gen_video()
