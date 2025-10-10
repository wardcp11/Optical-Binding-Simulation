import imageio.v2 as imageio
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from constants import L


def gen_frames():
    data = np.load("./data/position_data.npy")
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
            ax.set_xlim(-1e5 * L, 1e5 * L)
            ax.set_ylim(-1e5 * L, 1e5 * L)
            ax.set_zlim(-1e5 * L, 1e5 * L)
        plt.savefig(f"./data/frames/time_{t}.png")
        ax.cla()


def gen_video():
    images = glob.glob("./data/frames/time_*.png")
    images.sort(key=lambda f: int(re.search(r"time_(\d+)", f).group(1)))

    with imageio.get_writer("./data/video.mp4", fps=30) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("video saved")


if __name__ == "__main__":
    gen_frames()
    gen_video()
