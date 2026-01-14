import numpy as cp
import imageio.v2 as imageio
import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from constants import L, dt, w0, k, E0, use_gaussian_beam

N = 256
x = cp.linspace(-L / 2, L / 2, N)
X, Y = cp.meshgrid(x, x)

uniform = E0 * cp.ones_like(X)
intensity = cp.abs(uniform) ** 2

frame_time_step = 1000


def gen_frames_uniform():
    data = np.load("./data/position_data.npy")

    time_steps = data.shape[0]
    num_of_particle = data.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot()

    # for t in range(0, time_steps, frame_time_step):
    for t in tqdm(
        range(0, time_steps, frame_time_step), desc="Generating frames", unit="step"
    ):
        ax.cla()
        for n in range(num_of_particle):
            x = data[t, n, 0]
            y = data[t, n, 1]

            ax.imshow(
                intensity,
                cmap="jet",
                alpha=0.5,
                extent=(-L / 2, L / 2, -L / 2, L / 2),
                origin="lower",
            )
            ax.scatter(x, y)
            ax.set_xlim(-L / 2, L / 2)
            ax.set_ylim(-L / 2, L / 2)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        plt.savefig(f"./plotting/videos/frames/2D/time_{t}.png")


def gen_frames_gaussian():
    gaussian = E0 * cp.exp(1j * k * 0) * cp.exp(-(X**2 + Y**2) / w0**2)
    intensity = cp.abs(gaussian) ** 2
    data = np.load("./data/position_data.npy")

    time_steps = data.shape[0]
    num_of_particle = data.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot()

    for t in tqdm(
        range(0, time_steps, frame_time_step), desc="Generating frames", unit="step"
    ):
        ax.cla()
        for n in range(num_of_particle):
            x = data[t, n, 0]
            y = data[t, n, 1]

            ax.imshow(
                intensity,
                cmap="jet",
                alpha=0.5,
                extent=(-L / 2, L / 2, -L / 2, L / 2),
                origin="lower",
            )
            ax.scatter(x, y)
            ax.set_xlim(-L / 2, L / 2)
            ax.set_ylim(-L / 2, L / 2)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        plt.savefig(f"./plotting/videos/frames/2D/time_{t}.png")


def gen_video():
    # images = glob.glob("./data/frames/2D/time_*.png")
    images = glob.glob("./plotting/videos/frames/2D/time_*.png")
    images.sort(key=lambda f: int(re.search(r"time_(\d+)", f).group(1)))  # type: ignore

    # with imageio.get_writer("./data/video_2D.mp4", fps=60) as writer:
    with imageio.get_writer("./plotting/videos/video_2D.mp4", fps=30) as writer:
        for filename in images:
            print(filename)
            image = imageio.imread(filename)
            writer.append_data(image)  # type: ignore

    print("Generated video")

    for image in images:
        try:
            os.remove(image)
        except OSError as e:
            print(f"Error deleting {image}: {e}")


gen_frames = gen_frames_gaussian if use_gaussian_beam else gen_frames_uniform

if __name__ == "__main__":
    gen_frames()
    gen_video()
