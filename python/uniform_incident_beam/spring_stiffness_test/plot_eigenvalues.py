import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import glob
import re


def gen_frames():
    data = np.load("./data/eigenvalues_data.npy")
    x = np.asarray(range(1, data.shape[1] + 1))

    time_steps = data.shape[0]

    fig, ax = plt.subplots()
    for t in range(0, time_steps, 100):
        ax.cla()
        ax.stem(x, data[t])
        ax.set_title(r"Magnitude of Eigenvalues of $G$")
        ax.grid(True)

        plt.savefig(f"./plotting/videos/frames/eigenvalues/time_{t}.png")


def gen_video():
    images = glob.glob("./plotting/videos/frames/eigenvalues/time_*.png")
    images.sort(key=lambda f: int(re.search(r"time_(\d+)", f).group(1)))  # type: ignore

    with imageio.get_writer("./plotting/videos/video_eigenvalues.mp4", fps=1) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)  # type: ignore

    print("Generated video")

    for image in images:
        try:
            os.remove(image)
        except OSError as e:
            print(f"Error deleting {image}: {e}")


if __name__ == "__main__":
    gen_frames()
    gen_video()
