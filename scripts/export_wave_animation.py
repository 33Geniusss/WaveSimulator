"""Render an exported wavefield archive as a GIF, with stationary receivers."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Directory exported from Wave Lab 2D")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.frames < 2 or args.fps < 1:
        parser.error("--frames >= 2 and --fps >= 1 required")
    data = np.load(args.directory / "wavefield.npz")
    config = json.loads((args.directory / "configuration.json").read_text(encoding="utf-8"))
    scene = config["simulation"]
    frames, times = data["frames"], data["frame_times"]
    selected = np.unique(np.linspace(0, len(frames)-1, min(args.frames, len(frames)), dtype=int))
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    limit = max(float(np.max(np.abs(frames))) / 4, 1e-7)
    x = data["x"]
    y = data["y"] if "y" in data else data["z"]
    dx, dy = x[1]-x[0], y[1]-y[0]
    image = ax.imshow(frames[0], origin="upper", extent=[x[0]-dx/2, x[-1]+dx/2, y[-1]+dy/2, y[0]-dy/2], cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=ax, label="Pressure / a.u. (display gain 4x)")
    ax.axhline(scene["interface_depth"], color="gray", linestyle="--")
    ax.axhline(scene["source_depth"], color="gray", linestyle=":", linewidth=.7)
    ax.scatter([scene["source_x"]], [scene["source_depth"]], marker="*", c="gold", edgecolors="black", s=90)
    for receiver in config["receivers"]:
        ax.scatter([receiver["x"]], [scene["source_depth"]], s=25, facecolors="none", edgecolors="black")
        ax.text(receiver["x"], scene["source_depth"]-.12, f"R{receiver['id']}", ha="center", fontsize=8)
    ax.set(xlabel="x / m", ylabel="y / m (positive down)")
    title = ax.set_title("")

    def update(index):
        image.set_data(frames[index])
        title.set_text(f"Wave Lab 2D | physical time {times[index]:.3f} s")
        return image, title

    animation = FuncAnimation(fig, update, frames=selected, interval=1000/args.fps)
    output = args.output or args.directory / "wave_animation.gif"
    animation.save(output, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
