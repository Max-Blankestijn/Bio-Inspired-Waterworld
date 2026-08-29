"""
Combine the four MAP-Elites archive fitness heatmaps into one figure
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# run label -> metrics dir, x-axis label, y-axis label
RUNS = [
    ("Position", "waterworld_metrics_position", "Final centroid x", "Final centroid y"),
    ("Dispersion", "waterworld_metrics_dispersion", "Final x-spread", "Final y-spread"),
    ("Kinematics", "waterworld_metrics_kinematics", "Mean speed", "Speed variability"),
    ("Diet", "waterworld_metrics_diet", "Food-eaten fraction", "Poison-eaten fraction"),
]

BINS = 10  
OUTPUT = Path("waterworld_metrics_comparison/archive_fitness_heatmap_grid.png")


def load_fitness_grid(metrics_dir):
    grid = np.full((BINS, BINS), np.nan)
    with (Path(metrics_dir) / "archive_validation.csv").open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            grid[int(row["cell_x"]), int(row["cell_y"])] = float(row["validation_mean"])
    return grid


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    figure_width_in = 20
    figure, axes = plt.subplots(1, 4, figsize=(figure_width_in, 5), constrained_layout=True)
    scale = (0.59 / 0.99) * (figure_width_in / 10)
    title_size, label_size, tick_size = 20 * scale, 16 * scale, 13 * scale

    for (title, metrics_dir, xlabel, ylabel), axis in zip(RUNS, axes.flat):
        grid = load_fitness_grid(metrics_dir)
        image = axis.imshow(grid.T, origin="lower", cmap="viridis")
        axis.set_title(title, fontsize=title_size)
        axis.set_xlabel(xlabel, fontsize=label_size)
        axis.set_ylabel(ylabel, fontsize=label_size)
        axis.tick_params(labelsize=tick_size)
        colorbar = figure.colorbar(image, ax=axis, label="Elite team reward")
        colorbar.set_label("Elite team reward", fontsize=label_size)
        colorbar.ax.tick_params(labelsize=tick_size)

    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"Combined heatmap saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
