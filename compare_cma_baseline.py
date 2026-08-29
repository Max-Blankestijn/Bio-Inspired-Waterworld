"""
Compare MAP-Elites archives against the CMA-ES baseline
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    "position": Path("waterworld_metrics_position"),
    "dispersion": Path("waterworld_metrics_dispersion"),
    "kinematics": Path("waterworld_metrics_kinematics"),
    "diet": Path("waterworld_metrics_diet"),
    "cma": Path("waterworld_metrics_cma"),
}

OUTPUT_DIR = Path("waterworld_metrics_comparison")

LABEL_OVERRIDES = {"cma": "CMA-ES"}


def display_label(name):
    override = LABEL_OVERRIDES.get(name.lower())
    if override is not None:
        return override
    return name[:1].upper() + name[1:] if name else name


def load_history(metrics_dir):
    with (metrics_dir / "training_metrics.csv").open(newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def best_so_far_curve(history):
    return history["generation"], list(np.maximum.accumulate(history["best_ever_reward"]))


def gen_best_curve(history):
    # The current generation's best candidate
    if "gen_best_reward" in history:
        raw = history["gen_best_reward"]
    else:
        raw = history["best_reward"]
    return history["generation"], raw


def save_curve_figure(curve_fn, output_name, histories):
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for name in RUNS:
        generations, values = curve_fn(histories[name])
        axis.plot(generations, values, label=display_label(name))
    axis.set_xlabel("Generation", fontsize=16)
    axis.set_ylabel("Total team reward", fontsize=16)
    axis.grid(alpha=0.25)
    axis.tick_params(labelsize=13)
    axis.legend(fontsize=13)
    figure.savefig(OUTPUT_DIR / output_name, dpi=180)
    plt.close(figure)


def save_grid_figure(curve_fn, output_name, histories):
    curves = {name: curve_fn(histories[name]) for name in RUNS}
    all_values = [v for _, values in curves.values() for v in values]
    pad = 0.05 * (max(all_values) - min(all_values))
    y_limits = (min(all_values) - pad, max(all_values) + pad)

    names = list(RUNS)
    cols = math.ceil(math.sqrt(len(names)))
    rows = math.ceil(len(names) / cols)
    figure_width_in = 4 * cols
    figure, axes = plt.subplots(rows, cols, figsize=(figure_width_in, 3 * rows), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    scale = (0.59 / 0.9) * (figure_width_in / 10)
    title_size, label_size, tick_size = 20 * scale, 16 * scale, 13 * scale

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, name in enumerate(names):
        axis = axes[index // cols][index % cols]
        generations, values = curves[name]
        axis.plot(generations, values, color=color_cycle[index % len(color_cycle)])
        axis.set_title(display_label(name), fontsize=title_size)
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=tick_size)
    for index in range(len(names), rows * cols):
        axes[index // cols][index % cols].axis("off")
    figure.supxlabel("Generation", fontsize=label_size)
    figure.supylabel("Total team reward", fontsize=label_size)
    figure.savefig(OUTPUT_DIR / output_name, dpi=180)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract histories
    histories = {name: load_history(metrics_dir) for name, metrics_dir in RUNS.items()}

    # Create and save plots
    save_curve_figure(best_so_far_curve, "fitness_comparison.png", histories)
    save_grid_figure(gen_best_curve, "gen_best_comparison_grid.png", histories)

    print(f"Comparison figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
