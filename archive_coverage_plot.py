"""
Plot MAP-Elites archive coverage vs. generation for all four behaviour descriptors on one figure
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# legend label, metrics dir
RUNS = [
    ("Position", "waterworld_metrics_position"),
    ("Dispersion", "waterworld_metrics_dispersion"),
    ("Kinematics", "waterworld_metrics_kinematics"),
    ("Diet", "waterworld_metrics_diet"),
]

TOTAL_CELLS = 100  # behaviour_bins=10 -> 10x10 archive
OUTPUT = Path("waterworld_metrics_comparison/archive_coverage_comparison.png")


def load_coverage(metrics_dir):
    metrics_dir = Path(metrics_dir)
    with (metrics_dir / "training_metrics.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    generations = [int(float(row["generation"])) for row in rows]
    coverage = [100.0 * int(row["coverage"]) / TOTAL_CELLS for row in rows]
    return generations, coverage


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    figure_width_in = 7
    figure, axis = plt.subplots(figsize=(figure_width_in, 5), constrained_layout=True)
    scale = (0.59 / 0.4) * (figure_width_in / 10)
    label_size, tick_size, legend_size = 16 * scale, 13 * scale, 13 * scale

    for label, metrics_dir in RUNS:
        generations, coverage = load_coverage(metrics_dir)
        axis.plot(generations, coverage, label=label)
    axis.set_xlabel("Generation", fontsize=label_size)
    axis.set_ylabel("Archive coverage (%)", fontsize=label_size)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.tick_params(labelsize=tick_size)
    axis.legend(fontsize=legend_size)

    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"Coverage comparison figure saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
