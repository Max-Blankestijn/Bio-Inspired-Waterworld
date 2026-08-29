# MAP-Elites vs. CMA-ES on Multi-Agent WaterWorld

Compares MAP-Elites (searching for diverse behaviours) against a plain
CMA-ES baseline (searching for one high-fitness solution) on EvoJAX's
`MultiAgentWaterWorld` environment, across four behaviour descriptors:
**position**, **dispersion**, **kinematics**, and **diet**.

## Files

| File | What it does |
|---|---|
| `waterworld.py` | Trains MAP-Elites for one behaviour descriptor. |
| `waterworld_cma_baseline.py` | Trains the plain CMA-ES baseline. |
| `policy.py` | The shared MLP controller both scripts evolve. |
| `save_gif.py` | Saves a rollout as a GIF. |
| `compare_cma_baseline.py` | Builds the training-curve comparison figures across all runs. |
| `archive_heatmap_grid.py` | Builds the combined 4-panel archive fitness heatmap. |
| `archive_coverage_plot.py` | Builds the archive coverage vs. generation figure. |

Each `waterworld_metrics_<name>/` directory holds one run's output
(`training_metrics.csv`, `archive_validation.csv` for MAP-Elites runs,
`training_curves.png`), and `waterworld_metrics_comparison/` holds the
combined report figures. The five `waterworld_<name>.gif` files are
example rollouts of the best solution found by each run.

## Setup

Requires Python with `jax`, `evojax`, `evosax`, `flax`, `numpy`,
`matplotlib`, and `pillow` installed.

## Running

Each script is configured by editing the `CONFIG` block at the top of the file, then running it directly.

**Train MAP-Elites for one behaviour descriptor:**
```bash
python waterworld.py
```
`CONFIG.behaviour_descriptor` defaults to `"position"`. To reproduce the
other three runs, edit `behaviour_descriptor`, `metrics_dir`, and `gif` at
the top of `waterworld.py` to one of the other three descriptors (e.g.
`"dispersion"` / `"waterworld_metrics_dispersion"` /
`"waterworld_dispersion.gif"`) and rerun once per descriptor.

**Train the CMA-ES baseline:**
```bash
python waterworld_cma_baseline.py
```

**Rebuild the report figures** (after all five runs above have produced
their metrics directories (this only reads their CSVs, it doesn't retrain
anything)):
```bash
python compare_cma_baseline.py
python archive_heatmap_grid.py
python archive_coverage_plot.py
```
