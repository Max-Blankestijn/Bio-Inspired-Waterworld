"""MAP-Elites training for EvoJAX's multi-agent WaterWorld."""

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from evojax.task.ma_waterworld import MultiAgentWaterWorld, TYPE_FOOD, TYPE_POISON
from policy import Policy
from save_gif import save_gif

NUM_AGENTS, NUM_ITEMS, MAX_STEPS, HIDDEN_SIZE, SEED = 8, 100, 500, 32, 42

# Configuration
CONFIG = SimpleNamespace(
    generations=300,
    population_size=64,
    initial_samples=256,
    eval_episodes=10,
    validation_episodes=16,
    validation_candidates=10,
    grid_validation_episodes=16,
    max_parallel_episodes=512,
    behaviour_bins=10,
    behaviour_descriptor="position", # options are position, dispersion, kinematics, diet
    mutation_std=0.02,
    exploit_fraction=0.5,
    tournament_size=4,
    render=True,
    metrics_dir="waterworld_metrics_position",
    gif="waterworld_position.gif",
)


class Elite:
    def __init__(self, params, fitness, behaviour):
        self.params = params
        self.fitness = fitness
        self.behaviour = behaviour


def run_episode(env, policy, params, key, max_steps, render=False):
    """Roll out one episode and return reward plus food/poison counts."""
    _, reset_key = random.split(key)
    state = env.reset(reset_key[None, :])
    rewards_total = jnp.zeros((state.obs.shape[1],), dtype=jnp.float32)
    food, poison = jnp.array(0.0), jnp.array(0.0)

    def transition(state, rewards_total, food, poison):
        # I use argmax here to make the action deterministic as the standard waterworld environment samples from a probability based on the softmax output of the policy
        action_probabilities = policy.apply(params, state.obs[0])[None, ...]
        action = jax.nn.one_hot(jnp.argmax(action_probabilities, axis=-1), action_probabilities.shape[-1])
        state, reward, _ = env.step(state, action)
        reward = reward[0]

        return (state, rewards_total + reward, food + jnp.sum(jnp.maximum(reward, 0.0)), poison + jnp.sum(jnp.maximum(-reward, 0.0)))

    if render:
        frames = []
        for _ in range(max_steps):
            state, rewards_total, food, poison = transition(state, rewards_total, food, poison)
            frames.append(MultiAgentWaterWorld.render(state, task_id=0))
        return rewards_total, food, poison, state, frames

    def scan_step(carry, _):
        return transition(*carry), None

    (state, rewards_total, food, poison), _ = lax.scan(scan_step, (state, rewards_total, food, poison), None, length=max_steps)

    return rewards_total, food, poison, state, []


def behaviour_descriptor_position(final_state):
    """Final team centroid"""
    agents = final_state.agent_state
    return jnp.clip(jnp.array((jnp.mean(agents.pos_x) / 600.0, jnp.mean(agents.pos_y) / 600.0)), 0.0, 1.0)


def behaviour_descriptor_dispersion(final_state):
    """Final team spread"""
    agents = final_state.agent_state
    return jnp.clip(jnp.array((jnp.std(agents.pos_x) / 300.0, jnp.std(agents.pos_y) / 300.0)), 0.0, 1.0)


def behaviour_descriptor_kinematics(final_state):
    """Final team speed and speed uniformity."""
    agents = final_state.agent_state
    speed = jnp.sqrt(agents.vel_x ** 2 + agents.vel_y ** 2)
    return jnp.clip(jnp.array((jnp.mean(speed) / 20.0, jnp.std(speed) / 20.0)), 0.0, 1.0)


def behaviour_descriptor_diet(final_state):
    """Fraction of food and of poison eaten over the episode."""
    items = final_state.item_state
    eaten = 1 - items.valid
    food_frac = jnp.sum(eaten * (items.bubble_type == TYPE_FOOD)) / jnp.sum(items.bubble_type == TYPE_FOOD)
    poison_frac = jnp.sum(eaten * (items.bubble_type == TYPE_POISON)) / jnp.sum(items.bubble_type == TYPE_POISON)
    return jnp.clip(jnp.array((food_frac, poison_frac)), 0.0, 1.0)


BEHAVIOUR_DESCRIPTORS = {
    "position": behaviour_descriptor_position,
    "dispersion": behaviour_descriptor_dispersion,
    "kinematics": behaviour_descriptor_kinematics,
    "diet": behaviour_descriptor_diet,
}


def archive_index(behaviour, bins):
    index = jnp.clip(jnp.floor(behaviour * bins).astype(jnp.int32), 0, bins - 1)
    return int(index[0]), int(index[1])


def mutate_params(params, key, mutation_std):
    '''Mutate with random noise according to standard deviation'''
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = random.split(key, len(leaves))
    leaves = [leaf + mutation_std * random.normal(k, leaf.shape, leaf.dtype) for leaf, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def stack_population(population):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *population)


def main():
    # Set up environment and policy
    args = CONFIG

    env = MultiAgentWaterWorld(num_agents=NUM_AGENTS, num_items=NUM_ITEMS,
                               max_steps=MAX_STEPS)
    policy = Policy(action_dim=env.act_shape[-1], hidden_size=HIDDEN_SIZE)
    key = random.PRNGKey(SEED)
    observation_shape = (NUM_AGENTS, env.obs_shape[-1])

    behaviour_descriptor = BEHAVIOUR_DESCRIPTORS[args.behaviour_descriptor]

    def evaluate_episode(params, episode_key):
        rewards, _, _, final_state, _ = run_episode(env, policy, params, episode_key, MAX_STEPS)
        return jnp.sum(rewards), behaviour_descriptor(final_state)

    def evaluate_population(population, evaluation_keys):
        rewards, behaviours = jax.vmap(jax.vmap(evaluate_episode, in_axes=(None, 0)), in_axes=(0, 0))(population, evaluation_keys)
        mean_rewards = jnp.mean(rewards, axis=1)
        return mean_rewards, jnp.mean(behaviours, axis=1)

    def evaluate_population_stats(population, evaluation_keys):
        rewards, behaviours = jax.vmap(
            jax.vmap(evaluate_episode, in_axes=(None, 0)), in_axes=(0, 0))(population, evaluation_keys)
        return (jnp.mean(rewards, axis=1), jnp.std(rewards, axis=1), jnp.mean(behaviours, axis=1))

    compiled_evaluate = jax.jit(evaluate_population)
    compiled_evaluate_stats = jax.jit(evaluate_population_stats)
    archive: Dict[Tuple[int, int], Elite] = {}
    history = []

    def evaluate_in_chunks(candidates, episode_keys, with_stats=False):
        """Evaluate a population without making my poor gpu suffer by running out of memory"""
        batch_size = max(1, args.max_parallel_episodes // len(episode_keys))
        evaluator = compiled_evaluate_stats if with_stats else compiled_evaluate
        result_chunks = []
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            keys = jnp.broadcast_to(episode_keys, (len(chunk), len(episode_keys), 2))
            result_chunks.append(jax.device_get(evaluator(stack_population(chunk), keys)))
        return tuple(np.concatenate(parts, axis=0) for parts in zip(*result_chunks))

    def record_metrics(generation, new_elites, gen_best_reward, gen_mean_reward, best_ever_reward):
        fitnesses = [elite.fitness for elite in archive.values()]
        history.append({
            "generation": generation,
            "coverage": len(archive),
            "new_elites": new_elites,
            "gen_best_reward": gen_best_reward,
            "gen_mean_reward": gen_mean_reward,
            "best_ever_reward": best_ever_reward,
            "best_search_reward": max(fitnesses),
            "mean_archive_reward": sum(fitnesses) / len(fitnesses),
            "archive_reward_sum": sum(fitnesses),
        })

    def evaluate_and_insert(candidates, evaluation_key):
        episode_keys = random.split(evaluation_key, args.eval_episodes)
        fitnesses, behaviours = evaluate_in_chunks(candidates, episode_keys)
        gen_best_reward = float(np.max(fitnesses))
        gen_mean_reward = float(np.mean(fitnesses))

        # Keep only the strongest contender per cell
        contenders = {}
        for candidate, fitness, behaviour in zip(candidates, fitnesses, behaviours):
            cell, fitness = archive_index(behaviour, args.behaviour_bins), float(fitness)
            if cell not in contenders or fitness > contenders[cell][1]:
                contenders[cell] = (candidate, fitness, tuple(map(float, behaviour)))

        occupied_cells = [cell for cell in contenders if cell in archive]
        incumbent_fitness = {}
        if occupied_cells:
            incumbent_params = [archive[cell].params for cell in occupied_cells]
            padded_params = incumbent_params + [
                incumbent_params[i % len(incumbent_params)]
                for i in range(len(candidates) - len(incumbent_params))]
            scores, _ = evaluate_in_chunks(padded_params, episode_keys)
            incumbent_fitness = dict(zip(
                occupied_cells, map(float, scores[:len(occupied_cells)])
            ))

        inserted = 0
        for cell, (candidate, fitness, behaviour) in contenders.items():
            if cell not in archive or fitness > incumbent_fitness[cell]:
                archive[cell] = Elite(candidate, fitness, behaviour)
                inserted += 1
        return inserted, gen_best_reward, gen_mean_reward

    print("=" * 60 + "\nMAP-Elites: EvoJAX Multi-Agent WaterWorld\n" + "=" * 60)
    print(f"Devices: {jax.devices()} | policy: {env.obs_shape[-1]} -> "
          f"{HIDDEN_SIZE} -> {env.act_shape[-1]}")
    print(f"Environment: {NUM_AGENTS} agents | {NUM_ITEMS} items | {MAX_STEPS} steps")
    print(f"Archive: {args.behaviour_bins}x{args.behaviour_bins} | population: {args.population_size}")
    print("Compiling batched evaluator and seeding archive...")

    # Seed from independent random policies, rather than tiny perturbations of one policy.
    key, initial_key, eval_key = random.split(key, 3)
    initial_population = [policy.init(k, jnp.zeros(observation_shape))
                          for k in random.split(initial_key, args.initial_samples)]
    _, gen_best_reward, gen_mean_reward = evaluate_and_insert(initial_population, eval_key)
    best_ever_reward = gen_best_reward
    record_metrics(0, len(archive), gen_best_reward, gen_mean_reward, best_ever_reward)
    print(f"Initial archive coverage: {len(archive)}/{args.behaviour_bins ** 2} cells")

    for generation in range(1, args.generations + 1):
        cells = tuple(archive)
        key, parent_key, mutation_key, eval_key = random.split(key, 4)
        uniform_count = int(args.population_size * (1.0 - args.exploit_fraction))
        exploit_count = args.population_size - uniform_count
        uniform_indices = random.randint(parent_key, (uniform_count,), 0, len(cells))
        key, tournament_key = random.split(key)
        tournament_indices = random.randint(
            tournament_key, (exploit_count, args.tournament_size), 0, len(cells)
        )
        cell_fitness = jnp.array([archive[cell].fitness for cell in cells])
        winner_columns = jnp.argmax(cell_fitness[tournament_indices], axis=1)
        exploit_indices = tournament_indices[jnp.arange(exploit_count), winner_columns]
        parent_indices = jnp.concatenate((uniform_indices, exploit_indices))
        mutation_keys = random.split(mutation_key, args.population_size)
        candidates = [mutate_params(archive[cells[int(i)]].params, k, args.mutation_std)
                      for i, k in zip(parent_indices, mutation_keys)]
        inserted, gen_best_reward, gen_mean_reward = evaluate_and_insert(candidates, eval_key)
        best_ever_reward = max(best_ever_reward, gen_best_reward)
        record_metrics(generation, inserted, gen_best_reward, gen_mean_reward, best_ever_reward)
        if generation == 1 or generation % 10 == 0:
            best = max(elite.fitness for elite in archive.values())
            print(f"Generation {generation:4d}/{args.generations} | coverage "
                  f"{len(archive):3d}/{args.behaviour_bins ** 2} | new elites {inserted:2d} | "
                  f"best reward {best:7.2f}")

    # Rescore top performing elites
    shortlist = sorted(archive.items(), key=lambda entry: entry[1].fitness,
                       reverse=True)[:args.validation_candidates]
    shortlist_cells, shortlist_elites = zip(*shortlist)
    key, validation_key = random.split(key)
    validation_keys = random.split(validation_key, args.validation_episodes)
    validation_rewards, _ = evaluate_in_chunks(
        [elite.params for elite in shortlist_elites], validation_keys
    )
    best_index = int(jnp.argmax(validation_rewards))
    best_cell, best = shortlist_cells[best_index], shortlist_elites[best_index]
    print(f"\nBest cell: {best_cell}; search reward: {best.fitness:.2f}; behaviour: {best.behaviour}")
    held_out_reward = float(validation_rewards[best_index])
    print(f"Held-out team reward ({args.validation_episodes} episodes): {held_out_reward:.2f} "
          f"| per-agent: {held_out_reward / NUM_AGENTS:.2f}")

    # Validate heat map
    archive_cells = tuple(archive)
    key, grid_validation_key = random.split(key)
    grid_episode_keys = random.split(grid_validation_key, args.grid_validation_episodes)
    grid_means, grid_stds, _ = evaluate_in_chunks([archive[cell].params for cell in archive_cells], grid_episode_keys, with_stats=True)

    key, test_key = random.split(key)
    rewards, food, poison, _, frames = run_episode(env, policy, best.params, test_key, MAX_STEPS, args.render)
    rewards, food, poison = jax.device_get((rewards, food, poison))
    illustrative_team_reward = float(jnp.sum(rewards))
    print(f"Illustrative fresh team reward: {illustrative_team_reward:.2f} | "
          f"per-agent: {illustrative_team_reward / NUM_AGENTS:.2f} | "
          f"food: {float(food):.0f} | poison: {float(poison):.0f}")
    if args.render and save_gif(frames, args.gif, duration=40, loop=1):
        print(f"GIF saved to: {args.gif}")

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "training_metrics.csv").open("w", newline="") as metric_file:
        writer = csv.DictWriter(metric_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (metrics_dir / "archive_validation.csv").open("w", newline="") as archive_file:
        writer = csv.DictWriter(archive_file, fieldnames=(
            "cell_x", "cell_y", "search_reward", "validation_mean",
            "validation_std", "validation_per_agent_mean", "behaviour_x", "behaviour_y",
        ))
        writer.writeheader()
        for cell, mean, std in zip(archive_cells, grid_means, grid_stds):
            elite = archive[cell]
            writer.writerow({
                "cell_x": cell[0], "cell_y": cell[1],
                "search_reward": elite.fitness,
                "validation_mean": float(mean), "validation_std": float(std),
                "validation_per_agent_mean": float(mean) / NUM_AGENTS,
                "behaviour_x": elite.behaviour[0], "behaviour_y": elite.behaviour[1],
            })

    generations = [row["generation"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(generations, [row["coverage"] for row in history])
    axes[0, 0].set(title="Archive coverage", xlabel="Generation", ylabel="coverage")
    for series, label in (("gen_best_reward", "best (this generation)"),
                            ("gen_mean_reward", "mean (this generation)"),
                            ("best_ever_reward", "best ever (running max)"),
                            ("best_search_reward", "archive current max")):
        axes[0, 1].plot(generations, [row[series] for row in history], label=label)
    axes[0, 1].set(title="Search reward", xlabel="Generation", ylabel="team reward")
    axes[0, 1].legend(fontsize="small")
    for axis, key_name, title in (
            (axes[1, 0], "mean_archive_reward", "Mean archive reward"),
            (axes[1, 1], "archive_reward_sum", "Archive reward sum (QD-score)")):
        axis.plot(generations, [row[key_name] for row in history])
        axis.set(title=title, xlabel="Generation", ylabel=key_name.replace("_", " "))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    fitness_grid = np.full((args.behaviour_bins, args.behaviour_bins), np.nan)
    for cell, mean in zip(archive_cells, grid_means):
        fitness_grid[cell] = float(mean)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(fitness_grid.T, origin="lower", cmap="viridis")
    axis.set(title=(f"MAP-Elites held-out fitness — {args.behaviour_descriptor} "
                    f"behaviour ({args.grid_validation_episodes} episodes/cell)"),
                xlabel=f"{args.behaviour_descriptor} axis 1",
                ylabel=f"{args.behaviour_descriptor} axis 2")
    figure.colorbar(image, ax=axis, label="Mean total reward")
    figure.savefig(metrics_dir / "archive_fitness_heatmap.png", dpi=180)
    plt.close(figure)
    print(f"Metrics and figures saved to: {metrics_dir}")

if __name__ == "__main__":
    main()
