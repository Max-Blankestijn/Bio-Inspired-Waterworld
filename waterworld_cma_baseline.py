"""
Performs a CMA-ES training run of the Waterworld environment, used as a baseline
"""

import csv
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from evosax.algorithms import CMA_ES
from evojax.task.ma_waterworld import MultiAgentWaterWorld
from policy import Policy
from save_gif import save_gif
from waterworld import (HIDDEN_SIZE, MAX_STEPS, NUM_AGENTS, NUM_ITEMS, SEED, run_episode)

CONFIG = SimpleNamespace(
    generations=300,
    population_size=64,
    eval_episodes=10,
    validation_episodes=16,
    max_parallel_episodes=512,
    render=True,
    metrics_dir="waterworld_metrics_cma",
    gif="waterworld_cma.gif",
)


def main():
    # Set up environment and policy
    args = CONFIG

    env = MultiAgentWaterWorld(num_agents=NUM_AGENTS, num_items=NUM_ITEMS, max_steps=MAX_STEPS)
    policy = Policy(action_dim=env.act_shape[-1], hidden_size=HIDDEN_SIZE)
    key = random.PRNGKey(SEED)

    key, init_key = random.split(key)
    dummy_obs = jnp.zeros((NUM_AGENTS, env.obs_shape[-1]))
    init_params = policy.init(init_key, dummy_obs)

    def evaluate_episode(params, episode_key):
        rewards, _, _, _, _ = run_episode(env, policy, params, episode_key, MAX_STEPS)
        return jnp.sum(rewards)

    def evaluate_population(population, evaluation_keys):
        rewards = jax.vmap(jax.vmap(evaluate_episode, in_axes=(None, 0)), in_axes=(0, 0))(population, evaluation_keys)
        return jnp.mean(rewards, axis=1)

    compiled_evaluate = jax.jit(evaluate_population)

    def evaluate_in_chunks(population, episode_keys):
        """Separate in chunks to avoid running out of memory on my poor abused laptop's GPU ):"""
        total = jax.tree_util.tree_leaves(population)[0].shape[0]
        batch_size = max(1, args.max_parallel_episodes // len(episode_keys))
        chunks = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            chunk = jax.tree_util.tree_map(lambda leaf: leaf[start:end], population)
            keys = jnp.broadcast_to(episode_keys, (end - start, len(episode_keys), 2))
            chunks.append(jax.device_get(compiled_evaluate(chunk, keys)))
        return np.concatenate(chunks, axis=0)

    strategy = CMA_ES(population_size=args.population_size, solution=init_params)
    es_params = strategy.default_params
    key, es_init_key = random.split(key)
    es_state = strategy.init(es_init_key, init_params, es_params)

    history = []
    best_ever_params, best_ever_reward = None, -np.inf

    # Main training loop
    for generation in range(1, args.generations + 1):
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        population, es_state = strategy.ask(ask_key, es_state, es_params)

        episode_keys = random.split(eval_key, args.eval_episodes)
        mean_rewards = evaluate_in_chunks(population, episode_keys)

        es_state, _ = strategy.tell(
            tell_key, population, -mean_rewards, es_state, es_params)  # CMA-ES minimizes

        gen_best_idx = int(np.argmax(mean_rewards))
        gen_best_reward = float(mean_rewards[gen_best_idx])
        if gen_best_reward > best_ever_reward:
            best_ever_reward = gen_best_reward
            best_ever_params = jax.tree_util.tree_map(
                lambda leaf: leaf[gen_best_idx], population)

        history.append({
            "generation": generation,
            "best_reward": gen_best_reward,
            "mean_reward": float(np.mean(mean_rewards)),
            "reward_std": float(np.std(mean_rewards)),
            "best_ever_reward": best_ever_reward,
        })

        if generation == 1 or generation % 10 == 0:
            print(f"Generation {generation:4d}/{args.generations} | " f"best {gen_best_reward:7.2f} | mean {float(np.mean(mean_rewards)):7.2f} | " f"best-ever {best_ever_reward:7.2f}")

    # Rescore best performer
    key, validation_key = random.split(key)
    validation_keys = random.split(validation_key, args.validation_episodes)
    held_out_rewards = jax.vmap(evaluate_episode, in_axes=(None, 0))(best_ever_params, validation_keys)
    held_out_reward = float(jnp.mean(held_out_rewards))
    print(f"\nBest-ever search reward: {best_ever_reward:.2f}")
    print(f"Held-out team reward ({args.validation_episodes} episodes): {held_out_reward:.2f} | per-agent: {held_out_reward / NUM_AGENTS:.2f}")

    # Run one more single noisy episode to illustrate the behavior of the best performing policy
    key, test_key = random.split(key)
    rewards, food, poison, _, frames = run_episode(env, policy, best_ever_params, test_key, MAX_STEPS, args.render)
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

    generations = [row["generation"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for series, label in (("best_reward", "best"), ("mean_reward", "mean"),
                            ("best_ever_reward", "best-ever")):
        axes[0].plot(generations, [row[series] for row in history], label=label)
    axes[0].set(title="CMA-ES reward", xlabel="Generation", ylabel="team reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(generations, [row["reward_std"] for row in history])
    axes[1].set(title="Population reward spread", xlabel="Generation", ylabel="reward std")
    axes[1].grid(alpha=0.25)
    figure.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(figure)
    print(f"Metrics and figures saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
