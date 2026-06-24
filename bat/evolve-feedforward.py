"""Train NEAT on the BatGliderEnv.

Usage:
    python3 evolve-feedforward.py
    python3 evolve-feedforward.py --generations 5
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import neat
import numpy as np

from bat_glider_env import BatGliderEnv


CHECKPOINT_PREFIX = "checkpoint-"
WINNER_FILE = "winner-feedforward"


def newest_checkpoint() -> str | None:
    checkpoints = glob.glob(f"{CHECKPOINT_PREFIX}*")
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.split(CHECKPOINT_PREFIX)[-1]))


def scale_action(outputs):
    values = np.asarray(outputs, dtype=np.float32)
    return np.clip(values, -1.0, 1.0)


def run_episode(net, seed=None, max_steps=600):
    env = BatGliderEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    reached = False
    for _ in range(max_steps):
        action = scale_action(net.activate(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        reached = reached or bool(info.get("reached", False))
        if terminated or truncated:
            break
    env.close()
    if reached:
        total += 10.0
    return total


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitnesses = [run_episode(net, seed=genome_id + offset * 10000) for offset in range(3)]
        genome.fitness = float(np.mean(fitnesses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoints and start a new population")
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__) or "."
    config_path = os.path.join(local_dir, "config-feedforward")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    checkpoint = None if args.fresh else newest_checkpoint()
    if os.path.exists(WINNER_FILE) and not args.fresh:
        print(f"{WINNER_FILE} already exists. Delete it or use --fresh to train again.")
        return
    if checkpoint:
        print(f"Resuming from {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(10, filename_prefix=CHECKPOINT_PREFIX))

    winner = population.run(eval_genomes, args.generations)
    with open(WINNER_FILE, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved {WINNER_FILE}")


if __name__ == "__main__":
    main()
