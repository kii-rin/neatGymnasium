import argparse
import glob
import os
import pickle

import neat
from gymnasium.wrappers import RecordVideo

from hummingbird_env import HummingbirdHoverEnv


def shape(output):
    return [max(-1.0, min(1.0, value)) for value in output]


def load_genome(local_dir):
    winner_path = os.path.join(local_dir, "winner-feedforward")
    if os.path.exists(winner_path):
        with open(winner_path, "rb") as file:
            return pickle.load(file)

    checkpoints = glob.glob(os.path.join(local_dir, "checkpoint-*"))
    if not checkpoints:
        raise RuntimeError("No winner or checkpoint found. Train the environment first.")
    latest = max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[1]))
    population = neat.Checkpointer.restore_checkpoint(latest)
    genomes = [g for g in population.population.values() if g.fitness is not None]
    if not genomes:
        raise RuntimeError("The latest checkpoint has no evaluated genomes.")
    return max(genomes, key=lambda genome: genome.fitness)


def main(seed):
    local_dir = os.path.dirname(__file__)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(local_dir, "config-feedforward"),
    )
    net = neat.nn.FeedForwardNetwork.create(load_genome(local_dir), config)
    video_dir = os.path.join(local_dir, "videos")
    env = RecordVideo(
        HummingbirdHoverEnv(render_mode="rgb_array"),
        video_folder=video_dir,
        name_prefix="hummingbird-hover",
        episode_trigger=lambda _: True,
    )

    obs, _ = env.reset(seed=seed)
    total = 0.0
    terminated = truncated = False
    last_info = {}
    while not (terminated or truncated):
        obs, reward, terminated, truncated, last_info = env.step(
            shape(net.activate(obs))
        )
        total += reward
    env.close()
    print("Replay total reward:", total)
    print("Final position error:", last_info.get("position_error"))
    print("Recorded video in:", video_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    main(parser.parse_args().seed)
