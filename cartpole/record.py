import glob
import os
import pickle

import gymnasium as gym
import neat
from gymnasium.wrappers import RecordVideo


env_name = "CartPole-v1"
video_dir = "videos"


def shape(output):
    return output.index(max(output))


def load_genome(local_dir):
    winner = os.path.join(local_dir, "winner-feedforward")
    if os.path.exists(winner):
        print("Loading winner-feedforward")
        with open(winner, "rb") as f:
            return pickle.load(f)

    files = glob.glob(os.path.join(local_dir, "checkpoint-*"))
    if not files:
        raise Exception("No winner or checkpoint found")

    checkpoint = max(files, key=lambda p: int(os.path.basename(p).split("-")[1]))
    print("Loading", checkpoint)
    pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    genomes = [g for g in pop.population.values() if g.fitness is not None]
    best = max(genomes, key=lambda g: g.fitness)
    print("Best checkpoint fitness:", best.fitness)
    return best


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    output_dir = os.path.join(local_dir, video_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    net = neat.nn.FeedForwardNetwork.create(load_genome(local_dir), config)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=output_dir, name_prefix="cartpole")

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(shape(net.activate(obs)))
        total_reward += reward

    env.close()
    print("Recorded video in:", output_dir)
    print("Replay total reward:", total_reward)


if __name__ == "__main__":
    main()
