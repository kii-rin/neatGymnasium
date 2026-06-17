import glob
import os
import pickle

import gymnasium as gym
import neat
from gymnasium.wrappers import RecordVideo


env_name = "LunarLander-v3"


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


def choose_mode():
    choice = input("Replay mode? [human/cli]: ").strip().lower()
    if choice in ("h", "human"):
        return "human"
    return "cli"


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    mode = choose_mode()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(load_genome(local_dir), config)

    if mode == "human":
        env = gym.make(env_name, render_mode="human")
    else:
        output_dir = os.path.join(local_dir, "videos")
        os.makedirs(output_dir, exist_ok=True)
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=output_dir, name_prefix="lunarlander")

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(shape(net.activate(obs)))
        total_reward += reward

    print("Replay total reward:", total_reward)
    if mode == "cli":
        print("Recorded video in:", os.path.join(local_dir, "videos"))
    env.close()


if __name__ == "__main__":
    main()
