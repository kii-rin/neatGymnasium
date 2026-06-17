import glob
import os
import pickle

import gymnasium as gym
import neat


env_name = "BipedalWalker-v3"
render_mode = None


def shape(output):
    return [max(-1.0, min(1.0, x)) for x in output]


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
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(load_genome(local_dir), config)
    env = gym.make(env_name, render_mode=render_mode)

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(shape(net.activate(obs)))
        total_reward += reward

    print("Replay total reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()
