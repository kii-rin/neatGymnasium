import os
import pickle

import gymnasium as gym
import neat


env_name = "BipedalWalker-v3"
render_mode = None
episodes = 1


def shape(output):
    return [max(-1.0, min(1.0, x)) for x in output]


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    winner_path = os.path.join(local_dir, "winner-feedforward")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make(env_name, render_mode=render_mode)

    try:
        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = shape(net.activate(obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            print(f"Episode {episode} total reward: {total_reward}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
