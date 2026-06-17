import argparse
import os
import pickle

import gymnasium as gym
import neat


env_name = "BipedalWalker-v3"


def shape(output):
    return [max(-1.0, min(1.0, x)) for x in output]


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a trained BipedalWalker winner.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--render-mode",
        choices=("none", "human", "rgb_array"),
        default="none",
        help="Use 'none' on headless servers. Use 'human' only with a display.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    winner_path = os.path.join(local_dir, "winner-feedforward")

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path,
    )

    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = gym.make(env_name, render_mode=render_mode)

    try:
        for episode in range(1, args.episodes + 1):
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
