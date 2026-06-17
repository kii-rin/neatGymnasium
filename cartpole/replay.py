import os
import pickle

import gymnasium as gym
import neat


env_name = "CartPole-v1"


def shape(output):
    return output.index(max(output))


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
env = gym.make(env_name, render_mode="human")

obs, _ = env.reset()
terminated = truncated = False
total_reward = 0.0
while not (terminated or truncated):
    obs, reward, terminated, truncated, _ = env.step(shape(net.activate(obs)))
    total_reward += reward

print("Replay total reward:", total_reward)
env.close()
