import pickle
import gymnasium as gym
import neat


env_name = "CartPole-v1"
config_path = "/Users/kiirin/Desktop/neat/cartpole/config-feedforward"
winner_path = "/Users/kiirin/Desktop/neat/cartpole/winner-feedforward"


def shape(output):
    return output.index(max(output))


config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path,
)

with open(winner_path, 'rb') as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)
env = gym.make(env_name, render_mode="human")

obs, _ = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    obs, _, terminated, truncated, _ = env.step(shape(net.activate(obs)))

env.close()