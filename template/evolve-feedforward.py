import multiprocessing
import os
import pickle

import gymnasium as gym
import neat


# Change these for each environment.
env_name = "CartPole-v1"
runs_per_genome = 5
resume_path = None  # Optional manual override, e.g. "checkpoint-80"


def find_latest_checkpoint(local_dir):
    checkpoints = []
    for filename in os.listdir(local_dir):
        if not filename.startswith("checkpoint-"):
            continue
        try:
            generation = int(filename.split("checkpoint-")[1])
        except ValueError:
            continue
        checkpoints.append((generation, os.path.join(local_dir, filename)))

    if not checkpoints:
        return None

    checkpoints.sort()
    return checkpoints[-1][1]


def shape(output):
    # Discrete, N actions | num_outputs=N
    return output.index(max(output))

    # Continuous, N actions | num_outputs=N, tanh range [-1, 1]
    # return [max(-1.0, min(1.0, x)) for x in output]


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(env_name)

    fitnesses = []
    for _ in range(runs_per_genome):
        obs, _ = env.reset()
        fitness = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = shape(net.activate(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    env.close()
    return min(fitnesses)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    winner_path = os.path.join(local_dir, "winner-feedforward")

    if os.path.exists(winner_path):
        raise RuntimeError(
            f"winner-feedforward already exists at {winner_path}. "
            "Delete it if you want to train again."
        )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    checkpoint = resume_path or find_latest_checkpoint(local_dir)
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("No checkpoint found. Starting new population.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(
        neat.Checkpointer(
            generation_interval=20,
            filename_prefix=os.path.join(local_dir, "checkpoint-"),
        )
    )

    workers = max(1, multiprocessing.cpu_count() - 1)
    pe = neat.ParallelEvaluator(workers, eval_genome)
    winner = pop.run(pe.evaluate)

    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == "__main__":
    run()
