import argparse
import multiprocessing
import os
import pickle

import neat

from hummingbird_env import HummingbirdHoverEnv


runs_per_genome = 3


def find_latest_checkpoint(local_dir):
    checkpoints = []
    for filename in os.listdir(local_dir):
        if filename.startswith("checkpoint-"):
            try:
                checkpoints.append(
                    (int(filename.removeprefix("checkpoint-")), filename)
                )
            except ValueError:
                pass
    if not checkpoints:
        return None
    return os.path.join(local_dir, max(checkpoints)[1])


def shape(output):
    return [max(-1.0, min(1.0, value)) for value in output]


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = HummingbirdHoverEnv()
    fitnesses = []
    for run in range(runs_per_genome):
        obs, _ = env.reset(seed=genome.key * runs_per_genome + run)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = env.step(
                shape(net.activate(obs))
            )
            total += reward
        fitnesses.append(total)
    env.close()
    return min(fitnesses)


def run(generations=None, workers=None):
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

    checkpoint = find_latest_checkpoint(local_dir)
    if checkpoint:
        print("Resuming from checkpoint:", checkpoint)
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(local_dir, "checkpoint-"),
        )
    )
    worker_count = workers or max(1, multiprocessing.cpu_count() - 1)
    evaluator = neat.ParallelEvaluator(worker_count, eval_genome)
    winner = population.run(evaluator.evaluate, generations)
    with open(winner_path, "wb") as file:
        pickle.dump(winner, file)
    print("Saved", winner_path, "fitness:", winner.fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run(args.generations, args.workers)
