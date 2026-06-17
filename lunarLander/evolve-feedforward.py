import multiprocessing
import os
import pickle
import gymnasium as gym
import neat


env_name = "LunarLander-v3"
runs_per_genome = 5
resume_path = None  # Optional manual override, e.g. "checkpoint-100"


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
    # discrete, 2 actions      | num_outputs=1, sigmoid
    # return 0 if output[0] < 0.5 else 1

    # discrete, N actions      | num_outputs=N, sigmoid or tanh
    return output.index(max(output))

    # continuous, scaled       | num_outputs=1, tanh
    # return [output[0] * 2.0]


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(env_name)

    fitnesses = []
    for _ in range(runs_per_genome):
        obs, _ = env.reset()
        fitness = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            output = net.activate(obs)
            action = shape(output)
            obs, reward, terminated, truncated, _ = env.step(action)
            fitness += reward
        fitnesses.append(fitness)

    env.close()
    return min(fitnesses)


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    winner_path = os.path.join(local_dir, 'winner-feedforward')

    if os.path.exists(winner_path):
        raise RuntimeError(
            f"winner-feedforward already exists at {winner_path}. "
            "Delete it if you want to train again."
        )

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path,
    )

    latest_checkpoint = resume_path or find_latest_checkpoint(local_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print("No checkpoint found. Starting new population.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(
        generation_interval=50,
        filename_prefix=os.path.join(local_dir, 'checkpoint-'),
    ))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == '__main__': run()
