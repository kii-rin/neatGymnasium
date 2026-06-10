# neatGymnasium

A small Python project for training NEAT neural networks on [Gymnasium](https://gymnasium.farama.org/) environments.

The repository currently includes examples for:

- `CartPole-v1`
- `LunarLander-v3`

Each example trains a feed-forward neural network policy using `neat-python`, evaluates candidate genomes in a Gymnasium environment, and saves the best genome as `winner-feedforward`.

## What this project does

`neatGymnasium` uses NeuroEvolution of Augmenting Topologies (NEAT) to evolve neural networks that act as policies for reinforcement-learning environments.

At a high level, each training script:

1. Loads a NEAT configuration from `config-feedforward`.
2. Creates a Gymnasium environment.
3. Builds a feed-forward neural network for each genome.
4. Runs the genome in the environment multiple times.
5. Uses the minimum score across runs as the genome fitness.
6. Evolves the population until a winner is found or NEAT stops.
7. Saves the winning genome to `winner-feedforward`.

The replay scripts load the saved winner, rebuild the network, and run it in the environment with rendering enabled.

## Repository layout

The repository is organized around individual Gymnasium tasks. The initial examples include CartPole and LunarLander configurations and scripts.

A typical environment folder contains:

```text
config-feedforward   # NEAT configuration for the environment
run.py               # Trains a population and saves the winning genome
show.py              # Loads and renders the saved winning genome
```

## Requirements

This project is written in Python and depends on:

- `gymnasium`
- `neat-python`
- `box2d-py` or Gymnasium Box2D extras for LunarLander

For LunarLander, install Gymnasium with Box2D support:

```bash
pip install "gymnasium[box2d]" neat-python
```

For CartPole only, this is usually enough:

```bash
pip install gymnasium neat-python
```

## Usage

Clone the repository:

```bash
git clone https://github.com/kii-rin/neatGymnasium.git
cd neatGymnasium
```

Install dependencies:

```bash
pip install gymnasium neat-python
```

If you want to run LunarLander, install Box2D support as well:

```bash
pip install "gymnasium[box2d]"
```

Run a training script from the relevant environment directory:

```bash
python run.py
```

After training completes, the script writes the best genome to:

```text
winner-feedforward
```

To replay a trained winner with rendering enabled:

```bash
python show.py
```

## Configuration notes

The NEAT configuration controls the number of inputs, outputs, population size, mutation rates, activation function, and fitness threshold.

Current environment settings include:

| Environment | Inputs | Outputs | Population size | Fitness threshold |
| --- | ---: | ---: | ---: | ---: |
| `CartPole-v1` | 4 | 2 | 200 | 475.0 |
| `LunarLander-v3` | 8 | 4 | 300 | 200.0 |

The output layer is interpreted as a discrete action selector:

```python
return output.index(max(output))
```

This means the network output with the highest activation becomes the selected Gymnasium action.

## Important setup note

Some replay scripts may contain absolute local paths for the NEAT config and winner files. If you run into file path errors, update those paths to point to your local `config-feedforward` and `winner-feedforward` files, or replace them with paths relative to the script directory.

For example:

```python
import os

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-feedforward")
winner_path = os.path.join(local_dir, "winner-feedforward")
```

## Ideas for future improvements

- Add a `requirements.txt` or `pyproject.toml` file.
- Replace absolute replay paths with relative paths.
- Add per-environment folders with consistent names.
- Add command-line options for choosing environments and run counts.
- Save training statistics and plots.
- Add CI checks for formatting and import errors.

## License

No license file is currently included. Add a license before distributing or reusing the project publicly.
