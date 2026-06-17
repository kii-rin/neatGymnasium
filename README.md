# neatGymnasium

A small Python project for training NEAT neural networks on [Gymnasium](https://gymnasium.farama.org/) environments.

The repository currently includes examples for:

- `CartPole-v1`
- `LunarLander-v3`
- `BipedalWalker-v3`

Each environment trains a feed-forward neural network policy using `neat-python`, evaluates candidate genomes in a Gymnasium environment, and saves the best final genome as `winner-feedforward`.

## What this project does

`neatGymnasium` uses NeuroEvolution of Augmenting Topologies (NEAT) to evolve neural networks that act as policies for reinforcement-learning environments.

At a high level, each training script:

1. Loads a NEAT configuration from `config-feedforward`.
2. Creates a Gymnasium environment.
3. Builds a feed-forward neural network for each genome.
4. Runs the genome in the environment multiple times.
5. Uses the minimum score across runs as the genome fitness.
6. Evolves the population until a winner is found or NEAT stops.
7. Saves checkpoints during training.
8. Saves the winning genome to `winner-feedforward` when a winner is found.

Training scripts automatically resume from the highest numbered `checkpoint-*` file when no winner exists.

## Repository layout

Each environment folder contains:

```text
config-feedforward       # NEAT configuration for the environment
evolve-feedforward.py    # Trains or resumes a population
replay.py                # Replays winner/checkpoint, with GUI or CLI video mode
```

Generated files are ignored by Git:

```text
checkpoint-*
winner-feedforward
videos/
*.mp4
```

## Requirements

Install the project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `gymnasium[box2d]`
- `neat-python`
- `numpy`
- `moviepy` for Gymnasium video recording

On Ubuntu/Vast.ai, Box2D environments may also need system packages:

```bash
apt-get update
apt-get install -y swig build-essential python3-dev ffmpeg
```

## Usage

Clone the repository:

```bash
git clone https://github.com/kii-rin/neatGymnasium.git
cd neatGymnasium
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training from an environment folder:

```bash
cd biwalker
python3 evolve-feedforward.py
```

If checkpoints exist, training resumes automatically from the highest checkpoint:

```text
checkpoint-180
checkpoint-181
```

Running training again will continue from `checkpoint-181`.

## Replay and recording

Use `replay.py` for both GUI replay and CLI video recording:

```bash
python3 replay.py
```

It asks:

```text
Replay mode? [human/cli]:
```

Choose `human` on a local machine with a display. This uses Gymnasium `render_mode="human"`.

Choose `cli` on a headless server such as Vast.ai. This prints the reward and records an `.mp4` into:

```text
videos/
```

`replay.py` loads `winner-feedforward` first. If no winner exists, it loads the highest numbered `checkpoint-*` and replays the best genome from that checkpoint.

Example on Vast.ai:

```bash
cd /root/neatGymnasium/biwalker
python3 replay.py
# choose cli
ls -lh videos/
```

Download videos with `scp` from your local machine:

```bash
scp -P YOUR_PORT 'root@YOUR_HOST:/root/neatGymnasium/biwalker/videos/*.mp4' .
```

## Configuration notes

The NEAT configuration controls inputs, outputs, population size, mutation rates, activation function, speciation behavior, and fitness threshold.

Current environment settings include:

| Environment | Inputs | Outputs | Action type | Population size | Fitness threshold |
| --- | ---: | ---: | --- | ---: | ---: |
| `CartPole-v1` | 4 | 2 | Discrete | 100 | 475.0 |
| `LunarLander-v3` | 8 | 4 | Discrete | 300 | 200.0 |
| `BipedalWalker-v3` | 24 | 4 | Continuous | 300 | 300.0 |

For discrete action environments, the highest output activation is selected:

```python
return output.index(max(output))
```

For BipedalWalker, network outputs are clipped into the valid continuous action range:

```python
return [max(-1.0, min(1.0, x)) for x in output]
```

## Vast.ai notes

Training is CPU-focused. Use `tmux` to keep long runs alive after SSH disconnects:

```bash
tmux new -s biwalker
python3 evolve-feedforward.py
```

Detach without stopping training:

```text
Ctrl+b, then d
```

Reconnect later:

```bash
tmux attach -t biwalker
```

Plain SSH/tmux does not show GUI windows. Use `cli` replay mode to record videos instead.

## Ideas for future improvements

Already done:

- Added `requirements.txt`.
- Replaced absolute replay paths with relative paths.
- Added BipedalWalker environment.
- Added checkpoint auto-resume for training.
- Added replay fallback from latest checkpoint when no winner exists.
- Combined replay and video recording into `replay.py`.

Still useful:

- Add command-line flags instead of interactive replay prompts.
- Add training statistics export and plots.
- Add a PPO or SAC baseline for comparison against NEAT.
- Add CI checks for formatting and import errors.
- Add a license file.

## License

No license file is currently included. Add a license before distributing or reusing the project publicly.
