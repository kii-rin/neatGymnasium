# neatGymnasium

A small Python project for training NEAT neural networks on Gymnasium environments.

Included environments:

- `CartPole-v1`
- `LunarLander-v3`
- `BipedalWalker-v3`
- `BatGliderEnv` in `bat/`

Each environment trains a feed-forward neural network policy with `neat-python`, saves checkpoints during training, and saves a final `winner-feedforward` when NEAT reaches the configured fitness threshold.

## Layout

Each environment folder contains:

```text
config-feedforward       # NEAT config for that environment
evolve-feedforward.py    # train or auto-resume from latest checkpoint
replay.py                # replay winner/checkpoint; GUI or CLI video mode
```

The `template/` folder contains updated starter scripts for new environments.

Ignored generated files:

```text
checkpoint-*
winner-feedforward
videos/
*.mp4
```

## Install

```bash
pip install -r requirements.txt
```

On Ubuntu/Vast.ai, Box2D environments may also need:

```bash
apt-get update
apt-get install -y swig build-essential python3-dev ffmpeg
```

## Train

```bash
cd biwalker
python3 evolve-feedforward.py
```

Training automatically resumes from the highest numbered `checkpoint-*` when no `winner-feedforward` exists.

### Train the bat environment

```bash
cd bat
python3 evolve-feedforward.py
```

For a quick smoke run:

```bash
python3 evolve-feedforward.py --generations 1
```

The bat environment uses 4 normalized inputs and 2 continuous outputs:

```text
Inputs:  range_distance, range_change, pitch, roll
Outputs: left_wing_joint, right_wing_joint
```

## Replay or record

```bash
python3 replay.py
```

It asks:

```text
Replay mode? [human/cli]:
```

Use `human` on a local machine with a display. Use `cli` on Vast/headless servers. CLI mode prints reward and records an `.mp4` in `videos/`.

`replay.py` loads `winner-feedforward` first. If no winner exists, it loads the highest checkpoint and replays the best genome from that checkpoint.

Download videos from your local machine:

```bash
scp -P YOUR_PORT 'root@YOUR_HOST:/root/neatGymnasium/biwalker/videos/*.mp4' .
scp -P YOUR_PORT 'root@YOUR_HOST:/root/neatGymnasium/bat/videos/*.mp4' .
```

## Current config summary

| Environment | Inputs | Outputs | Action type | Population | Fitness threshold |
| --- | ---: | ---: | --- | ---: | ---: |
| `CartPole-v1` | 4 | 2 | Discrete | 100 | 475.0 |
| `LunarLander-v3` | 8 | 4 | Discrete | 300 | 200.0 |
| `BipedalWalker-v3` | 24 | 4 | Continuous | 300 | 300.0 |
| `BatGliderEnv` | 4 | 2 | Continuous | 300 | 45.0 |

CartPole stays at 100 because it is small and discrete. LunarLander, BipedalWalker, and BatGlider use 300 to keep more diversity in harder spaces.

BipedalWalker currently uses:

```text
pop_size = 300
compatibility_threshold = 2.0
max_stagnation = 25
species_elitism = 2
```

BatGlider currently uses:

```text
pop_size = 300
compatibility_threshold = 2.0
max_stagnation = 25
species_elitism = 2
```

The lower compatibility thresholds are meant to avoid collapsing into one species too quickly.

Discrete actions use the index of the highest output:

```python
return output.index(max(output))
```

Continuous action environments clip outputs to `[-1, 1]`:

```python
return [max(-1.0, min(1.0, x)) for x in output]
```

## Vast.ai notes

Use `tmux` for long training:

```bash
tmux new -s biwalker
python3 evolve-feedforward.py
```

For bat training:

```bash
tmux new -s bat
cd bat
python3 evolve-feedforward.py
```

Detach with `Ctrl+b`, then `d`. Reattach with:

```bash
tmux attach -t biwalker
tmux attach -t bat
```

Plain SSH/tmux does not show GUI windows. Use `cli` replay mode to record videos instead.

## Done

- Added `requirements.txt`.
- Replaced absolute replay paths with relative paths.
- Added BipedalWalker.
- Added checkpoint auto-resume for training.
- Added replay fallback from latest checkpoint.
- Combined replay and recording in `replay.py`.
- Removed duplicate `record.py` scripts.
- Refreshed `template/` scripts.
- Retuned BipedalWalker config from 100 population / 3.0 threshold to 300 population / 2.0 threshold.
- Added a minimal bat environment with 4 inputs and 2 continuous outputs.

## TODO

- Add command-line flags instead of interactive replay prompts.
- Add training statistics export and plots.
- Add a PPO or SAC baseline for comparison against NEAT.
- Add CI checks for formatting and import errors.
- Add a license file.
- Add later bat stages after the minimal environment is working.

## License

No license file is currently included.
