# Bat Glider

A lightweight Gymnasium environment for training NEAT on a bat-inspired glider.

This is intentionally simple: it is not full bat aerodynamics. It combines the first two training stages:

1. **Stage 1:** stay airborne and stable.
2. **Stage 2:** glide toward a forward waypoint.

## Controls

The NEAT network has 4 continuous outputs in `[-1, 1]`:

| Output | Meaning |
| --- | --- |
| 0 | left wing spread |
| 1 | right wing spread |
| 2 | tail pitch |
| 3 | flap assist |

## Observations

The network has 16 inputs:

```text
x, y, z,
vx, vy, vz,
pitch, roll, yaw,
pitch_rate, roll_rate, yaw_rate,
target_dx, target_dy, target_dz,
last_lift
```

All values are normalized and clipped to roughly `[-1, 1]`.

## Train

```bash
cd bat
python3 evolve-feedforward.py
```

Short smoke run:

```bash
python3 evolve-feedforward.py --generations 1
```

## Replay / record

```bash
python3 replay.py
```

Use `cli` mode on a headless server. It records:

```text
bat/videos/bat-glider-replay.mp4
```

## Why this env exists

Older flapping-wing research environments can depend on outdated simulator stacks. This env keeps the repo modern and easy to run while giving NEAT a useful flying task: survival, stability, waypoint progress, wind, and continuous control.
