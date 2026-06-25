# Bat Flapper

A lightweight Gymnasium environment for NEAT experiments with a small two-wing flying body.

The environment now uses a minimal observation and action design:

```text
Inputs:  4
Outputs: 2
```

## Environment idea

The simulated body has:

- one forward range reading
- pitch and roll feedback
- constant base lift from the simplified flapper model
- one controllable joint on the left wing
- one controllable joint on the right wing

The policy does not see exact world position or full simulator state.

## Stages

This task combines the first two goals:

1. Stay airborne and level in simulation.
2. Move toward the forward range target.

## Controls

The NEAT network has 2 continuous outputs in `[-1, 1]`:

| Output | Meaning |
| --- | --- |
| 0 | left wing joint |
| 1 | right wing joint |

## Observations

The network has 4 inputs:

```text
range_distance
range_change
pitch
roll
```

All values are normalized and clipped to `[-1, 1]`.

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

This version keeps the controller small and realistic for experimentation: range, pitch, and roll go in; left and right wing joint commands come out.
