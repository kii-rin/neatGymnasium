# Hummingbird hover

This is a lightweight Gymnasium environment inspired by Purdue Flappy's
high-level control interface. It is a new implementation and does not depend
on the legacy DART, pydart2, TensorFlow 1, or classic Gym stack.

The network receives 18 values: a 3x3 rotation matrix, target-relative XYZ,
linear velocity, and angular velocity. It outputs normalized thrust, roll,
pitch, and yaw commands.

```bash
cd flappy
python3 smoke_test.py
python3 evolve-feedforward.py
python3 replay.py
```

For a quick training check:

```bash
python3 evolve-feedforward.py --generations 1 --workers 1
```

Replay records `videos/hummingbird-hover-episode-0.mp4`. Training includes
random initial attitude, position, velocity, and steady wind. The physics are
intentionally small and understandable; they are suitable for learning and
experimentation, not hardware validation.
