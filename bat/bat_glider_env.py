"""Simple Gymnasium bat-glider environment for NEAT experiments.

The goal is intentionally not full bat aerodynamics. It is a lightweight
stage-1/2 simulator: stay airborne, then glide toward a forward waypoint.
Actions are continuous controls inspired by a bat/glider body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class BatGliderConfig:
    max_steps: int = 600
    dt: float = 0.05
    gravity: float = 9.81
    mass: float = 1.0
    max_speed: float = 24.0
    target_distance: float = 35.0
    target_radius: float = 4.0
    ground_z: float = 0.0
    start_altitude: float = 12.0
    wind_randomization: float = 1.0


class BatGliderEnv(gym.Env):
    """A small continuous-control bat-like glider task.

    Observation, 16 floats in roughly [-1, 1]:
      x, y, z, vx, vy, vz, pitch, roll, yaw, pitch_rate, roll_rate, yaw_rate,
      target_dx, target_dy, target_dz, last_lift

    Action, 4 floats in [-1, 1]:
      left_wing_spread, right_wing_spread, tail_pitch, flap_assist

    The reward combines stage 1 and 2:
      - stay alive and airborne
      - keep smooth/stable attitude
      - move toward the forward waypoint
      - get a success bonus near the target
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: str | None = None, config: BatGliderConfig | None = None):
        self.render_mode = render_mode
        self.cfg = config or BatGliderConfig()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._last_lift = 0.0
        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.steps = 0
        self.pos = np.array([0.0, 0.0, self.cfg.start_altitude], dtype=np.float32)
        self.vel = np.array([6.0 + rng.uniform(-0.5, 0.5), 0.0, 0.0], dtype=np.float32)
        self.angles = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.0], dtype=np.float32)
        self.rates = np.zeros(3, dtype=np.float32)
        self.target = np.array([self.cfg.target_distance, 0.0, self.cfg.start_altitude * 0.65], dtype=np.float32)
        self.wind = rng.uniform(-self.cfg.wind_randomization, self.cfg.wind_randomization, size=3).astype(np.float32)
        self.wind[2] *= 0.2
        self._prev_distance = float(np.linalg.norm(self.target - self.pos))
        self._last_lift = 0.0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        left, right, tail, flap = action
        dt = self.cfg.dt
        self.steps += 1

        left_area = 0.55 + 0.45 * ((left + 1.0) * 0.5)
        right_area = 0.55 + 0.45 * ((right + 1.0) * 0.5)
        wing_area = left_area + right_area
        asymmetry = right_area - left_area
        air_vel = self.vel - self.wind
        speed = max(0.1, float(np.linalg.norm(air_vel)))

        pitch, roll, yaw = self.angles
        lift = 0.034 * wing_area * speed * speed * max(0.15, math.cos(abs(roll)))
        flap_lift = 0.45 * ((flap + 1.0) * 0.5) * wing_area
        drag = 0.012 * wing_area * speed * speed
        self._last_lift = lift + flap_lift

        forward = np.array([math.cos(yaw), math.sin(yaw), math.sin(pitch)], dtype=np.float32)
        forward /= max(1e-6, np.linalg.norm(forward))
        drag_vec = -drag * air_vel / speed
        force = np.array([0.0, 0.0, lift + flap_lift - self.cfg.gravity * self.cfg.mass], dtype=np.float32)
        force += drag_vec.astype(np.float32)
        force += 0.09 * speed * forward

        self.vel += (force / self.cfg.mass) * dt
        self.vel = np.clip(self.vel, -self.cfg.max_speed, self.cfg.max_speed)
        self.pos += self.vel * dt

        # Attitude dynamics: wing asymmetry rolls/yaws, tail controls pitch.
        self.rates[0] += (0.9 * tail - 0.35 * self.rates[0] - 0.08 * pitch) * dt
        self.rates[1] += (1.2 * asymmetry - 0.45 * self.rates[1] - 0.12 * roll) * dt
        self.rates[2] += (0.45 * asymmetry + 0.18 * roll - 0.35 * self.rates[2]) * dt
        self.angles += self.rates * dt
        self.angles = np.clip(self.angles, [-0.9, -1.1, -math.pi], [0.9, 1.1, math.pi]).astype(np.float32)

        distance = float(np.linalg.norm(self.target - self.pos))
        progress = self._prev_distance - distance
        self._prev_distance = distance

        stable = 1.0 - min(1.0, (abs(self.angles[0]) + abs(self.angles[1])) / 1.4)
        alive = self.pos[2] > self.cfg.ground_z and abs(self.angles[1]) < 1.05
        reached = distance < self.cfg.target_radius
        crashed = not alive
        truncated = self.steps >= self.cfg.max_steps
        terminated = bool(reached or crashed)

        reward = 0.03                    # stage 1: stay alive
        reward += 1.5 * progress          # stage 2: move toward waypoint
        reward += 0.03 * stable
        reward -= 0.002 * float(np.linalg.norm(self.rates))
        reward -= 0.001 * abs(float(action[3]))
        if reached:
            reward += 25.0
        if crashed:
            reward -= 15.0

        return self._obs(), float(reward), terminated, truncated, {"distance": distance, "reached": reached}

    def _obs(self):
        delta = self.target - self.pos
        vals = np.array([
            self.pos[0] / self.cfg.target_distance,
            self.pos[1] / 20.0,
            self.pos[2] / 20.0,
            self.vel[0] / self.cfg.max_speed,
            self.vel[1] / self.cfg.max_speed,
            self.vel[2] / self.cfg.max_speed,
            self.angles[0] / 0.9,
            self.angles[1] / 1.1,
            self.angles[2] / math.pi,
            self.rates[0] / 4.0,
            self.rates[1] / 4.0,
            self.rates[2] / 4.0,
            delta[0] / self.cfg.target_distance,
            delta[1] / 20.0,
            delta[2] / 20.0,
            self._last_lift / 18.0,
        ], dtype=np.float32)
        return np.clip(vals, -1.0, 1.0)

    def render(self):
        width, height = 640, 360
        img = np.full((height, width, 3), 245, dtype=np.uint8)
        ground = int(height * 0.82)
        img[ground:, :, :] = np.array([210, 230, 210], dtype=np.uint8)

        def world_to_px(p):
            x = int(60 + (p[0] / max(1.0, self.cfg.target_distance + 10.0)) * (width - 120))
            y = int(ground - p[2] * 11)
            return np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)

        tx, ty = world_to_px(self.target)
        bx, by = world_to_px(self.pos)
        img[max(0, ty - 6):min(height, ty + 7), max(0, tx - 6):min(width, tx + 7)] = [50, 180, 80]
        img[max(0, by - 4):min(height, by + 5), max(0, bx - 10):min(width, bx + 11)] = [40, 40, 40]
        img[max(0, by - 2):min(height, by + 3), max(0, bx - 25):min(width, bx + 26)] = [90, 70, 120]
        return img

    def close(self):
        pass


if __name__ == "__main__":
    env = BatGliderEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(120):
        obs, reward, terminated, truncated, info = env.step([0.7, 0.7, 0.05, -0.2])
        total += reward
        if terminated or truncated:
            break
    print(f"steps={env.steps} total_reward={total:.2f} info={info}")
