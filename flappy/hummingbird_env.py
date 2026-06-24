import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HummingbirdHoverEnv(gym.Env):
    """Small flapping-wing-inspired hover task with Gymnasium's modern API."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None, max_episode_steps=250, wind_strength=0.15):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.wind_strength = wind_strength
        self.dt = 0.04
        self.target = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # thrust, roll torque, pitch torque, yaw torque
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # rotation matrix (9), target-relative position (3), velocity (3), rates (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        self.mass = 0.03
        self.gravity = 9.81
        self.max_torque = np.array([0.003, 0.003, 0.0015])
        self.inertia = np.array([0.00018, 0.00022, 0.00030])
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.wind = np.zeros(3)
        self.previous_action = np.zeros(4)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        randomize = options.get("randomize", True)

        self.position = self.target.copy()
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        if randomize:
            self.position += self.np_random.uniform(-0.12, 0.12, 3)
            self.angles = self.np_random.uniform(-0.12, 0.12, 3)
            self.velocity = self.np_random.uniform(-0.08, 0.08, 3)
            self.wind = self.np_random.uniform(
                -self.wind_strength, self.wind_strength, 3
            )
            self.wind[2] *= 0.25
        else:
            self.wind = np.zeros(3)

        self.previous_action = np.zeros(4)
        self.steps = 0
        return self._observation(), {"wind": self.wind.copy()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.steps += 1

        rotation = self._rotation_matrix()
        hover_thrust = self.mass * self.gravity
        thrust = hover_thrust * (1.0 + action[0])
        acceleration = rotation[:, 2] * thrust / self.mass
        acceleration += np.array([0.0, 0.0, -self.gravity])
        acceleration += self.wind - 0.35 * self.velocity

        torque = action[1:] * self.max_torque
        angular_acceleration = torque / self.inertia - 1.8 * self.angular_velocity

        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        self.angular_velocity += angular_acceleration * self.dt
        self.angles += self.angular_velocity * self.dt
        self.angles[2] = (self.angles[2] + math.pi) % (2 * math.pi) - math.pi

        position_error = np.linalg.norm(self.position - self.target)
        tilt = np.linalg.norm(self.angles[:2])
        speed = np.linalg.norm(self.velocity)
        spin = np.linalg.norm(self.angular_velocity)
        action_change = np.linalg.norm(action - self.previous_action)
        self.previous_action = action.copy()

        reward = 2.0
        reward -= 1.8 * position_error
        reward -= 0.35 * tilt + 0.12 * speed + 0.04 * spin
        reward -= 0.015 * np.linalg.norm(action[1:]) ** 2
        reward -= 0.01 * action_change

        out_of_bounds = position_error > 3.0
        overturned = np.any(np.abs(self.angles[:2]) > math.radians(80))
        below_ground = self.position[2] < 0.0
        terminated = bool(out_of_bounds or overturned or below_ground)
        truncated = self.steps >= self.max_episode_steps
        if terminated:
            reward -= 20.0

        info = {
            "position_error": float(position_error),
            "tilt": float(tilt),
            "is_success": bool(position_error < 0.15 and tilt < 0.15),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self):
        obs = np.concatenate(
            (
                self._rotation_matrix().reshape(-1),
                self.position - self.target,
                self.velocity,
                self.angular_velocity,
            )
        )
        return obs.astype(np.float32)

    def _rotation_matrix(self):
        roll, pitch, yaw = self.angles
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        frame = np.full((512, 768, 3), 245, dtype=np.uint8)
        frame[420:424, :, :] = np.array([85, 110, 80], dtype=np.uint8)
        scale = 120
        center = np.array([384, 420])
        target_px = center + np.array([self.target[0], -self.target[2]]) * scale
        bird_px = center + np.array([self.position[0], -self.position[2]]) * scale
        self._disc(frame, target_px.astype(int), 9, (70, 150, 90))
        self._disc(frame, bird_px.astype(int), 10, (35, 55, 70))

        wing = 34
        roll = self.angles[0]
        left = bird_px + np.array([-wing * math.cos(roll), wing * math.sin(roll)])
        right = bird_px + np.array([wing * math.cos(roll), -wing * math.sin(roll)])
        self._line(frame, left.astype(int), bird_px.astype(int), (30, 120, 170))
        self._line(frame, bird_px.astype(int), right.astype(int), (30, 120, 170))
        return frame

    @staticmethod
    def _disc(frame, center, radius, color):
        y, x = np.ogrid[: frame.shape[0], : frame.shape[1]]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        frame[mask] = color

    @staticmethod
    def _line(frame, start, end, color):
        count = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        xs = np.linspace(start[0], end[0], count + 1).astype(int)
        ys = np.linspace(start[1], end[1], count + 1).astype(int)
        valid = (
            (xs >= 0)
            & (xs < frame.shape[1])
            & (ys >= 0)
            & (ys < frame.shape[0])
        )
        frame[ys[valid], xs[valid]] = color
