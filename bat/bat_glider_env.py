"""Simple Gymnasium bat-flapper environment with visible wing animation."""
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
    max_speed: float = 18.0
    corridor_length: float = 40.0
    safe_sonar_distance: float = 8.0
    min_sonar_distance: float = 1.2
    ground_z: float = 0.0
    ceiling_z: float = 16.0
    start_altitude: float = 8.0
    wind_randomization: float = 0.35

class BatGliderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    def __init__(self, render_mode: str | None = None, config: BatGliderConfig | None = None):
        self.render_mode = render_mode
        self.cfg = config or BatGliderConfig()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.steps = 0
        self.pos = np.array([0.0, rng.uniform(-0.4, 0.4), self.cfg.start_altitude], dtype=np.float32)
        self.vel = np.array([4.8 + rng.uniform(-0.25, 0.25), 0.0, rng.uniform(-0.1, 0.1)], dtype=np.float32)
        self.angles = np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), 0.0], dtype=np.float32)
        self.rates = np.zeros(3, dtype=np.float32)
        self.wind = rng.uniform(-self.cfg.wind_randomization, self.cfg.wind_randomization, size=3).astype(np.float32)
        self.wind[2] *= 0.25
        self._prev_sonar = self._sonar_distance()
        self._last_left_open = 0.625
        self._last_right_open = 0.625
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        left_joint, right_joint = action
        dt = self.cfg.dt
        self.steps += 1
        left_open = 0.25 + 0.75 * ((left_joint + 1.0) * 0.5)
        right_open = 0.25 + 0.75 * ((right_joint + 1.0) * 0.5)
        self._last_left_open = float(left_open)
        self._last_right_open = float(right_open)
        wing_open = left_open + right_open
        asymmetry = right_open - left_open
        pitch, roll, yaw = self.angles
        air_vel = self.vel - self.wind
        speed = max(0.1, float(np.linalg.norm(air_vel)))
        forward = np.array([math.cos(yaw), math.sin(yaw), math.sin(pitch)], dtype=np.float32)
        forward /= max(1e-6, np.linalg.norm(forward))
        stability_factor = max(0.2, math.cos(abs(pitch)) * math.cos(abs(roll)))
        base_lift = 4.9 * wing_open * stability_factor
        glide_lift = 0.018 * wing_open * speed * speed * stability_factor
        drag = 0.022 * wing_open * speed * speed
        drive = 0.42 * wing_open * stability_factor
        force = np.array([0.0, 0.0, base_lift + glide_lift - self.cfg.gravity * self.cfg.mass], dtype=np.float32)
        force += (-drag * air_vel / speed).astype(np.float32)
        force += drive * forward
        self.vel += (force / self.cfg.mass) * dt
        self.vel = np.clip(self.vel, -self.cfg.max_speed, self.cfg.max_speed)
        self.pos += self.vel * dt
        average_open = 0.5 * wing_open
        self.rates[0] += (0.18 * (average_open - 1.15) - 0.35 * self.rates[0] - 0.10 * pitch) * dt
        self.rates[1] += (1.45 * asymmetry - 0.55 * self.rates[1] - 0.18 * roll) * dt
        self.rates[2] += (0.35 * asymmetry + 0.16 * roll - 0.45 * self.rates[2]) * dt
        self.angles += self.rates * dt
        self.angles = np.clip(self.angles, [-0.9, -1.05, -math.pi], [0.9, 1.05, math.pi]).astype(np.float32)
        sonar = self._sonar_distance()
        sonar_change = self._prev_sonar - sonar
        self._prev_sonar = sonar
        stable = 1.0 - min(1.0, (abs(self.angles[0]) + abs(self.angles[1])) / 1.25)
        altitude_ok = self.cfg.ground_z < self.pos[2] < self.cfg.ceiling_z
        attitude_ok = abs(self.angles[0]) < 0.85 and abs(self.angles[1]) < 0.95
        too_close = sonar < self.cfg.min_sonar_distance
        reached = sonar < self.cfg.safe_sonar_distance and altitude_ok and attitude_ok
        crashed = not altitude_ok or not attitude_ok or too_close
        truncated = self.steps >= self.cfg.max_steps
        terminated = bool(reached or crashed)
        reward = 0.035 + 0.04 * stable + 0.15 * float(sonar_change)
        reward -= 0.0015 * float(np.linalg.norm(self.rates))
        reward -= 0.001 * abs(float(asymmetry))
        if reached: reward += 20.0
        if crashed: reward -= 15.0
        return self._obs(), float(reward), terminated, truncated, {
            "sonar_distance": float(sonar),
            "sonar_change": float(sonar_change),
            "reached": bool(reached),
            "crashed": bool(crashed),
            "left_open": float(left_open),
            "right_open": float(right_open),
        }

    def _sonar_distance(self) -> float:
        return max(0.0, self.cfg.corridor_length - float(self.pos[0]))

    def _obs(self):
        sonar = self._sonar_distance()
        sonar_norm = (sonar / self.cfg.corridor_length) * 2.0 - 1.0
        sonar_change = np.clip((self._prev_sonar - sonar) / 1.2, -1.0, 1.0)
        return np.clip(np.array([sonar_norm, sonar_change, self.angles[0] / 0.9, self.angles[1] / 1.05], dtype=np.float32), -1.0, 1.0)

    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, color, thickness=2):
        h, w = img.shape[:2]
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        dx, dy = abs(x1-x0), -abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for yy in range(y0-thickness, y0+thickness+1):
                for xx in range(x0-thickness, x0+thickness+1):
                    if 0 <= xx < w and 0 <= yy < h:
                        img[yy, xx] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def render(self):
        width, height = 640, 360
        img = np.full((height, width, 3), 245, dtype=np.uint8)
        ground = int(height * 0.82)
        img[ground:, :, :] = np.array([210, 230, 210], dtype=np.uint8)
        def world_to_px(p):
            x = int(50 + (p[0] / max(1.0, self.cfg.corridor_length)) * (width - 100))
            y = int(ground - p[2] * 13)
            return np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
        wall_x, _ = world_to_px([self.cfg.corridor_length, 0.0, 0.0])
        img[:, max(0, wall_x - 2):min(width, wall_x + 3)] = [120, 190, 120]
        bx, by = world_to_px(self.pos)
        img[max(0, by - 5):min(height, by + 6), max(0, bx - 8):min(width, bx + 9)] = [35, 35, 35]
        flap = math.sin(self.steps * 0.65) * 0.85
        base_span = 44
        left_len = base_span * self._last_left_open
        right_len = base_span * self._last_right_open
        left_tip = (bx - left_len, by - 16 * flap)
        right_tip = (bx + right_len, by + 16 * flap)
        self._draw_line(img, bx, by, left_tip[0], left_tip[1], [90, 70, 160], thickness=3)
        self._draw_line(img, bx, by, right_tip[0], right_tip[1], [90, 70, 160], thickness=3)
        self._draw_line(img, bx - base_span, by + 24, bx + base_span, by + 24, [185, 175, 205], thickness=1)
        self._draw_line(img, bx + 10, by, wall_x, by, [220, 180, 60], thickness=1)
        return img

    def close(self):
        pass

if __name__ == "__main__":
    env = BatGliderEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(120):
        obs, reward, terminated, truncated, info = env.step([0.4, 0.4])
        total += reward
        if terminated or truncated:
            break
    print(f"steps={env.steps} total_reward={total:.2f} info={info}")
