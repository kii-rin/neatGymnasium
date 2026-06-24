"""Replay a trained bat glider NEAT policy and optionally record video."""

from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import neat

from bat_glider_env import BatGliderEnv


WINNER_FILE = "winner-feedforward"
CHECKPOINT_PREFIX = "checkpoint-"


def newest_checkpoint():
    checkpoints = glob.glob(f"{CHECKPOINT_PREFIX}*")
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.split(CHECKPOINT_PREFIX)[-1]))


def load_genome(config):
    if os.path.exists(WINNER_FILE):
        with open(WINNER_FILE, "rb") as f:
            return pickle.load(f), "winner-feedforward"

    checkpoint = newest_checkpoint()
    if not checkpoint:
        raise FileNotFoundError("No winner-feedforward or checkpoint-* found. Train first.")
    population = neat.Checkpointer.restore_checkpoint(checkpoint)
    best = max(population.population.values(), key=lambda g: g.fitness or float("-inf"))
    return best, checkpoint


def save_video(frames, path="videos/bat-glider-replay.mp4", fps=20):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps)
    except Exception:
        from moviepy.editor import ImageSequenceClip
        ImageSequenceClip(frames, fps=fps).write_videofile(path, codec="libx264", audio=False)
    return path


def main():
    mode = input("Replay mode? [human/cli]: ").strip().lower() or "cli"
    local_dir = os.path.dirname(__file__) or "."
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(local_dir, "config-feedforward"),
    )
    genome, source = load_genome(config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = BatGliderEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    frames = []
    total = 0.0

    for _ in range(env.cfg.max_steps):
        action = net.activate(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        frame = env.render()
        if mode == "cli":
            frames.append(frame)
        if terminated or truncated:
            break

    print(f"Loaded {source}")
    print(f"steps={env.steps} reward={total:.2f} distance={info.get('distance', 0):.2f} reached={info.get('reached', False)}")
    if mode == "cli":
        print(f"Saved {save_video(frames)}")


if __name__ == "__main__":
    main()
