import numpy as np
from gymnasium.utils.env_checker import check_env

from hummingbird_env import HummingbirdHoverEnv


def main():
    check_env(HummingbirdHoverEnv(), skip_render_check=False)
    env = HummingbirdHoverEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=1, options={"randomize": False})
    assert obs.shape == (18,)
    total = 0.0
    for _ in range(25):
        obs, reward, terminated, truncated, _ = env.step(
            np.zeros(4, dtype=np.float32)
        )
        total += reward
        assert env.render().shape == (512, 768, 3)
        if terminated or truncated:
            break
    env.close()
    print("Smoke test passed; hover reward:", total)


if __name__ == "__main__":
    main()
