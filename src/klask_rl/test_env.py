import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import KlaskEnv

def test():
    print("Testing KlaskEnv instantiation...")
    env = KlaskEnv()
    obs, _ = env.reset()
    print("Reset successful. Initial obs shape:", obs.shape)
    
    # 履歴数が3の場合、4変数 * 3 = 12次元のはず
    expected_dim = 4 * env.num_history
    assert obs.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {obs.shape}"
    
    print("Stepping through environment with motor limits...")
    for i in range(10):
        action = np.array([0.5, -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        # 各種値のプリント
        print(f"Step {i+1}:")
        print(f"  obs shape: {obs.shape}")
        print(f"  reward: {reward:.4f}")
        print(f"  striker p1 pos in mjData: {env.data.qpos[0:2]}")
        print(f"  striker p1 ctrl (filtered): {env.data.ctrl[0:2]}")
        print(f"  striker p1 motor speed: {env.motor_filter_p1_x.vel:.4f}, {env.motor_filter_p1_y.vel:.4f}")

    print("All environment step tests passed successfully!")

if __name__ == "__main__":
    test()
