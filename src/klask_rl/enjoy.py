import mujoco
import glfw
import numpy as np
import os
import time
from stable_baselines3 import PPO
from env import KlaskEnv

def main():
    env = KlaskEnv()
    model_path = "models/klask_ppo_model"
    
    if not os.path.exists(model_path + ".zip"):
        print("Error: Model not found.")
        return

    model = PPO.load(model_path)
    env.opponent_policy = model

    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, "Klask Debug Mode", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 1.0
    cam.lookat = np.array([0, 0, 0])

    obs, _ = env.reset()
    
    print("\n--- Klask Debug: Real-time Coordinates ---")
    print("Format: [Puck X, Y] | [S1 X, Y] | [Action X, Y]")

    step_count = 0
    while not glfw.window_should_close(window):
        # AI 推論
        action, _ = model.predict(obs, deterministic=True)
        
        # 物理座標の取得 (デバッグ用)
        p_pos = env.data.qpos[4:6] # ボール
        s1_pos = env.data.qpos[0:2] + np.array([0, -0.105]) # P1位置
        
        # 10ステップに1回表示 (ログが早すぎるのを防ぐ)
        if step_count % 10 == 0:
            print(f"Puck: {p_pos[0]:>6.3f}, {p_pos[1]:>6.3f} | "
                  f"S1: {s1_pos[0]:>6.3f}, {s1_pos[1]:>6.3f} | "
                  f"Act: {action[0]:>6.3f}, {action[1]:>6.3f}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("--- Reset Episode ---")
            obs, _ = env.reset()

        viewport = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, viewport[0], viewport[1]), scene, context)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)
        step_count += 1

    glfw.terminate()

if __name__ == "__main__":
    main()
