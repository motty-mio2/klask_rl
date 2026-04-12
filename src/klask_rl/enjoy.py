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

    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, "Klask AI Self-Play Mode", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 視点を少し調整
    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 1.0
    cam.lookat = np.array([0, 0, 0])

    obs, _ = env.reset()
    
    print("Self-Play Mode Started. AI vs AI.")

    while not glfw.window_should_close(window):
        # 1つのモデルが 4次元アクション(P1 & P2)を同時出力
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()

        viewport = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, viewport[0], viewport[1]), scene, context)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)

    glfw.terminate()

if __name__ == "__main__":
    main()
