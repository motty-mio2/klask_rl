import mujoco
import glfw
import numpy as np
import os
import time
from stable_baselines3 import PPO
from env import KlaskEnv

def main():
    # 環境の作成 (レンダリングなしで初期化し、自前でGLFW描画)
    env = KlaskEnv()
    model_path = "models/klask_ppo_model"
    
    if not os.path.exists(model_path + ".zip"):
        print("Error: Model not found.")
        return

    # 学習済みモデルの読み込み
    model = PPO.load(model_path)

    # GLFWの初期化
    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, "Klask AI Enjoy Mode", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # MuJoCo 描画コンテキスト
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 視点設定
    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 0.8
    cam.lookat = np.array([0, -0.05, 0])

    obs, _ = env.reset()
    
    print("Enjoy Mode Started. Press ESC to exit.")

    while not glfw.window_should_close(window):
        # AI による行動決定
        action, _states = model.predict(obs, deterministic=True)
        
        # 環境のステップ実行
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()

        # 描画
        viewport = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, viewport[0], viewport[1]), scene, context)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        # 実時間に近い速度で表示
        time.sleep(0.01)

    glfw.terminate()

if __name__ == "__main__":
    main()
