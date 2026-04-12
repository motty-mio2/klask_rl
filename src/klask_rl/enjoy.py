import os
import time
from typing import Any

import glfw  # type: ignore
import mujoco  # type: ignore
import numpy as np
from env import KlaskEnv
from stable_baselines3 import PPO

# マウス操作用のグローバル変数
button_left = False
button_middle = False
button_right = False
lastx: float = 0
lasty: float = 0


def main() -> None:
    env = KlaskEnv()
    model_path = "models/klask_ppo_model"

    if not os.path.exists(model_path + ".zip"):
        print("Error: Model not found.")
        return

    model = PPO.load(model_path)
    env.opponent_policy = model

    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, "Klask AI Free View Mode", None, None)
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

    # 初期カメラ設定
    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 1.0
    cam.lookat = np.array([0, 0, 0])

    # マウスコールバック
    def mouse_button(window: Any, button: int, act: int, mods: int) -> None:
        global button_left, button_middle, button_right
        button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

    def mouse_move(window: Any, xpos: float, ypos: float) -> None:
        global lastx, lasty
        dx = xpos - lastx
        dy = ypos - lasty
        lastx = xpos
        lasty = ypos

        if not button_left and not button_middle and not button_right:
            return

        _, height = glfw.get_window_size(window)

        # どのボタンが押されているかでカメラ操作を切り替え
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        action = (
            mujoco.mjtMouse.mjMOUSE_ZOOM
            if button_right
            else mujoco.mjtMouse.mjMOUSE_MOVE_H
            if (button_left and mod_shift)
            else mujoco.mjtMouse.mjMOUSE_ROTATE_H
        )

        mujoco.mjv_moveCamera(env.model, action, dx / height, dy / height, scene, cam)

    def scroll(window: Any, xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(
            env.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scene, cam
        )

    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)

    obs, _ = env.reset()

    print("\n--- Klask AI Free View Mode ---")
    print("Mouse Left: Rotate")
    print("Mouse Right / Scroll: Zoom")
    print("Shift + Left: Move")
    print("ESC: Exit")

    while not glfw.window_should_close(window):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        viewport = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(
            env.model, env.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, viewport[0], viewport[1]), scene, context
        )

        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)

    glfw.terminate()


if __name__ == "__main__":
    main()
