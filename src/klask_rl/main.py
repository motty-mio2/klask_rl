import mujoco  # type: ignore
import glfw  # type: ignore
import numpy as np
import os


def main() -> None:
    # XMLパス
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.normpath(os.path.join(current_dir, "../../model/klask.xml"))

    # モデルの読み込み
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # GLFWの初期化
    if not glfw.init():
        return

    # ウィンドウ作成
    window = glfw.create_window(1200, 900, "Klask Simulation (GLFW)", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # MuJoCo 描画コンテキストの初期化
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 初期視点の設定
    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 0.8
    cam.lookat = np.array([0, 0, 0])

    # 制御変数
    ctrl_x, ctrl_y = 0.0, 0.0
    step_size = 0.005
    range_x = [-0.1511, 0.1511]
    range_y = [-0.0976, 0.0976]

    print("Klask Simulation Started")
    print("Use Arrow Keys to move the striker.")
    print("Press ESC to exit.")

    while not glfw.window_should_close(window):
        # キー入力
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            ctrl_x -= step_size
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            ctrl_x += step_size
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            ctrl_y += step_size
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            ctrl_y -= step_size

        ctrl_x = np.clip(ctrl_x, range_x[0], range_x[1])
        ctrl_y = np.clip(ctrl_y, range_y[0], range_y[1])

        data.ctrl[0] = ctrl_x
        data.ctrl[1] = ctrl_y

        # 物理演算を複数ステップ進めて実時間に合わせる
        sim_start = data.time
        while data.time - sim_start < 1.0 / 60.0:
            mujoco.mj_step(model, data)

        # 描画
        viewport = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(
            model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, viewport[0], viewport[1]), scene, context
        )

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
