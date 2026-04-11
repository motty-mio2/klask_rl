import mujoco
import glfw
import numpy as np
import os
import time

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.normpath(os.path.join(current_dir, "../../model/klask.xml"))
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, "Klask Simulation (GLFW)", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 視点
    cam.azimuth = 90
    cam.elevation = -50
    cam.distance = 0.8
    cam.lookat = np.array([0, -0.1, 0])

    # 制御変数 (初期位置 0, 0 は XML の pos="0 -0.105" に相当)
    ctrl_x, ctrl_y = 0.0, 0.0
    step_size = 0.005 # 直径が小さくなったのでステップも微調整
    
    # XML の joint range と厳密に合わせる
    range_x = [-0.1511, 0.1511]
    range_y = [-0.0976, 0.0976]

    print("Klask Simulation: Accurate Dimensions Applied")
    print(f"Board: 317mm x 210mm (half)")
    print(f"Striker: Diameter 14.8mm")

    while not glfw.window_should_close(window):
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

        sim_start = data.time
        while data.time - sim_start < 1.0/60.0:
            mujoco.mj_step(model, data)

        viewport = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, viewport[0], viewport[1]), scene, context)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
