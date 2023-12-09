import os

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import inverse_kinematics as ik
from pyquaternion import Quaternion

xml_path = 'franka_emika_panda/scene.xml'  # xml file (assumes this is in the same folder as this file)
simend = 100  # simulation time
print_camera_config = 0  # set to 1 to print camera config
# this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def init_controller(model, data):
    # initialize the controller here. This function is called once, in the beginning
    # 来到初始位置
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)


def controller(model, data):
    # put the controller here. This function is called inside the simulation.
    # pass
    data.ctrl = IKResult.qpos


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height,
                      dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 107.00000000000009;
cam.elevation = -34.99999999999994;
cam.distance = 2.092493995802527
cam.lookat = np.array([0.11680416692690154, 0.0030176585738958166, 0.38820110102502614])

# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

N = 200

# 获取初始末端位置
position_Q = data.site_xpos[0]
# print(position_Q)

# 定义末端轨迹
r = 0.1
center = np.array([position_Q[0] - r, position_Q[1], position_Q[2]])
phi = np.linspace(0, 10 * np.pi, N)
x_ref = center[0] + r * np.cos(phi)
y_ref = center[1] + r * np.sin(phi)


# target_quat = data.site_xmat
# target_quat = target_quat[0].reshape(3,3)
# target_quat = Quaternion(matrix=target_quat)
# target_quat = [target_quat.x,target_quat.y,target_quat.z,target_quat.w]
# print(target_quat)
# print(len(x_ref))

i = 0
time = 0
dt = 0.001

while not glfw.window_should_close(window):
    # time_prev = data.time
    time_prev = time

    while (time - time_prev < 1.0 / 60.0):
        target_pos = np.array([x_ref[i], y_ref[i], position_Q[2]])

        # 复制一个独立的data变量出来
        # 需要以下量  qpos site_xpos site_xmat 就这三个变量
        data_copy = data.__copy__()
        data_copy.qpos = data.qpos
        data_copy.site_xpos = data.site_xpos
        data_copy.site_xmat = data.site_xmat
        # 得到 dq
        # IKResult = ik.qpos_from_site_pose(model, data_copy, "attachment_site", target_pos=target_pos,target_quat=target_quat)
        IKResult = ik.qpos_from_site_pose(model, data_copy, "attachment_site", target_pos=target_pos)
        # print(data.qfrc_passive)

        # 更新参考轨迹序列
        # data.qpos = IKResult.qpos
        mj.mj_step(model, data)
        time += dt

    i += 1

    if (data.time >= simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    if (print_camera_config == 1):
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
