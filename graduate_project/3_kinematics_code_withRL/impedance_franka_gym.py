import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import os
import mujoco as mj
from mujoco.glfw import glfw
import time
import inverse_kinematics as ik
import copy


class ImpedanceFrankaGym(gym.Env):

    def __init__(self, render_mode=None):


        # 给定模型文件
        xml_path = "franka_emika_panda/scene.xml"
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # 其他配置
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

        # 初始化GLFW环境 创建window
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # For callback functions
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self._keyboard)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_scroll_callback(self.window, self._scroll)

        # Example on how to set camera configuration
        self.cam.azimuth = 107.00000000000009;
        self.cam.elevation = -34.99999999999994;
        self.cam.distance = 2.092493995802527
        self.cam.lookat = np.array([0.11680416692690154, 0.0030176585738958166, 0.38820110102502614])

        # 观察空间和动作空间
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0.1, 10.0, shape=(7,), dtype=np.float64),
                "target": spaces.Box(0, np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.obs = np.array([np.inf])

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Box(0.1, 10.0, shape=(7,), dtype=np.float64)
        self.action = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0])

        # 不懂第一句是什么
        self.render_mode = render_mode



    # 定义观察值
    def _get_obs(self):
        return {"agent": self.action, "target": self.obs}

    # 定义信息值
    def _get_info(self):
        # pass
        return {
            "distance": 1
        }

    def pos_controller(self, target_pos, target_quat, action):
        IKResult = ik.qpos_from_site_pose(self.model, self.data, site_name="attachment_site",
                                          target_pos=target_pos, target_quat=target_quat,
                                          regularization_strength=action,
                                          regularization_threshold=0.0001,
                                          max_steps=1)
        self.data.ctrl = IKResult.qpos

    # 重置函数
    def reset(self, seed=None, options=None):

        # 重置模型位置  这个key我暂时不确定是啥意思
        mj.mj_resetDataKeyframe(self.model,self.data,0)
        mj.mj_forward(self.model, self.data)

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        # 给轨迹,空间圆形
        center = np.array(self.data.site_xpos[0][:3])
        self.N = 100
        r = 0.1
        phi = np.linspace(0, 2 * np.pi, self.N)
        x_traj = center[0] + r * np.cos(phi) * np.cos(-np.pi / 4)
        y_traj = center[1] + r * np.sin(phi) * np.cos(-np.pi / 4)
        z_traj = center[2] + r * np.cos(phi) * np.sin(-np.pi / 4)
        target_traj = np.dstack((x_traj, y_traj, z_traj))
        self.target_traj = target_traj[0]

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # 一步就是一个轨迹循环
    def step(self, action):

        if self.render_mode == "human":
            self._render_frame()

        # 得到site初始位置和姿态
        target_pos = copy.copy(self.data.site_xpos[0][:3])
        target_quat = np.empty(4, dtype=float)
        mj.mju_mat2Quat(target_quat, self.data.site_xmat[0])


        # 循环体主要轨迹跟踪部分
        now_r = time.time()
        i = 1
        traj_tracking_record = self.data.site_xpos[0][:3]

        # 主循环
        while i < self.N - 1:
            # 计算和更新轨迹
            if time.time() - now_r >= 0.001:
                # 更新目标位置
                now_r = time.time()
                i += 1
                # 控制器部分
                self.action = action
                mj.set_mjcb_control(self.pos_controller(target_pos, target_quat, action))
            target_pos = self.target_traj[i]
            # 记录当前跟踪效果
            traj_tracking_record = np.vstack((traj_tracking_record, self.data.site_xpos[0][:3]))
            mj.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self._render_frame()

        err_traj = self.target_traj - traj_tracking_record
        err_norm = np.linalg.norm(err_traj,axis=1)
        reward = np.mean(err_norm)
        self.obs = np.array([reward])
        print(reward)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        viewport_width, viewport_height = glfw.get_framebuffer_size(
            self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                           mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()


    def close(self):
        glfw.terminate()

        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def _mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

    def _mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
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
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx / height,
                          dy / height, self.scene, self.cam)

    def _scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                          yoffset, self.scene, self.cam)
