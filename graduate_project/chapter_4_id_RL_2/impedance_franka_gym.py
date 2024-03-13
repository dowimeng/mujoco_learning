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
import matplotlib.pyplot as plt


class ImpedanceFrankaGym(gym.Env):

    def __init__(self, render_mode=None):


        # 给定模型文件
        xml_path = "franka_emika_panda/scene.xml"
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.data_copy = copy.copy(self.data)

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(56,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)


        # 一些必要的全局变量
        self.target_pos = None
        self.target_quat = None
        self.N = None  # 这是轨迹点的数量
        self.i = None  # 这是运行到第几个轨迹点
        self.traj_qpos = None # 控制器上一轮的解算位置
        self.target_traj = None
        self.saturation_time = None

        # 初始化记录数据
        self.record_torque_control = None  # 控制力矩
        self.record_qpos_control = None # 控制关节角
        self.record_qpos_real = None  # 实际关节角
        self.record_spring = None  # spring值
        self.record_damp = None  # damp值
        self.record_weight = None  # 阻尼最小二乘权重
        self.record_xpos = None  # 末端位置
        self.record_xmat = None  # 末端角度

        self.spring = None
        self.damp = None
        self.weight = None

        # 不懂第一句是什么
        self.render_mode = render_mode

    # 定义观察值
    def _get_obs(self):

        site_quat = np.empty(4, dtype=np.float64)
        mj.mju_mat2Quat(site_quat, self.data.site_xmat[0])

        if self.i == 0 :
            return np.concatenate(
                [
                    # 预留一个受限问题
                    self.record_qpos_control,
                    self.record_qpos_real,
                    self.record_torque_control,
                    self.target_pos[self.i],
                    self.data.site_xpos[0][:3],
                    self.target_quat,
                    site_quat,
                    self.spring,
                    self.damp,
                    self.weight,
                ]
            )
        else:
            return np.concatenate(
                [
                    # 预留一个受限问题
                    self.record_qpos_control[-1, :],
                    self.record_qpos_real[-1, :],
                    self.record_torque_control[-1, :],
                    self.target_pos[self.i],
                    self.data.site_xpos[0][:3],
                    self.target_quat,
                    site_quat,
                    self.spring,
                    self.damp,
                    self.weight,
                ]
            )

    # 定义信息值
    def _get_info(self):
        # pass
        return {
            "None" : None
        }

    def pos_controller(self, target_pos, target_quat, regularization_strength = np.array([1., 1., 1., 1., 1., 1., 1.])):
        IKResult = ik.qpos_from_site_pose(self.model, self.data_copy, site_name="attachment_site",
                                          target_pos=target_pos, target_quat=target_quat,
                                          regularization_strength=regularization_strength,
                                          regularization_threshold=0.0001,
                                          max_steps=100)
        # self.data.ctrl = IKResult.qpos
        return IKResult

    def impedance_controller(self,target_pos, target_quat,
                             regularization_strength = np.ones(7),
                             damp = np.ones(7) * 10,
                             spring = np.ones(7) * 100):

        # 计算运动学解
        IKResult = ik.qpos_from_site_pose(self.model, self.data_copy, site_name="attachment_site",
                                          target_pos=target_pos, target_quat=target_quat,
                                          regularization_strength=regularization_strength,
                                          regularization_threshold=0.0001,
                                          max_steps=100)

        # 计算单关节力矩控制值 简化版 设置M_d = M
        # 计算M矩阵
        # M = np.zeros(shape=(self.model.nv, self.model.nv), dtype=np.float64)
        # mj.mj_fullM(self.model, M, self.data.qM)
        # 科氏力和重力的和 data.qfrc_bias
        # 计算轨迹速度
        q_d_vel = IKResult.qpos - self.data.qpos
        # 阻抗控制律
        tau = self.data.qfrc_bias + \
              damp * (q_d_vel - self.data.qvel) + \
            spring * (IKResult.qpos - self.data.qpos)

        # self.data.ctrl = tau
        return tau, IKResult

    # 重置函数
    def reset(self, seed=None, options=None):

        # 重置模型位置  这个key我暂时不确定是啥意思
        mj.mj_resetDataKeyframe(self.model,self.data,0)
        mj.mj_forward(self.model, self.data)
        self.data_copy = copy.copy(self.data)

        # 初始动作参数
        self.spring = np.ones(7) * 100
        self.damp = np.ones(7) * 10
        self.weight = np.ones(7)

        self.N = 300
        self.i = 0
        # xy方向双纽线轨迹
        a = 0.2
        theta = np.linspace(-np.pi/2, 3*np.pi/2, self.N)
        center = np.array(self.data.site_xpos[0][:3])
        x_traj = center[0] + a * np.sin(theta) * np.cos(theta) / (1 + np.sin(theta) ** 2)
        y_traj = center[1] + a * np.cos(theta) / (1 + np.sin(theta) ** 2)
        z_traj = center[2] + np.zeros((np.shape(theta)[0]))
        self.target_traj = np.dstack((x_traj, y_traj, z_traj))

        # 目标点位置和姿态
        self.target_pos = self.target_traj[0]
        self.target_quat = np.empty(4, dtype=np.float64)
        mj.mju_mat2Quat(self.target_quat, self.data.site_xmat[0])

        if self.render_mode == "human":
            self._render_frame()

        ## 初始化记录数据并执行一步
        # 控制器计算
        self.data_copy.qpos = self.data.qpos
        self.data_copy.site_xpos = self.data.site_xpos
        self.data_copy.site_xmat = self.data.site_xmat
        tau, IKResult = self.impedance_controller(target_pos=self.target_pos[self.i], target_quat=self.target_quat,
                                  regularization_strength=self.weight,
                                  damp=self.damp,
                                  spring=self.spring)
        self.data.ctrl = tau

        # 一步仿真
        step_time = self.data.time
        while (self.data.time - step_time < 1.0 / 60.0):
            mj.mj_step(self.model, self.data)
        if self.render_mode == "human":
            self._render_frame()

        self.record_torque_control = tau  # 控制力矩
        self.record_qpos_control = IKResult.qpos  # 控制关节角
        self.record_qpos_real = self.data.qpos  # 实际关节角
        self.record_spring = self.spring  # spring值
        self.record_damp = self.damp  # damp值
        self.record_weight = self.weight  # 阻尼最小二乘权重
        self.record_xpos = self.data.site_xpos[0][:3]  # 末端位置
        self.record_xmat = self.data.site_xmat[0]  # 末端角度

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        self.i += 1

        # 将传进的归一化action修改 暂不考虑阻尼最小二乘矩阵的参数修改
        # spring每一步可以改变（1） damp每一步可以改变（0.1）weight每一步可以改变（0.01）初始值为100 10 1
        self.spring += action[:7]
        self.damp += 0.1 * action[7:14]
        self.weight += 0.01 * action[7:14]
        # 取正操作
        self.spring = np.max(np.array((self.spring, np.zeros(7))), axis=0)
        self.damp = np.max(np.array((self.damp, np.zeros(7))), axis=0)
        self.weight = np.max(np.array((self.weight, np.zeros(7))), axis=0)
        # 控制器计算
        self.data_copy.qpos = self.data.qpos
        self.data_copy.site_xpos = self.data.site_xpos
        self.data_copy.site_xmat = self.data.site_xmat
        tau, IKResult = self.impedance_controller(target_pos=self.target_pos[self.i], target_quat=self.target_quat,
                                  regularization_strength=self.weight,
                                  damp=self.damp,
                                  spring=self.spring)
        self.data.ctrl = tau

        # 一步仿真
        step_time = self.data.time
        while (self.data.time - step_time < 1.0 / 60.0):
            mj.mj_step(self.model, self.data)
        if self.render_mode == "human":
            self._render_frame()

        # 设计观察量 存储好各类变量
        self.record_torque_control = np.vstack((self.record_torque_control, tau))
        self.record_qpos_control = np.vstack((self.record_qpos_control, IKResult.qpos))
        self.record_qpos_real = np.vstack((self.record_qpos_real, self.data.qpos))
        self.record_spring = np.vstack((self.record_spring, self.spring))
        self.record_damp = np.vstack((self.record_damp, self.damp))
        self.record_weight = np.vstack((self.record_weight, self.weight))
        self.record_xpos = np.vstack((self.record_xpos, self.data.site_xpos[0][:3]))
        self.record_xmat = np.vstack((self.record_xmat, self.data.site_xmat[0]))

        ## 设计reward
        # 这部分是末端跟踪误差
        err_pos = self.target_pos[self.i] - self.data.site_xpos[0][:3]
        site_xquat = np.empty(4, dtype=np.float64)
        neg_site_xquat = np.empty(4, dtype=np.float64)
        err_rot_quat = np.empty(4, dtype=np.float64)
        err_rot = np.empty(3, dtype=np.float64)
        mj.mju_mat2Quat(site_xquat, self.data.site_xmat[0])
        mj.mju_negQuat(neg_site_xquat, site_xquat)
        mj.mju_mulQuat(err_rot_quat, self.target_quat, neg_site_xquat)
        mj.mju_quat2Vel(err_rot, err_rot_quat, 1)
        err_x = np.linalg.norm(err_pos) + np.linalg.norm(err_rot) * 1.0
        # 这部分是关节跟踪误差
        err_qpos = IKResult.qpos - self.data.qpos
        err_qpos = np.absolute(err_qpos)
        err_qpos = np.sum(err_qpos)

        reward = err_x + err_qpos
        # 绘制图像和存储等功能
        if self.i == self.N - 1:
            plt.ion()
            plt.clf()
            ax_tracking = plt.axes(projection='3d')
            ax_tracking.scatter3D(self.record_xpos[:, 0], self.record_xpos[:, 1],
                                  self.record_xpos[:, 2], cmap='b')
            ax_tracking.plot3D(self.target_pos[:, 0], self.target_pos[:, 1], self.target_pos[:, 2], 'red')
            ax_tracking.set_xlim((0.16, 0.96))
            ax_tracking.set_ylim((-0.2, 0.2))
            ax_tracking.set_zlim((0.4, 0.8))

            plt.pause(0.1)
            plt.ioff()
            done = True
        else:
            done = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

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
