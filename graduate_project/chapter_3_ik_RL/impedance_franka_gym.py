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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)


        # 一些必要的全局变量
        self.target_pos = None
        self.target_quat = None
        self.action = None
        self.N = None  # 这是轨迹点的数量
        self.i = None  # 这是运行到第几个轨迹点
        self.rot_weight = 1.0 # 确定旋转误差相对于平移误差的权重
        self.err_norm = None# 位置姿态总误差
        self.IKResult = None # 运动学逆解
        self.traj_qpos = None # 控制器上一轮的解算位置
        self.record_ex_ctrl = None        # 控制器本应该输出的位置记录
        self.record_real_ctrl = None # 控制器因为受限实际的控制位置记录
        self.target_traj = None
        self.max_qpos = None
        self.min_qpos = None
        self.saturation_time = None

        self.ex_ctrl = None
        self.record_qpos = None
        self.error_ctrl = None
        self.record_site_xpos = None
        self.error_xpos = None
        self.record_site_quat = None

        self.first_time = True
        
        
        self.site_tracking_record = None # 记录每一次轨迹后末端跟踪情况

        # 不懂第一句是什么
        self.render_mode = render_mode

    # 定义观察值
    def _get_obs(self):
        return np.concatenate(
            [
                # 预留一个受限问题
                # self.data.qpos.flatten()[:7],
                # self.data.site_xpos[0][:3],
                # site_quat,
                # self.target_pos[self.i],
                # self.target_quat,
                self.max_qpos,
                self.min_qpos,
                self.saturation_time,
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
                             regularization_strength = np.array([1., 1., 1., 1., 1., 1., 1.]),
                             damp = np.array([0,0,0,0,0,0,0]),
                             spring = np.array([0,0,0,0,0,0,0])):

        if self.IKResult == None:
            self.traj_qpos = np.zeros(shape=(7,))
        else:
            self.traj_qpos = self.IKResult.qpos
        # 计算运动学解
        self.IKResult = ik.qpos_from_site_pose(self.model, self.data_copy, site_name="attachment_site",
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
        q_d_vel = self.IKResult.qpos - self.traj_qpos
        # 阻抗控制律
        damp = damp * 10 + 10
        spring = spring * 100 + 100
        tau = self.data.qfrc_bias + \
              damp * (q_d_vel - self.data.qvel) + \
            spring * (self.IKResult.qpos - self.data.qpos)

        self.data.ctrl = tau


    # 重置函数
    def reset(self, seed=None, options=None):

        # 重置模型位置  这个key我暂时不确定是啥意思
        mj.mj_resetDataKeyframe(self.model,self.data,0)
        mj.mj_forward(self.model, self.data)
        self.data_copy = copy.copy(self.data)

        # 初始参数
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

        self.max_qpos = np.ones(7) * np.pi
        self.min_qpos = - np.ones(7) * np.pi
        self.saturation_time = np.zeros(7)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # 清空记录
        self.record_ex_ctrl = None
        self.record_qpos = None
        self.record_site_xpos = None
        self.record_site_quat = None

        done = False
        # self.saturation_time = None

        action = (action + 1) * 2
        print("action = ", action)

        # 每一步循环一次轨迹循环 记住 重构代码
        for i in range(self.N):
            # 执行一步控制器 pos_controller
            self.data_copy.qpos = self.data.qpos
            self.data_copy.site_xpos = self.data.site_xpos
            self.data_copy.site_xmat = self.data.site_xmat

            # 这里修改一下action的范围 将(-1 1)转到(0,10)
            IKResult = self.pos_controller(target_pos=self.target_pos[i],
                                           target_quat=self.target_quat,
                                           regularization_strength=action)

            # 这里仅用于受限情况的强化学习 使用max min函数就行
            # 本应该用于控制的量
            ex_ctrl = IKResult.qpos
            # 实际因为关节受限给到的控制量
            real_ctrl = ex_ctrl
            # 限制关节4的运动角度
            if real_ctrl[3] > -1.4:
                real_ctrl[3] = -1.4
            elif real_ctrl[3] < -1.6:
                real_ctrl[3] = -1.6
            # 把执行过程中的self.ex_ctrl控制量记录下来
            if i == 0:
                self.record_ex_ctrl = ex_ctrl
            else:
                self.record_ex_ctrl = np.vstack((self.record_ex_ctrl, ex_ctrl))
            # 给到控制量
            self.data.ctrl = real_ctrl

            # 控制仿真帧率
            # 注意模拟帧率和控制器帧率不同
            step_time = self.data.time
            #  执行仿真 mj_step
            while (self.data.time - step_time < 1.0 / 60.0):
                mj.mj_step(self.model, self.data)

            # 执行一步render
            if self.render_mode == "human":
                self._render_frame()

            # 记录关节位置和末端姿态
            if i == 0:
                self.record_qpos = self.data.qpos
                self.record_site_xpos = self.data.site_xpos[0][:3]
                self.record_site_quat = np.empty(4, dtype=np.float64)
                mj.mju_mat2Quat(self.record_site_quat, self.data.site_xmat[0])

            else:
                self.record_qpos = np.vstack((self.record_qpos, self.data.qpos))
                self.record_site_xpos = np.vstack((self.record_site_xpos, self.data.site_xpos[0][:3]))

                site_quat = np.empty(4, dtype=np.float64)
                mj.mju_mat2Quat(site_quat, self.data.site_xmat[0])
                self.record_site_quat = np.vstack((self.record_site_quat, site_quat))


        # 需要记录一些数据
        # obs需要 受限关节编号和时长
        # reward 连续 or 离散?? 需要 目标轨迹和实际末端轨迹
        # 因此需要 关节角度变化 关节命令变化 末端点姿态变化 用几个数据存一下

        # 控制偏差
        self.error_ctrl = self.record_qpos - self.record_ex_ctrl
        # 位置偏差
        self.error_xpos = self.record_site_xpos - self.target_traj

        # 绘制一下跟踪图像
        plt.ion()
        plt.clf()
        ax_tracking = plt.axes(projection='3d')
        ax_tracking.scatter3D(self.record_site_xpos[:, 0], self.record_site_xpos[:, 1],
                              self.record_site_xpos[:, 2], cmap='b')
        ax_tracking.plot3D(self.target_pos[:, 0], self.target_pos[:, 1], self.target_pos[:, 2], 'red')
        ax_tracking.set_xlim((0.16,0.96))
        ax_tracking.set_ylim((-0.2,0.2))
        ax_tracking.set_zlim((0.4,0.8))

        plt.pause(1)

        joints = plt.axes()
        for i in range(7):
            joints.plot(self.record_qpos[:,i])
        joints.legend(['1','2','3','4','5','6','7'])
        # plt.pause(0.1)
        plt.pause(1)
        plt.ioff()

        # 这里给定观察值 就设置为关节最大转动差值 和 饱和偏差时间 共14个变量吧
        # 关节最大转动差值
        self.max_qpos = np.max(self.record_qpos, axis=0)
        self.min_qpos = np.min(self.record_qpos, axis=0)
        # 饱和偏差时间
        self.error_ctrl = np.absolute(self.error_ctrl)
        self.saturation_time = np.sum( self.error_ctrl > 0.05, axis=0)
        print("saturation_time = ", self.saturation_time)

        # 计算reward 综合考虑跟踪误差 和 饱和偏差时间
        # 计算每个点的误差值后求和
        error_xpos = np.linalg.norm(self.error_xpos, axis=1) # 每个点的位置误差
        error_xpos = - np.mean(error_xpos)
        # 计算饱和偏差时间
        total_saturation_time = sum(self.saturation_time)

        # 终止条件
        if error_xpos < -0.5 or total_saturation_time > self.N * 7 * 0.1:
            done = True
            reward = - 10
        elif total_saturation_time == 0:
            done = True
            reward = 1 + error_xpos
        else:
            done = False
            reward = 1 + error_xpos + (-0.1 * total_saturation_time)


        print(reward)
        # # 计算位置误差
        # err_pos = self.target_pos[self.i] - self.data.site_xpos[0][:3]
        # self.err_norm = np.linalg.norm(err_pos)
        # # 计算姿态误差
        # site_xquat = np.empty(4, dtype=np.float64)
        # neg_site_xquat = np.empty(4, dtype=np.float64)
        # err_rot_quat = np.empty(4, dtype=np.float64)
        # err_rot = np.empty(3, dtype=np.float64)
        # mj.mju_mat2Quat(site_xquat, self.data.site_xmat[0])
        # mj.mju_negQuat(neg_site_xquat, site_xquat)
        # mj.mju_mulQuat(err_rot_quat, self.target_quat, neg_site_xquat)
        # mj.mju_quat2Vel(err_rot, err_rot_quat, 1)
        # self.err_norm += np.linalg.norm(err_rot) * self.rot_weight
        #
        # reward = -self.err_norm

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
