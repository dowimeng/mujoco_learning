'''
这个代码用于测试 位置控制
请提前将xml文件中的控制器换为位置控制器
'''

import copy
import threading
import time

import matplotlib.pyplot as plt
import numpy as np

import impedance_franka_gym
import inverse_kinematics as ik


# 定义位置控制器
def pos_controller():
    global custom_env, target_pos

    # 控制频率设置
    ctrl_rate = 1 / custom_env.model.opt.timestep
    # 用于冗余位置控制计算inverse_kinematics使用
    data_copy = copy.copy(custom_env.data)

    while run_controller:
        now_c = time.time()
        # 更新目标位置
        target_pos[0], target_pos[1], target_pos[2] = x_ref, y_ref, z_ref
        # 给到冗余运动学控制器进行逆运动学解算
        # data_copy = copy.copy(custom_env.data)
        # data_copy.qpos = custom_env.data.qpos
        # data_copy.site_xpos = custom_env.data.site_xpos
        # data_copy.site_xmat = custom_env.data.site_xmat
        IKResult = ik.qpos_from_site_pose(custom_env.model, data_copy, site_name="attachment_site",
                                          target_pos=target_pos,
                                          max_steps=2)
        custom_env.data.ctrl = IKResult.qpos
        # print(custom_env.data.ctrl)

        elapsed_c = time.time() - now_c
        sleep_time_c = (1. / ctrl_rate) - elapsed_c
        if sleep_time_c > 0:
            time.sleep(sleep_time_c)


# 定义轨迹生成器 这里的轨迹就是一个圆
def traj_generator():
    global x_ref, y_ref, z_ref, target_pos, x_ref_traj, y_ref_traj, z_ref_traj, N

    # 轨迹部分
    N = 200
    # 获取初始末端位置
    position_Q = copy.copy(custom_env.data.site_xpos[0])
    # 定义末端轨迹
    r = 0.1
    center = np.array([position_Q[0] - r, position_Q[1], position_Q[2]])
    phi = np.linspace(0, 2 * np.pi, N)
    x_ref_traj = center[0] + r * np.cos(phi)
    y_ref_traj = center[1] + r * np.sin(phi)
    z_ref_traj = position_Q[2].__copy__()

    x_ref = x_ref_traj[0]
    y_ref = y_ref_traj[0]
    z_ref = copy.copy(position_Q[2])

    target_pos = position_Q


if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 重置环境
    custom_env.reset()
    time.sleep(0.01)

    # 生成圆形轨迹
    traj_generator()

    # 得到site初始位置
    target_pos = copy.copy(custom_env.data.site_xpos[0])

    # 开启独立的控制器线程
    run_controller = True
    ctrl_thread = threading.Thread(target=pos_controller)
    ctrl_thread.start()

    # 循环体主要轨迹跟踪部分
    now_r = time.time()
    i = 0
    x_all = []
    y_all = []
    while i < N:
        # 计算和更新轨迹
        if time.time() - now_r >= 0.01:
            # 更新目标位置
            x_ref, y_ref, z_ref = x_ref_traj[i], y_ref_traj[i], z_ref_traj
            x_all.append(custom_env.data.site_xpos[0][0])
            y_all.append(custom_env.data.site_xpos[0][1])
            now_r = time.time()
            i += 1
        custom_env.step(action=None)

    # 绘制跟踪效果
    plt.figure(1)
    plt.plot(x_all, y_all, 'bx')
    plt.plot(x_ref_traj, y_ref_traj, 'r-')
    plt.show()

    while 1:
        custom_env.step(action=None)

        # custom_env.render()
