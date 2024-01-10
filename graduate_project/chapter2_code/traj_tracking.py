'''
这个代码用于测试 位置控制
请提前将xml文件中的控制器换为位置控制器
主要用于测试阻尼最小二乘法的轨迹跟踪效果
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
        # target_pos[0], target_pos[1], target_pos[2] = x_ref, y_ref, z_ref
        # 给到冗余运动学控制器进行逆运动学解算
        # data_copy = copy.copy(custom_env.data)
        IKResult = ik.qpos_from_site_pose(custom_env.model, data_copy, site_name="attachment_site",
                                          target_pos=target_pos,
                                          regularization_strength=1,
                                          max_steps=2)
        custom_env.data.ctrl = IKResult.qpos

        # print(custom_env.data.ctrl)

        elapsed_c = time.time() - now_c
        sleep_time_c = (1. / ctrl_rate) - elapsed_c
        if sleep_time_c > 0:
            time.sleep(sleep_time_c)



if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 重置环境
    custom_env.reset()
    time.sleep(0.01)

    # 得到site初始位置
    target_pos = copy.copy(custom_env.data.site_xpos[0][:3])
    err = []

    # 给轨迹,空间圆形
    center = np.array(custom_env.data.site_xpos[0][:3])
    N = 2000
    r = 0.2
    phi = np.linspace(0, 6*np.pi, N)
    x_traj = center[0] + r * np.cos(phi) * np.cos(np.pi/4)
    y_traj = center[1] + r * np.sin(phi) * np.cos(np.pi/4)
    z_traj = center[2] + r * np.cos(phi) * np.sin(np.pi/4)


    # 开启独立的控制器线程
    run_controller = True
    ctrl_thread = threading.Thread(target=pos_controller)
    ctrl_thread.start()

    # 循环体主要轨迹跟踪部分
    now_r = time.time()
    i = 0

    while i < N:
        # 计算和更新轨迹
        if time.time() - now_r >= 0.01:
            # 更新目标位置
            now_r = time.time()
            i += 1
        target_pos[0], target_pos[1], target_pos[2] = x_traj[i], y_traj[i], z_traj[i]
        custom_env.step(action=None)

    custom_env.close()

    # 随时间变化误差收敛情况
    # plt.figure(1)
    # plt.plot(err[:,0])
    # plt.plot(err[:,1])
    # plt.show()
