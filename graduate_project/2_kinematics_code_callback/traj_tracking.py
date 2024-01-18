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

import mujoco as mj

# 定义位置控制器
def pos_controller():
    global custom_env, target_pos, target_quat

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
                                          target_pos=target_pos, target_quat=target_quat,
                                          regularization_strength=np.diag([1,1,1,1,1,1,1]),
                                          regularization_threshold=0.0001,
                                          max_steps=1)
        custom_env.data.ctrl = IKResult.qpos

        # print(custom_env.data.ctrl)

        elapsed_c = time.time() - now_c
        sleep_time_c = (1. / ctrl_rate) - elapsed_c
        if sleep_time_c > 0:
            time.sleep(sleep_time_c)


def pos_callback(model, data, target_pos, target_quat):
    IKResult = ik.qpos_from_site_pose(model, data, site_name="attachment_site",
                                      target_pos=target_pos, target_quat=target_quat,
                                      regularization_strength=np.diag([1, 1, 1, 1, 1, 1, 1]),
                                      regularization_threshold=0.0001,
                                      max_steps=1)
    data.ctrl = IKResult.qpos

    # data.ctrl[2] = (i/1000) - 0.5
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 重置环境
    custom_env.reset()
    time.sleep(0.01)

    # 得到site初始位置和姿态
    target_pos = copy.copy(custom_env.data.site_xpos[0][:3])
    target_quat = np.empty(4,dtype=float)
    mj.mju_mat2Quat(target_quat, custom_env.data.site_xmat[0])
    err = []

    # 给轨迹,空间圆形
    center = np.array(custom_env.data.site_xpos[0][:3])
    # center = np.array([0,0,0.5])
    N = 1000
    r = 0.1
    phi = np.linspace(0, 6*np.pi, N)
    x_traj = center[0] + r * np.cos(phi) * np.cos(-np.pi/4)
    y_traj = center[1] + r * np.sin(phi) * np.cos(-np.pi/4)
    z_traj = center[2] + r * np.cos(phi) * np.sin(-np.pi/4)
    target_traj = np.dstack((x_traj, y_traj, z_traj))
    target_traj = target_traj[0]

    # 循环体主要轨迹跟踪部分
    now_r = time.time()
    i = 0
    traj_tracking_record = custom_env.data.site_xpos[0][:3]

    while i < N-1:
        # 计算和更新轨迹
        if time.time() - now_r >= 0.01:
            # 更新目标位置
            now_r = time.time()
            i += 1
            # 控制器部分
            mj.set_mjcb_control(pos_callback(custom_env.model, custom_env.data, target_pos, target_quat))
            mj.get_mjcb_control()
        target_pos[0], target_pos[1], target_pos[2] = x_traj[i], y_traj[i], z_traj[i]
        if i % 10 == 0:
            traj_tracking_record = np.vstack((traj_tracking_record,custom_env.data.site_xpos[0][:3]))
        custom_env.step(action=None)

    # 关闭控制器线程和主环境线程
    run_controller = False
    custom_env.close()

    # 绘制跟踪效果
    fig_tracking = plt.figure()
    ax_tracking = plt.axes(projection='3d')
    ax_tracking.plot3D(x_traj, y_traj, z_traj, 'red')
    ax_tracking.scatter3D(traj_tracking_record[:,0], traj_tracking_record[:,1], traj_tracking_record[:,2], cmap='b')
    plt.show()
