import matplotlib.pyplot as plt
import numpy as np
import copy
import impedance_franka_gym
import inverse_kinematics as ik
import mujoco as mj
import threading
import time


# 定义位置控制器
def pos_controller(ctrl_rate):

    global custom_env, target_pos

    while run_controller:
        now_c = time.time()
        # 更新目标位置
        target_pos[0], target_pos[1], target_pos[2] = x_ref, y_ref, z_ref
        # 给到冗余运动学控制器进行逆运动学解算
        # data_copy = copy.copy(custom_env.data)
        IKResult = ik.qpos_from_site_pose(custom_env.model, data_copy, site_name="attachment_site", target_pos=target_pos)
        custom_env.data.ctrl = IKResult.qpos
        # print(custom_env.data.ctrl)

        elapsed_c = time.time() - now_c
        sleep_time_c = (1./ctrl_rate) - elapsed_c
        if sleep_time_c > 0:
            time.sleep(sleep_time_c)

def torque_controller(ctrl_rate):

    global custom_env

    while run_controller:
        now_c = time.time()

        # 力矩控制
        # custom_env.data.ctrl = copy.copy(custom_env.data.qfrc_bias)
        custom_env.data.ctrl[1] = -30
        custom_env.data.ctrl[3] = 20
        # custom_env.data.qfrc_actuator = copy.copy(custom_env.data.qfrc_bias)

        elapsed_c = time.time() - now_c
        sleep_time_c = (1./ctrl_rate) - elapsed_c
        if sleep_time_c > 0:
            time.sleep(sleep_time_c)

def impedance_controller(ctrl_rate):

    # 先尝试一下定点的阻抗控制
    global custom_env, target_pos

    D_d = np.eye(7) * 1
    K_d = np.eye(7) * 10

    while run_controller:
        now_c = time.time()

        # 得到当前质量矩阵 C矩阵-qfrc_bias 速度和位置差值 外力矩（可选）
        # custom_env.data.qM
        # custom_env.data.qfrc_bias
        # - custom_env.data.qvel
        IKResult = ik.qpos_from_site_pose(custom_env.model, data_copy, site_name="attachment_site", target_pos=target_pos)
        target_qpos = IKResult.qpos
        # tau是一个七维向量
        tau = custom_env.data.qfrc_bias + \
            D_d.dot(-custom_env.data.qvel) + K_d.dot(target_qpos - custom_env.data.qpos)

        # 力矩控制
        custom_env.data.ctrl = tau

        elapsed_c = time.time() - now_c
        sleep_time_c = (1./ctrl_rate) - elapsed_c
        if sleep_time_c > 0:
            time.sleep(sleep_time_c)


def traj_generator():
    global x_ref, y_ref, z_ref, target_pos, x_ref_traj, y_ref_traj, z_ref_traj, N

    # 轨迹部分
    N = 200
    # 获取初始末端位置
    position_Q = custom_env.data.site_xpos[0]
    # 定义末端轨迹
    r = 0.1
    center = np.array([position_Q[0] - r, position_Q[1], position_Q[2]])
    phi = np.linspace(0, 2 * np.pi, N)
    x_ref_traj = center[0] + r * np.cos(phi)
    y_ref_traj = center[1] + r * np.sin(phi)
    z_ref_traj= position_Q[2].__copy__()

    x_ref = x_ref_traj[0]
    y_ref = y_ref_traj[0]
    z_ref = copy.copy(position_Q[2])

    x_all = []
    y_all = []

    target_pos = position_Q

if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    ctrl_rate = 1/custom_env.model.opt.timestep
    render_rate = 100

    custom_env.reset()
    time.sleep(0.01)

    # traj_generator()

    target_pos =  copy.copy(custom_env.data.site_xpos[0])

    data_copy = copy.copy(custom_env.data)

    run_controller = True
    # ctrl_thread = threading.Thread(target=torque_controller, args=[ctrl_rate])
    ctrl_thread = threading.Thread(target=impedance_controller, args=[ctrl_rate])
    ctrl_thread.start()

    now_r = time.time()


    # 这部分用于位置控制测试
    # i = 0
    # while i < len(x_ref_traj):
    #     # 更新目标位置
    #     x_ref, y_ref, z_ref= x_ref_traj[i], y_ref_traj[i], z_ref_traj
    #     # 渲染
    #     elapsed_r = time.time() - now_r
    #     if elapsed_r >= 0.01:
    #         if i < N - 1 :
    #             i += 1
    #         else:
    #             i = 0
    #         now_r = time.time()
    #         # x_all.append(custom_env.data.site_xpos[0][0])
    #         # y_all.append(custom_env.data.site_xpos[0][1])

        # custom_env.step(action=None)

    # 这部分用于阻抗控制测试

    while time.time() - now_r < 10:
        if time.time() - now_r > 1 and time.time() - now_r < 3 :
            custom_env.data.qfrc_applied[3] = 20
            print("happend")
        else:
            custom_env.data.qfrc_applied = 0
            print("not happend")

        custom_env.step(action=None)

    # print(len(custom_env.data.qM))

    # plt.figure(1)
    # plt.plot(x_all,y_all,'bx')
    # plt.plot(x_ref_traj,y_ref_traj,'r-')
    # plt.show()

    while 1:
        custom_env.step(action=None)

        # custom_env.render()