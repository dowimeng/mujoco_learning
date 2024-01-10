'''
用于测试力矩控制
请提前将xml文件中的控制器换为motor控制器
'''
import copy
import threading
import time

import numpy as np

import impedance_franka_gym
import inverse_kinematics as ik


def impedance_controller():
    # 先尝试一下定点的阻抗控制
    global custom_env, target_pos

    # 控制频率设置
    ctrl_rate = 1 / custom_env.model.opt.timestep
    # 用于冗余位置控制计算inverse_kinematics使用
    data_copy = copy.copy(custom_env.data)

    # 设置阻抗控制参数 D矩阵和K矩阵
    D_d = np.eye(7) * 10
    K_d = np.eye(7) * 100

    while run_controller:
        now_c = time.time()

        # 得到当前质量矩阵 C矩阵-qfrc_bias 速度和位置差值 外力矩（可选）
        # custom_env.data.qM
        # custom_env.data.qfrc_bias
        # - custom_env.data.qvel
        data_copy.qpos = copy.copy(custom_env.data.qpos)
        IKResult = ik.qpos_from_site_pose(custom_env.model, data_copy, site_name="attachment_site",
                                          target_pos=target_pos)
        target_qpos = IKResult.qpos
        # tau是一个七维向量
        tau = custom_env.data.qfrc_bias + \
              D_d.dot(-custom_env.data.qvel) + K_d.dot(target_qpos - custom_env.data.qpos)

        # 力矩控制
        custom_env.data.ctrl = tau

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
    target_pos = copy.copy(custom_env.data.site_xpos[0])

    # 开启独立的控制器线程
    run_controller = True
    ctrl_thread = threading.Thread(target=impedance_controller)
    ctrl_thread.start()

    # 这部分用于阻抗控制测试
    now_r = time.time()
    while time.time() - now_r < 10:
        if time.time() - now_r > 1 and time.time() - now_r < 3:
            custom_env.data.qfrc_applied[3] = 20
        else:
            custom_env.data.qfrc_applied = 0

        custom_env.step(action=None)

    while 1:
        custom_env.step(action=None)

        # custom_env.render()
