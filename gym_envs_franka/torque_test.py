'''
用于测试力矩控制
请提前将xml文件中的控制器换为motor控制器
'''

import matplotlib.pyplot as plt
import numpy as np
import copy
import impedance_franka_gym
import inverse_kinematics as ik
import mujoco as mj
import threading
import time

def torque_controller():

    global custom_env

    # 控制频率设置
    ctrl_rate = 1/custom_env.model.opt.timestep

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


if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 重置环境
    custom_env.reset()
    time.sleep(0.01)

    # 得到site初始位置
    target_pos =  copy.copy(custom_env.data.site_xpos[0])

    # 开启独立的控制器线程
    run_controller = True
    ctrl_thread = threading.Thread(target=torque_controller)
    ctrl_thread.start()

    while 1:
        custom_env.step(action=None)

        # custom_env.render()