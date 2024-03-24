'''
这个代码用于测试 位置控制
请提前将xml文件中的控制器换为位置控制器
主要用于测试阻尼最小二乘法的轨迹跟踪效果
'''

import matplotlib.pyplot as plt
import numpy as np

import impedance_franka_gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker
import pandas as pd
# 定义位置控制器
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")
    # 检查环境
    # stable_baselines3.common.env_checker.check_env(custom_env)

    # action = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    # 第2个关节的情况
    i = 1
    for j in range(10000):
        print(j)
        joint, a, joints_pos_init, spring_big, damp_big, spring_small, damp_small = custom_env.reset(joint=i)
        traj_joint, record_joint_qpos_before, record_joint_qvel, record_joint_qpos = custom_env.step()

        joint = joint * np.ones(200)
        a = a * np.ones(200)
        spring_big = spring_big * np.ones(200)
        damp_big = damp_big * np.ones(200)
        spring_small = spring_small * np.ones(200)
        damp_small = damp_small * np.ones(200)

        record = np.dstack((joint, a))
        record = np.dstack((record, spring_big))
        record = np.dstack((record, damp_big))
        record = np.dstack((record, spring_small))
        record = np.dstack((record, damp_small))
        record = np.dstack((record, traj_joint))
        record = np.dstack((record, record_joint_qpos_before))
        record = np.dstack((record, record_joint_qvel))
        record = np.dstack((record, record_joint_qpos))

        df = pd.DataFrame(record[0])
        df.to_csv('test.csv', mode='a', header=False, index=False)
