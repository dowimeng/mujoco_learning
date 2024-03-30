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
import time
import pandas as pd
# 定义位置控制器
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")
    # custom_env = Monitor(custom_env, 'train')
    # 检查环境
    # stable_baselines3.common.env_checker.check_env(custom_env)
    custom_env.reset()
    spring_range = np.linspace(start=10, stop=1000, num=20)
    damp_range = np.linspace(start=1, stop=100, num=20)

    for i in spring_range:
        for j in damp_range:
            action = np.array([i, j])
            done = False
            custom_env.reset()
            while not done:
                _, reward, done, _, _ = custom_env.step(action=action)