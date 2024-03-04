'''
这个代码用于测试 位置控制
请提前将xml文件中的控制器换为位置控制器
主要用于测试阻尼最小二乘法的轨迹跟踪效果
'''

import matplotlib.pyplot as plt
import numpy as np

import impedance_franka_gym
# import stable_baselines3
from stable_baselines3 import A2C
import stable_baselines3.common.env_checker
import time
import pandas as pd
# 定义位置控制器
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 检查环境
    # stable_baselines3.common.env_checker.check_env(custom_env)

    # print(custom_env.observation_space.sample())
    custom_env.reset()
    # 初始化一个csv文件用作记录
    df_record_data = pd.DataFrame()
    df_record_data.to_csv('realtime_data.csv', mode='w', index= False, header=False)

    # 不改变对角矩阵模的大小，只改变单一关节的权重对关节总角度变化的影响
    for i in np.linspace(start=0.1, stop=5.0, num=19):
        custom_env.reset()
        df_record_data = pd.DataFrame(np.zeros(shape=(1,13)))
        df_record_data.to_csv('realtime_data.csv', mode='a', index=False, header=False)
        loop_time = time.time()
        while (time.time() - loop_time < 2.0):
            input_action = np.ones(7) * ((7 - i)/6.0)
            input_action[4] = i
            _, _, _, _, record_data = custom_env.step(action=input_action)
            df_record_data = pd.DataFrame([record_data])
            df_record_data.to_csv('realtime_data.csv', mode='a', index=False, header=False)