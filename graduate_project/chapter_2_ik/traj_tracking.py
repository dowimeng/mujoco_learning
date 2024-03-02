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

    # 仿真一个单位矩阵
    # loop_time = time.time()
    # while (time.time() - loop_time < 3.0) :
    #     _,_,_,_,record_data = custom_env.step(action=np.array([1,1,1,1,1,1,1]))
    #     df_record_data = pd.DataFrame([record_data])
    #     df_record_data.to_csv('realtime_data.csv', mode='a', index= False, header=False)

    # 不同倍数的对角矩阵对关节总角度变化的影响
    # weight_realtime_data
    for i in np.linspace(start=-1.0, stop=1.0, num=20):
        custom_env.reset()
        df_record_data = pd.DataFrame(np.zeros(shape=(1,10)))
        df_record_data.to_csv('realtime_data.csv', mode='a', index=False, header=False)
        loop_time = time.time()
        while (time.time() - loop_time < 3.0):
            _, _, _, _, record_data = custom_env.step(action=np.array([1, 1, 1, 1, 1, 1, 1]) * 10 ** i)
            df_record_data = pd.DataFrame([record_data])
            df_record_data.to_csv('realtime_data.csv', mode='a', index=False, header=False)