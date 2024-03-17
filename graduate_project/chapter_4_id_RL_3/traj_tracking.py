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
# 定义位置控制器
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")
    custom_env = Monitor(custom_env, 'train')
    # 检查环境
    # stable_baselines3.common.env_checker.check_env(custom_env)

    custom_env.reset()
    while 1:
        action = np.zeros(5)
        _, reward, done, _, _ = custom_env.step(action=action)
        if done:
            custom_env.reset()

    # model = PPO("MlpPolicy", env=custom_env, verbose=1)
    # model.learn(total_timesteps=100000)
    # #
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # # while 1:
    # for i in range(1):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    #     if done:
    #         custom_env.reset()