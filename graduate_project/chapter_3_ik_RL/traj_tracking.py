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
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
import stable_baselines3.common.env_checker
import time
# 定义位置控制器
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 检查环境
    # stable_baselines3.common.env_checker.check_env(custom_env)

    # print(custom_env.observation_space.sample())
    custom_env.reset()
    #
    # while 1 :
    #     custom_env.step(action=np.ones(7))
    for i in range(2):
        custom_env.step(action=np.zeros(7))

    # model = DDPG("MlpPolicy", custom_env, verbose=1)
    # model.learn(total_timesteps=3)
    #
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(5):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")