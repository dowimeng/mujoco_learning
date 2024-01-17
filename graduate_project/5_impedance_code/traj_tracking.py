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
# 定义位置控制器
if __name__ == "__main__":

    custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")

    # 检查环境
    # stable_baselines3.common.env_checker.check_env(custom_env)

    # print(custom_env.observation_space.sample())
    custom_env.reset()

    while 1 :
        custom_env.step(action=np.array([1,1,1,1,1,1,1]))

    # model = A2C("MlpPolicy", custom_env, verbose=1)
    # model.learn(total_timesteps=10000)
    #
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # print(obs)
    # for i in range(1000):
    #     # print(1)
    #     action, _state = model.predict(obs, deterministic=True)
    #     # print(2)
    #     obs, reward, done, info = vec_env.step(action)
    #     # print(3)
    #     print(action, reward)
    #     vec_env.render("human")