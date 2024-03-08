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
    while 1 :
    #     # custom_env.step(action=np.array([-0.98,-0.98,-0.98,-0.98,-0.98,-0.98,-0.98,
    #     #                                  -0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8]))
        custom_env.step(action=np.zeros(14))

    # input_action = np.zeros(14)
    # loop_time = time.time()
    #
    # f_joint = 2
    # while (time.time() - loop_time < 60):
    #     _, _, _, _, record_data = custom_env.step(action=input_action)
    #     if (10 < time.time() - loop_time < 12):
    #         custom_env.data.qfrc_applied[f_joint] = 20
    #         # custom_env.data.xfrc_applied[0] = 100
    #     else:
    #         custom_env.data.qfrc_applied[f_joint] = 0

    # model = PPO("MlpPolicy", custom_env, verbose=1)
    #
    # model.learn(total_timesteps=3600)
    #
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")