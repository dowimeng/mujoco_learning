import matplotlib.pyplot as plt
import numpy as np
import copy
import impedance_franka_gym
import inverse_kinematics as ik
import threading

custom_env = impedance_franka_gym.ImpedanceFrankaGym(render_mode="human")
custom_env.reset()

# 定义位置控制器
def pos_controller(model, data, site_name, target_pos):

    data_copy = data.__copy__()
    IKResult = ik.qpos_from_site_pose(model, data_copy, site_name, target_pos=target_pos)
    data.ctrl = IKResult.qpos


N = 100

# 获取初始末端位置
position_Q = custom_env.data.site_xpos[0]
# print(position_Q)

# 定义末端轨迹
r = 0.1
center = np.array([position_Q[0] - r, position_Q[1], position_Q[2]])
phi = np.linspace(0, 2 * np.pi, N)
x_ref = center[0] + r * np.cos(phi)
y_ref = center[1] + r * np.sin(phi)
z_ref = position_Q[2].__copy__()

x_all = []
y_all = []

i = 0
pre_time = copy.copy(custom_env.data.time)

if __name__ == "__main__":
    pass

# while 1:
#
#     # 给定控制器频率
#     if custom_env.data.time - pre_time > 0.001:
#         target_pos = np.array([x_ref[i], y_ref[i], z_ref])
#         # cc.pos_controller(custom_env.model, custom_env.data, site_name="attachment_site", target_pos=target_pos)
#         i += 1
#         pre_time = copy.copy(custom_env.data.time)
#
#     custom_env.step(action=None)
#
#     x_all.append(custom_env.data.site_xpos[0][0])
#     y_all.append(custom_env.data.site_xpos[0][1])
#
#     if i < N - 1:
#         i += 1
#
#     else:
#         plt.figure(1)
#         plt.plot(x_all, y_all, 'bx')
#         plt.plot(x_ref, y_ref, 'r-')
#         plt.show()
