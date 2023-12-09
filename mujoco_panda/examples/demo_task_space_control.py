import os
import time
import threading
import mujoco_py
import quaternion
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.tf import quatdiff_in_euler
from mujoco_panda.utils.viewer_utils import render_frame

"""
Simplified demo of task-space control using joint torque actuation.

Robot moves its end-effector 10cm upwards (+ve Z axis) from starting position.
"""

# --------- Modify as required ------------
# Task-space controller parameters
# 此处设置的阻抗参数
# stiffness gains
P_pos = 1500.
P_ori = 200.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 1.
# -----------------------------------------

# 计算PD控制的六维力
def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg):

    # 计算差值
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])

    # 计算阻抗控制力
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3, 1]),
                   D_ori*(curr_omg).reshape([3, 1])])

    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)

    return F, error


def controller_thread(ctrl_rate):
    global p, target_pos

    threshold = 0.005

    # 当前末端执行器位置作为目标位置
    target_pos = curr_ee.copy()
    while run_controller:

        error = 100.
        while error > threshold:
            now_c = time.time()
            # 获取当前末端执行器的位置和姿态
            curr_pos, curr_ori = p.ee_pose()
            # 获取当前末端执行器的速度
            curr_vel, curr_omg = p.ee_velocity()

            # 更新目标位置Z值
            target_pos[2] = z_target
            # 计算阻抗控制的力和误差
            F, error = compute_ts_force(
                curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg)

            # 误差小于阈值 退出循环
            if error <= threshold:
                break
            
            # 计算阻抗控制加速度
            impedance_acc_des = np.dot(p.jacobian().T, F).flatten().tolist()
            
            # 为关节设置阻抗控制加速度
            p.set_joint_commands(cmd=impedance_acc_des, compensate_dynamics=True)

            # 执行一步模拟
            p.step(render=False)

            # 控制循环时间
            elapsed_c = time.time() - now_c
            sleep_time_c = (1./ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                time.sleep(sleep_time_c)


if __name__ == "__main__":

    p = PandaArm.withTorqueActuators(render=True, compensate_gravity=True)

    ctrl_rate = 1/p.model.opt.timestep
    # 设定渲染频率
    render_rate = 100

    # 机械臂设为中立姿态
    p.set_neutral_pose()
    p.step()
    time.sleep(0.01)

    # 获取当前机械臂关节角度
    new_pose = p._sim.data.qpos.copy()[:7]

    # 当前末端执行器位姿
    curr_ee, original_ori = p.ee_pose()

    # 设定目标Z轴位置
    target_z_traj = np.linspace(curr_ee[2], curr_ee[2]+0.1, 25).tolist()
    z_target = curr_ee[2]

    # 初始化目标位置
    target_pos = curr_ee
    # 开启控制线程 此时控制线程和仿真线程独立
    run_controller = True
    ctrl_thread = threading.Thread(target=controller_thread, args=[ctrl_rate])
    ctrl_thread.start()

    now_r = time.time()
    i = 0
    while i < len(target_z_traj):
        # 更新目标Z轴位置
        z_target = target_z_traj[i]

        # 获取当前末端执行器位置姿态  并渲染
        robot_pos, robot_ori = p.ee_pose()
        elapsed_r = time.time() - now_r

        if elapsed_r >= 0.1:
            i += 1
            now_r = time.time()
        render_frame(p.viewer, robot_pos, robot_ori)
        render_frame(p.viewer, target_pos, original_ori, alpha=0.2)

        p.render()

    print("Done controlling. Press Ctrl+C to quit.")

    # 一直渲染 直到用户手动退出
    while True:
        robot_pos, robot_ori = p.ee_pose()
        render_frame(p.viewer, robot_pos, robot_ori)
        render_frame(p.viewer, target_pos, original_ori, alpha=0.2)
        # p._sim.data.xfrc_applied[4][0] = 10  #加外力
        p.render()

    run_controller = False
    ctrl_thread.join()
