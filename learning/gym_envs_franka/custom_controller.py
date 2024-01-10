# 定义一个控制器类型 使用mj.set_mjcb_control来用 可以传入阻抗参数
import inverse_kinematics as ik

# 一个位置控制器
def pos_controller(model, data, site_name, target_pos):

    data_copy = data.__copy__()
    IKResult = ik.qpos_from_site_pose(model, data_copy, site_name, target_pos=target_pos)
    data.ctrl = IKResult.qpos



# 一个力矩控制器
def torque_controller(self):
    pass


# 一个阻抗控制器
def impedance_controller(self):
    pass
