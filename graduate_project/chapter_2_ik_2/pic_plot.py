import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
textsize = 14
# 绘制七关节角度变化图
df = pd.read_csv('realtime_data.csv', header=None)
# df = pd.read_csv('realtime_data_2.csv', header=None)
data = np.array(df.values)

# # 这部分分块 chunks存储的全零行的位置
chunks = np.where(np.all(data == 0, axis=1))[0]
chunks = np.append(chunks, np.shape(data)[0])
groups = np.shape(chunks)[0]

delta_joints = []
# 把每次执行完的关节差值记录下来
for i in range(groups - 1):
    if i == 0:
        delta_joints = data[chunks[i+1]-1, 3:10] - data[chunks[i]+1, 3:10]
    else:
        delta_joints = np.vstack((delta_joints, data[chunks[i+1]-1, 3:10] - data[chunks[i]+1, 3:10]))
delta_joints = np.absolute(delta_joints)
# 权重比例
weight_ratio = np.linspace(0.1, 5.0, 49)
weight_ratio = 6.0 * weight_ratio / (7.0 - weight_ratio)

plt.figure(figsize = (12,6))
for i in range(7):
    if i != 4:
        plt.plot(weight_ratio, delta_joints[:, i], '--')
    else:
        plt.plot(weight_ratio, delta_joints[:,i])
plt.legend(['关节1', '关节2', '关节3', '关节4', '关节5', '关节6', '关节7'],
           loc = 'lower right', fontsize = textsize)
plt.xlabel('关节%d的权重与其余关节权重的比值'%5, fontsize = textsize)
plt.ylabel('关节角度变化绝对值/rad', fontsize = textsize)
plt.savefig('picture/diff_weight_ratio.png')
plt.show()

# plt.figure()
# plt.plot(weight_ratio,delta_joints[:,3])
# plt.show()

# plt.figure()
# for j in range(3, 10):
#     plt.plot(length_time, data[chunks[i]+2:chunks[i+1], j])
# plt.legend(['关节1', '关节2', '关节3', '关节4', '关节5', '关节6', '关节7'],
#            loc = 'lower right', fontsize = textsize)
# plt.xlabel('运行时间/s', fontsize = textsize)
# plt.ylabel('关节角度/rad', fontsize = textsize)
# # plt.savefig('picture/I_joint_angles.png')
#
#
# i = 18
#
# length = chunks[i+1] - chunks[i] - 2
# length_time = []
# for j in range(length):
#     length_time.append(j / 60.0)
# # 七个关节角度的变化
# # fig, axes = plt.subplots(7,1)
#
# plt.figure()
# for j in range(3, 10):
#     plt.plot(length_time, data[chunks[i]+2:chunks[i+1], j])
# plt.legend(['关节1', '关节2', '关节3', '关节4', '关节5', '关节6', '关节7'],
#            loc = 'lower right', fontsize = textsize)
# plt.xlabel('运行时间/s', fontsize = textsize)
# plt.ylabel('关节角度/rad', fontsize = textsize)
# # plt.savefig('picture/I_joint_angles.png')
#
# plt.show()