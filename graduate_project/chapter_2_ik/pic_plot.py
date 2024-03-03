import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
textsize = 14
# 绘制七关节角度变化图
df = pd.read_csv('realtime_data.csv', header=None)
data = np.array(df.values)

# # 这部分分块 chunks存储的全零行的位置
chunks = np.where(np.all(data == 0, axis=1))[0]
chunks = np.append(chunks, np.shape(data)[0]+1)

# # 绘制所有图像
# for i in range(np.shape(chunks)[0] - 2):
#     print(i)
#     # 每一次循环的长度
#     length = chunks[i+1] - chunks[i] - 1
#     length_time = []
#     for j in range(length):
#         length_time.append(j/60.0)
#     # 七个关节角度的变化
#     plt.figure(1)
#     for j in range(3,10):
#         plt.plot(length_time, data[chunks[i]+1:chunks[i+1], j])
#     # 单位对角矩阵 末端位置变化
#     plt.figure(2)
#     for j in range(3):
#         plt.plot(length_time, data[chunks[i]+1:chunks[i+1], j])
#
# plt.show()

# 绘制单位对角矩阵下的关节角度变化图像并存储
i = 10
length = chunks[i+1] - chunks[i] - 2
length_time = []
for j in range(length):
    length_time.append(j / 60.0)
# 七个关节角度的变化
plt.figure(1)
for j in range(3, 10):
    plt.plot(length_time, data[chunks[i]+2:chunks[i+1], j])
plt.legend(['关节1', '关节2', '关节3', '关节4', '关节5', '关节6', '关节7'],
           loc = 'lower right', fontsize = textsize)
plt.xlabel('运行时间/s', fontsize = textsize)
plt.ylabel('关节角度/rad', fontsize = textsize)
plt.savefig('picture/I_joint_angles.png')

# 单位对角矩阵 末端位姿变化
fig = plt.figure()
ax2 = fig.add_subplot(111)
for j in range(3):
    ax2.plot(length_time, data[chunks[i]+2:chunks[i+1], j] )
# ax2.legend(['x轴坐标', 'y轴坐标', 'z轴坐标'])
ax2.set_ylabel('位置/m', fontsize = textsize)
ax2.set_xlabel('运行时间/s', fontsize = textsize)
# ax2.set_xticks(x_tics, fontsize = 12)
# ax2.set_yticks(y_tics, fontsize = 12)

ax3 = ax2.twinx()
for j in range(10, 13):
    ax3.plot(length_time, data[chunks[i]+2:chunks[i+1], j],  linestyle = '--')
ax3.set_ylabel('角度/rad', fontsize = textsize)

fig.legend(['x轴坐标', 'y轴坐标', 'z轴坐标', '偏航角yaw', '俯仰角pitch', '翻滚角roll'],
           loc = 'lower right', bbox_to_anchor=(0.9, 0.115), fontsize = textsize)

plt.savefig('picture/I_site_posquat.png')

plt.show()
