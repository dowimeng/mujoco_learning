import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
textsize = 14
# 绘制七关节角度变化图
df = pd.read_csv('realtime_data.csv', header=None)
data = np.array(df.values)

# # 这部分分块 chunks存储的全零行的位置
chunks = np.where(np.all(data == 0, axis=1))[0]
chunks = np.append(chunks, np.shape(data)[0]+1)
length = np.shape(chunks)[0]
# print(length)
# 存储结束时关节末端位置
joints_pos = []
for i in range(1, length - 1 ):
    # joints_pos = [joints_pos, data[chunks[i] - 1, 3:10]]
    joints_pos.append(data[chunks[i] - 1, 3:10])
# print(joints_pos)

fig = plt.figure()


ax1 = plt.subplot(311)
lie = 0
for i in [0, 10, 16, 17]:
    length = chunks[i + 1] - chunks[i] - 2
    length_time = []
    for j in range(length):
        length_time.append(j / 60.0)
    ax1.plot(length_time, data[chunks[i]+2:chunks[i+1], lie])
    ax1.set_ylabel('x轴坐标/m', fontsize = textsize)
    ax1.set_xlabel('运行时间/s', fontsize = textsize)
    ax1.axhline(0, linestyle = '-.')

ax2 = plt.subplot(312)
lie = 1
for i in [0, 10, 16, 17]:
    length = chunks[i + 1] - chunks[i] - 2
    length_time = []
    for j in range(length):
        length_time.append(j / 60.0)
    ax2.plot(length_time, data[chunks[i]+2:chunks[i+1], lie])
    ax2.set_ylabel('y轴坐标/m', fontsize = textsize)
    ax2.set_xlabel('运行时间/s', fontsize = textsize)

ax3 = plt.subplot(313)
lie = 2
for i in [0, 10, 16, 17]:
    length = chunks[i + 1] - chunks[i] - 2
    length_time = []
    for j in range(length):
        length_time.append(j / 60.0)
    ax3.plot(length_time, data[chunks[i]+2:chunks[i+1], lie])
    ax3.set_ylabel('z轴坐标/m', fontsize = textsize)
    ax3.set_xlabel('运行时间/s', fontsize = textsize)


    plt.legend(['0,1', '1', '3.98', '5.01'])
plt.show()
# 图例，尺寸，坐标title，曲线格式