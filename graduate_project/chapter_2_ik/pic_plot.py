import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Source Han Sans CN']
plt.rcParams['axes.unicode_minus']=False

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
length = chunks[i+1] - chunks[i] - 1
length_time = []
for j in range(length):
    length_time.append(j / 60.0)
# 七个关节角度的变化
plt.figure(1)
for j in range(3,10):
    plt.plot(length_time, data[chunks[i]+1:chunks[i+1], j])
plt.legend(['关节1', '关节2', '关节3', '关节4', '关节5', '关节6', '关节7'])
plt.xlabel('运行时间/s')
plt.ylabel('关节角度/rad')
# 单位对角矩阵 末端位置变化
plt.figure(2)
for j in range(3):
    plt.plot(length_time, data[chunks[i]+1:chunks[i+1], j])
plt.show()