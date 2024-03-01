import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 绘制七关节角度变化图
df = pd.read_csv('realtime_data.csv')
data = np.array(df)
loop_time = np.shape(data)[0]
length_time = []
for i in range(loop_time):
    length_time.append(i/60.0)

# 单位对角矩阵 7个关节角度变化
plt.figure(1)
for i in range(3,10):
    plt.plot(length_time, data[:, i])

# 单位对角矩阵 末端位置变化
plt.figure(2)
for i in range(3):
    plt.plot(length_time, data[:,i])

plt.show()
