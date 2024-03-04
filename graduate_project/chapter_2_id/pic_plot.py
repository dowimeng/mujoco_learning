import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
textsize = 14
# 绘制七关节角度变化图
df = pd.read_csv('realtime_data_id.csv', header=None)
# df = pd.read_csv('realtime_data_2.csv', header=None)
data = np.array(df.values)

plt.figure()
for i in range(7):
    plt.plot(data[1:,i+13])

plt.figure()
for i in [0,1,2,10,11,12]:
    plt.plot(data[1:,i])
plt.show()