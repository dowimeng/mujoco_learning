import pandas as pd
import numpy as np

data = [np.nan*4]
data2 = [5,6,7,8]
data3 = [9,10,11,12]
df_data = pd.DataFrame([data])
df_data2 = pd.DataFrame([data2])
df_data3 = pd.DataFrame([data3])

df_data.to_csv("realtime.csv", mode='w', index=False, header=False)

df_data2.to_csv("realtime.csv", mode='a', index=False, header=False)

df_data3.to_csv("realtime.csv", mode='a', index=False, header=False)

# 查询当前系统所有字体
from matplotlib.font_manager import FontManager
import subprocess

mpl_fonts = set(f.name for f in FontManager().ttflist)

print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)
