import pandas as pd
data = [1,2,3,4]
data2 = [5,6,7,8]
data3 = [9,10,11,12]
df_data = pd.DataFrame([data])
df_data2 = pd.DataFrame([data2])
df_data3 = pd.DataFrame([data3])

df_data.to_csv("realtime.csv", mode='w', index=False, header=False)

df_data2.to_csv("realtime.csv", mode='a', index=False, header=False)

df_data3.to_csv("realtime.csv", mode='a', index=False, header=False)
