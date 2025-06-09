import pandas as pd
import numpy as np

# 讀取 A 檔案
df_a = pd.read_csv('data.csv')
#print(np.mean(df_a['kpt']))
# 讀取 B 檔案
df_b = pd.read_csv('date_label.csv')

# 將欄位名稱對齊以便合併
df_b = df_b.rename(columns={"date": "Time"})

# 合併（依 time 對應 label）
df_merged = pd.merge(df_a, df_b, on="Time", how="left")

# 存成新檔案
df_merged.to_csv("merged.csv", index=False)
