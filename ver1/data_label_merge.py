import pandas as pd

# 讀取兩個 CSV
data_df = pd.read_csv("data.csv")
label_df = pd.read_csv("date_label.csv")

# 先確保兩邊的時間格式一致（這是關鍵）
data_df['Time'] = pd.to_datetime(data_df['Time'])
label_df['Time'] = pd.to_datetime(label_df['Time'])

# 使用 merge 合併資料（依照 Time 欄位）
merged_df = pd.merge(data_df, label_df, on='Time', how='left')

# 將結果儲存
merged_df.to_csv("merged.csv", index=False)
