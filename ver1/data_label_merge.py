import pandas as pd

# 1. 讀取 CSV 檔案
df = pd.read_csv('merged_hourly.csv')
date_label = pd.read_csv('date_label.csv')

# 2. 時間格式轉換
df['Time'] = pd.to_datetime(df['Time'])
date_label['Time'] = pd.to_datetime(date_label['Time'])

# 3. 刪除原本的 label_x 和 label_y（如果有）
df = df.drop(columns=['label'], errors='ignore')  # 先刪掉舊的
df = df.merge(date_label, on='Time', how='left')

# 4. 合併每日 label
df['label'] = df['label'].fillna(0).astype(int)


# 6. 儲存為 CSV（其餘欄位保留原格式）
df.to_csv('merged_hourly.csv', index=False)

# 7. 顯示前幾筆結果
print(df.head())
