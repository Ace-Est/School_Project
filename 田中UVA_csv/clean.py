import pandas as pd
from datetime import datetime, timedelta
import os

# 資料夾路徑
folder = "./UVA_csv"  # <-- 改成你的資料夾路徑

# 日期範圍
start_date = datetime.strptime("2024-07-11", "%Y-%m-%d")
end_date = datetime.strptime("2025-05-25", "%Y-%m-%d")

# 欲保留欄位
columns_to_keep = ["ObsTime", "GloblRad", "UVI"]

# 所有資料儲存處
all_data = []

# 遍歷日期
while start_date <= end_date:
    date_str = start_date.strftime("%Y-%m-%d")
    filename = f"467270-{date_str}.csv"
    filepath = os.path.join(folder, filename)

    if os.path.exists(filepath):
        try:
            # 注意：header=1 表示使用第 2 行當欄位名稱（0-based index）
            df = pd.read_csv(filepath, header=1)

            # 檢查是否包含三個欄位
            if all(col in df.columns for col in columns_to_keep):
                filtered = df[columns_to_keep].copy()
                filtered.insert(0, "Date", date_str)
                all_data.append(filtered)
            else:
                print(f" 欄位缺失於 {filename}")
        except Exception as e:
            print(f" 無法處理 {filename}：{e}")
    else:
        print(f" 找不到檔案：{filename}")

    start_date += timedelta(days=1)

# 合併所有結果並儲存
if all_data:
    result = pd.concat(all_data, ignore_index=True)
    result.to_csv("合併結果.csv", index=False, encoding="utf-8-sig")
    print(" 合併完成，結果儲存為『合併結果.csv』")
else:
    print(" 沒有成功讀取的資料")
