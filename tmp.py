import pandas as pd

# 讀入原始 CSV 檔
df = pd.read_csv("data.csv")  # 請將檔名換成你實際的檔名

# ✅ 刪除不要的欄位
df.drop(columns=["day_of_week", "Month", "Day"], inplace=True)

# ✅ 保留 Time 欄位中的「日期」部分（YYYY-MM-DD）
df["Time"] = pd.to_datetime(df["Time"]).dt.date.astype(str)

# ✅ 儲存成新的 CSV
df.to_csv("處理後.csv", index=False, encoding="utf-8-sig")

print("✅ 完成！Time 欄位已保留日期，其餘指定欄位已刪除。輸出為『處理後.csv』")
