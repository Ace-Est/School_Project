#2024-9~2025-3
import pandas as pd
"""

# 讀取 CSV 檔
df = pd.read_csv("縣市(彰化縣)小時值-每小時 (2024-09).csv")

# 只保留特定欄位（舉例：'姓名', '成績'）
filtered = df[
    (df["siteid"] == 33) &
    (df["itemname"].isin(["雨量", "懸浮微粒", "溫度", "相對溼度"]))
]
# 儲存成新檔
filtered.to_csv("2024-09.csv", index=False)
"""
df = pd.read_csv("縣市(彰化縣)小時值-每小時 (2024-09).csv")

# 只保留特定欄位（舉例：'姓名', '成績'）
filtered = df[
    (df["siteid"] == 33) &
    (df["itemname"].isin(["雨量", "懸浮微粒", "溫度", "相對溼度"]))
]
# 儲存成新檔
filtered.to_csv("2024-09.csv", index=False)