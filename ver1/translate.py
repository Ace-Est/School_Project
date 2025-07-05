import pandas as pd

def aggregate_to_hourly(input_csv='merged.csv', output_csv='merged_hourly.csv'):
    # 讀取原始15分鐘資料
    df = pd.read_csv(input_csv)

    # 轉換Time欄位為datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # 抽取日期（年月日）
    df['Date'] = df['Time'].dt.date

    # 確保Hour是整數
    df['Hour'] = df['Hour'].astype(int)

    # 定義groupby欄位
    group_cols = ['Date', 'Hour']

    # 聚合計算
    df_hourly = df.groupby(group_cols).agg({
        'kpt': 'sum',                # kpt累加
        'Temperature': 'mean',       # 溫度平均
        'Humidity': 'mean',          # 濕度平均
        'Rain': 'sum',               # Rain累加（最多1小時，單位小時）
        'day_off': 'first',          # day_off取第一筆
        'label': 'first'             # label取第一筆
    }).reset_index()

    # Rain最大值限制1（理論上不超過）
    df_hourly['Rain'] = df_hourly['Rain'].clip(upper=1)

    # 計算每一天的有效小時數（筆數）
    counts = df_hourly.groupby('Date').size().reset_index(name='hour_count')

    # 篩選只有小時數 == 24 的日期
    valid_dates = counts[counts['hour_count'] == 24]['Date']

    # 篩選完整天數的資料
    df_filtered = df_hourly[df_hourly['Date'].isin(valid_dates)]

    # 欄位排序
    df_filtered = df_filtered[['Date', 'kpt', 'Temperature', 'Humidity', 'Rain', 'day_off', 'Hour', 'label']]

    # 日期欄轉字串（可選）
    df_filtered['Date'] = df_filtered['Date'].astype(str)

    # 輸出csv
    df_filtered.to_csv(output_csv, index=False)
    print(f"已完成每小時資料彙整，且過濾不滿24小時的日期。輸出檔案：{output_csv}")

if __name__ == "__main__":
    aggregate_to_hourly()
