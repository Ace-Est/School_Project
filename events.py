import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# 解決中文字顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取資料
df = pd.read_csv("input_0712_0310.csv")

# 2. 時間欄位處理
df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = df['Time'].dt.date
df['Hour'] = df['Time'].dt.hour
df['Weekday'] = df['Time'].dt.weekday
df['is_weekend'] = df['Weekday'].isin([5, 6])

# 3. 加入行事曆事件（其他因素）
event_dict = {
    'is_midterm': pd.date_range('2024-11-04', '2024-11-08').date,
    'is_holiday': [datetime(2024, 9, 17).date(), datetime(2024, 10, 10).date(), datetime(2025, 1, 1).date()],
    'is_school_event': [
        datetime(2024, 10, 26).date(),
        datetime(2024, 12, 5).date(),
        datetime(2024, 12, 7).date(), datetime(2024, 12, 8).date(),
        datetime(2024, 12, 15).date(),
    ]
}
df['is_midterm'] = df['Date'].isin(event_dict['is_midterm'])
df['is_holiday'] = df['Date'].isin(event_dict['is_holiday'])
df['is_school_event'] = df['Date'].isin(event_dict['is_school_event'])

def label_event(row):
    if row['is_midterm']:
        return '期中考週'
    elif row['is_holiday']:
        return '校定假日'
    elif row['is_school_event']:
        return '校園活動日'
    else:
        return '一般日'

df['event_type'] = df.apply(label_event, axis=1)

# 📊 圖一：各事件類型 Boxplot
# 計算每個事件類型的平均用電量
event_avg = df.groupby('event_type')['kpt'].mean().reset_index()

# 畫圖：事件類型 vs 平均用電
plt.figure(figsize=(8, 5))
barplot = sns.barplot(data=event_avg, x='event_type', y='kpt', palette="Set2")

# 在柱子上方加數字標註
for i, row in event_avg.iterrows():
    plt.text(i, row['kpt'] + 5, f"{row['kpt']:.1f}", ha='center', fontsize=12)

plt.title("各事件類型平均用電量")
plt.xlabel("事件類型")
plt.ylabel("平均用電負載 (kpt)")
plt.tight_layout()
plt.savefig("event_avg_kpt_barplot.png")
plt.close()

# 📈 圖二：每日平均用電趨勢（依事件類型）
daily_avg = df.groupby(['Date', 'event_type'])['kpt'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=daily_avg, x='Date', y='kpt', hue='event_type', marker='o', palette="Set2")
plt.title("每日平均用電趨勢（依事件類型）")
plt.xticks(rotation=45)
plt.ylabel("平均用電負載 (kpt)")
plt.xlabel("日期")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("daily_trend_event_fixed.png")
plt.close()


# 📈 圖三：每小時平均用電趨勢（加註高峰）
plt.figure(figsize=(12, 6))
hourly_avg = df.groupby('Hour')['kpt'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o')
plt.title("每小時平均用電趨勢")
plt.xlabel("小時")
plt.ylabel("用電負載 (kpt)")

peak_hour = hourly_avg.idxmax()
peak_value = hourly_avg.max()
plt.annotate(f'高峰：{peak_value:.2f} kpt', xy=(peak_hour, peak_value),
             xytext=(peak_hour+1, peak_value+0.2),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=12, color='red')
plt.grid(True)
plt.tight_layout()
plt.savefig("hourly_trend_annotated.png")
plt.close()

# 7. XGBoost 模型訓練（含天氣與事件特徵）
features = [
    'Hour', 'Weekday', 'is_weekend',
    'is_midterm', 'is_holiday', 'is_school_event',
    'Temperature', 'Humidity'
]
X = df[features]
y = df['kpt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📊 評估指標
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5


# 📊 圖四：XGBoost 特徵重要性
xgb.plot_importance(model)
plt.title("XGBoost 特徵重要性")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 9. 預測結果與誤差存檔 + 圖表
output_df = X_test.copy()
output_df['Actual_kpt'] = y_test
output_df['Predicted_kpt'] = y_pred
output_df['Error'] = output_df['Actual_kpt'] - output_df['Predicted_kpt']
output_df['AbsError'] = np.abs(output_df['Error'])

output_df.to_csv("prediction_result.csv", index=False)
pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).to_csv("feature_importance.csv", index=False)

# 📈 圖五：預測 vs 實際
# 計算誤差（如果還沒算過）
output_df['Error'] = output_df['Actual_kpt'] - output_df['Predicted_kpt']
output_df['AbsError'] = np.abs(output_df['Error'])

# 📈 圖五：加上誤差顏色漸層 + 評估數值顯示
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    output_df['Actual_kpt'], output_df['Predicted_kpt'],
    c=output_df['AbsError'], cmap='coolwarm', alpha=0.6
)
plt.plot(
    [output_df['Actual_kpt'].min(), output_df['Actual_kpt'].max()],
    [output_df['Actual_kpt'].min(), output_df['Actual_kpt'].max()],
    linestyle='--', color='gray', label='理想預測'
)
cbar = plt.colorbar(scatter)
cbar.set_label('絕對誤差 (kpt)')

# 評估指標標註
plt.text(0.05, 0.95, f"R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
         transform=plt.gca().transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

plt.xlabel("實際用電負載 (kpt)")
plt.ylabel("預測用電負載 (kpt)")
plt.title("實際 vs 預測 用電負載（含誤差標示）")
plt.tight_layout()
plt.savefig("scatter_actual_vs_pred_annotated.png")
plt.close()


# 📈 圖六：預測誤差分布
plt.figure(figsize=(10, 5))
sns.histplot(output_df['AbsError'], bins=30, kde=True)
plt.title("預測誤差（MAE）分布")
plt.xlabel("絕對誤差（|Actual - Predicted|）")
plt.ylabel("出現次數")
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.close()

# 📈 圖七：Temperature 與 Humidity 影響趨勢
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Temperature', y='kpt', alpha=0.5)
plt.title("溫度 vs 用電負載")
plt.xlabel("溫度 (℃)")
plt.ylabel("用電負載 (kpt)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Humidity', y='kpt', alpha=0.5, color='green')
plt.title("濕度 vs 用電負載")
plt.xlabel("濕度 (%)")
plt.ylabel("用電負載 (kpt)")

plt.tight_layout()
plt.savefig("temp_humidity_vs_kpt.png")
plt.close()

weekday_map = {
    0: '星期一', 1: '星期二', 2: '星期三', 3: '星期四',
    4: '星期五', 5: '星期六', 6: '星期日'
}
df['weekday'] = df['day_of_week'].map(weekday_map)

# 分組計算平均
grouped = df.groupby('weekday').mean(numeric_only=True).reindex(
    ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
)

# 繪製比較圖（實際 vs 預測 vs 誤差）
plt.figure(figsize=(12, 6))
sns.lineplot(data=grouped[['actual', 'predicted', 'error']])
plt.title("每週平均（實際、預測、誤差）")
plt.xlabel("星期")
plt.ylabel("平均數值")
plt.legend(['實際', '預測', '誤差'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

