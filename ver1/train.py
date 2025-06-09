import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score

# 設定中文顯示
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 改為適合你系統的字型
matplotlib.rcParams['axes.unicode_minus'] = False


# ========== 1. 讀取資料 ==========
data_df = pd.read_csv("./merged.csv")
weather_df = pd.read_csv("./weather.csv")

# ========== 2. 預處理電力資料 ==========
data_df["Datetime"] = pd.to_datetime(data_df["Time"] + " " + data_df["Hour"].astype(str) + ":00")

# ========== 3. 預處理天氣資料 ==========
weather_df["hr"] = weather_df["hr"].astype(int)
weather_df["Date"] = pd.to_datetime(weather_df["Date"], format="%Y/%m/%d")
weather_df.loc[weather_df["hr"] == 24, "Date"] += pd.Timedelta(days=1)
weather_df.loc[weather_df["hr"] == 24, "hr"] = 0
weather_df["Datetime"] = pd.to_datetime(weather_df["Date"].dt.date.astype(str) + " " + weather_df["hr"].astype(str) + ":00")

# ========== 4. 合併資料 ==========
merged_df = pd.merge(
    data_df,
    weather_df[["Datetime", "GloblRad", "UVI"]],
    on="Datetime",
    how="inner"
)
merged_df["GloblRad"] = pd.to_numeric(merged_df["GloblRad"], errors='coerce')
merged_df["UVI"] = pd.to_numeric(merged_df["UVI"], errors='coerce')

#merged_df.to_csv("test.csv", index=False)

# ========== 5. 建立訓練資料集 ==========
features = [
    "Temperature", "Humidity", "Rain", "day_off",
    "Hour", "Minute", "GloblRad", "UVI","label"
]
X = merged_df[features]
y = merged_df["kpt"]
X = X.dropna()
y = y.loc[X.index]

# ========== 6. 切分資料集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ========== 7. 建立並訓練 XGBoost 模型 ==========
model = XGBRegressor(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.13,
    gamma=1,                 # 控制分裂門檻
    reg_alpha=1,          # L1 正則化
    reg_lambda=7,            # L2 正則化
    random_state=42,
    objective='reg:squarederror',
    eval_metric='rmse',
)

model.fit(X_train, y_train)

# ========== 8. 預測與評估 ==========



# 預測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 訓練集評估
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

# 測試集評估
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

# 輸出結果
print(f"訓練集 ➤ MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
print(f"測試集 ➤ MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = (-scores) ** 0.5
print("交叉驗證 RMSE 平均：", rmse_scores.mean())

# ========== 9. 畫出實際 vs 預測圖（測試集與訓練集） ==========

plt.figure(figsize=(16, 6))

# 訓練資料視覺化
plt.subplot(1, 2, 1)
plt.plot(y_train.values[:200], label="True", linewidth=2)
plt.plot(y_train_pred[:200], label="Pred", linestyle='--')
plt.title("訓練集前200筆：真實值 vs 預測值")
plt.xlabel("樣本編號")
plt.ylabel("kpt")
plt.legend()

# 測試資料視覺化
plt.subplot(1, 2, 2)
plt.plot(y_test.values[:200], label="True", linewidth=2)
plt.plot(y_test_pred[:200], label="Pred", linestyle='--')
plt.title("測試集前200筆：真實值 vs 預測值")
plt.xlabel("樣本編號")
plt.ylabel("kpt")
plt.legend()

plt.tight_layout()
plt.show()

