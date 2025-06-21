import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ========== 1. 讀取資料 ==========
data_df = pd.read_csv("./merged.csv")
weather_df = pd.read_csv("./weather.csv")

# ========== 2. 預處理電力資料 ==========
data_df["Datetime"] = pd.to_datetime(data_df["Time"] + " " + data_df["Hour"].astype(str) + ":00")

# ========== 3. 預處理天氣資料 ==========
weather_df["hr"] = weather_df["hr"].astype(int)
weather_df["Date"] = pd.to_datetime(weather_df["Date"], format="%Y-%m-%d")
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

# 選擇用來訓練的欄位
features = ['kpt', 'Temperature', 'Humidity', 'Rain', 'day_off', 'Hour', 'Minute']
target_col = 'kpt'

# ---------- 2. 特徵標準化 ----------
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(merged_df[features])

# ---------- 3. 建立時間序列資料 ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback):
        self.X, self.y = [], []
        for i in range(lookback, len(data)):
            self.X.append(data[i-lookback:i])
            self.y.append(data[i][0])  # kpt 為第 0 個欄位
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

lookback = 14
dataset = TimeSeriesDataset(scaled_features, lookback)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------- 4. 建立 LSTM 模型 ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最後時間點輸出
        return self.fc(out)

model = LSTMModel(input_size=len(features), hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------- 5. 訓練模型 ----------
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# ---------- 6. 預測下一筆 ----------
model.eval()
with torch.no_grad():
    last_sequence = scaled_features[-lookback:]  # 最後5筆資料
    input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)  # shape = (1, 5, input_size)
    predicted_scaled = model(input_tensor).item()

    # 將預測值還原為原始尺度
    dummy = np.zeros(len(features))
    dummy[0] = predicted_scaled  # 只還原 kpt，其餘為 0
    predicted_original = scaler.inverse_transform([dummy])[0][0]

    print(f"\n 預測下一筆 kpt（原始尺度）: {predicted_original:.2f}")
