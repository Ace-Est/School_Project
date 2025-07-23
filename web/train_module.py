import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def train_and_save_model(csv_path="hourly_aggregated.csv"):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    df["weekday"] = df["Date"].dt.weekday
    df["rain_percent"] = df["Rain"]
    df["avg_temperature"] = df["Temperature"]
    df["avg_humidity"] = df["Humidity"]
    df["hour"] = df["Hour"]

    # 計算 use = kpt + pv
    df["use"] = df["kpt"] + df["pv"]

    features_use = [
        "avg_temperature",
        "avg_humidity",
        "rain_percent",
        "weekday",
        "hour",
        "label"
    ]
    features_pv = [
        "avg_temperature",
        "avg_humidity",
        "rain_percent",
        "hour"
    ]
    X_use = df[features_use]
    X_pv = df[features_pv]
    # 訓練 pv 時，夜間強制設為 0
    df_pv = df.copy()
    df_pv.loc[(df_pv['hour'] < 6) | (df_pv['hour'] > 17), 'pv'] = 0
    targets = {"use": "use", "pv": "pv"}
    for name, target in targets.items():
        if name == "pv":
            X = X_pv
            y = df_pv[target]
        else:
            X = X_use
            y = df[target]
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_percentage_error
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
        print(f"[{name}] 訓練集 MAPE: {train_mape:.2f}%")
        print(f"[{name}] 驗證集 MAPE: {val_mape:.2f}%")
        model_path = f"xgb_{name}.json"
        model.save_model(model_path)
        print(f"[{name}] 模型已儲存至 {model_path}")


if __name__ == "__main__":
    train_and_save_model()
