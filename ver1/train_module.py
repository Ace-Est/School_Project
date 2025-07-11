import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def train_and_save_model(csv_path='merged_hourly.csv', model_path='xgb_model.json'):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])

    df['weekday'] = df['Date'].dt.weekday
    df['rain_percent'] = df['Rain']
    df['avg_temperature'] = df['Temperature']
    df['avg_humidity'] = df['Humidity']
    df['hour'] = df['Hour']

    features = ['avg_temperature', 'avg_humidity', 'rain_percent', 'weekday', 'hour', 'label']
    X = df[features]
    y = df['kpt']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100

    print(f"訓練集 MAPE: {train_mape:.2f}%")
    print(f"驗證集 MAPE: {val_mape:.2f}%")

    model.save_model(model_path)
    print(f"模型已儲存至 {model_path}")

if __name__ == "__main__":
    train_and_save_model()
