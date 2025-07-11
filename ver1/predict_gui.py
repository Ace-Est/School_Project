import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import xgboost as xgb
from datetime import datetime
import numpy as np

# 設定字體解決中文問題
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 載入資料
df = pd.read_csv('merged_hourly.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['weekday'] = df['Date'].dt.weekday
df['rain_percent'] = df['Rain']
df['avg_temperature'] = df['Temperature']
df['avg_humidity'] = df['Humidity']
df['hour'] = df['Hour']

features = ['avg_temperature', 'avg_humidity', 'rain_percent', 'weekday', 'hour', 'label']

# 載入模型
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

def compute_metrics(y_true, y_pred):
    y_true_safe = y_true.replace(0, 1)
    mape = (abs((y_true - y_pred) / y_true_safe)).mean() * 100
    rmse = (((y_true - y_pred) ** 2).mean()) ** 0.5
    return mape, rmse

def plot_and_table(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        messagebox.showerror("格式錯誤", "請輸入正確日期格式：YYYY-MM-DD")
        return

    subset = df[df['Date'].dt.date == date_obj.date()]
    if subset.empty:
        messagebox.showinfo("無資料", f"{date_str} 沒有資料")
        return

    X_subset = subset[features]
    y_true = subset['kpt']
    y_pred = model.predict(X_subset)

    mape, rmse = compute_metrics(y_true, y_pred)

    hours_full = np.arange(24)

    actual_dict = dict(zip(subset['hour'], y_true))
    pred_dict = dict(zip(subset['hour'], y_pred))

    actual_full = [actual_dict.get(h, np.nan) for h in hours_full]
    pred_full = [pred_dict.get(h, np.nan) for h in hours_full]

    for widget in frame_plot.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours_full, actual_full, marker='o', label='實際')
    ax.plot(hours_full, pred_full, marker='x', label='預測')

    ax.set_title(f"{date_str} 準確度: RMSE={rmse:.2f}")
    ax.set_xlabel("小時")
    ax.set_ylabel("KPT (耗電量-發電量)")

    ax.set_xticks(hours_full)
    ax.set_xlim(-0.5, 23.5)

    ax.set_ylim(-600, 2800)  # Y軸範圍改成 -600 ~ 2800
    ax.set_yticks(np.arange(-600, 2801, 200))  # 固定200刻度間距
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    for widget in frame_table.winfo_children():
        widget.destroy()

    table = ttk.Treeview(frame_table, columns=("hour", "actual", "predicted", "error"), show='headings', height=24)
    table.heading("hour", text="小時")
    table.heading("actual", text="實際")
    table.heading("predicted", text="預測")
    table.heading("error", text="誤差")

    sum_actual = 0
    sum_predicted = 0
    sum_error = 0

    for h in hours_full:
        a = actual_dict.get(h, np.nan)
        p = pred_dict.get(h, np.nan)
        err = a - p if (not np.isnan(a) and not np.isnan(p)) else np.nan
        display_a = round(a, 2) if not np.isnan(a) else ''
        display_p = round(p, 2) if not np.isnan(p) else ''
        display_err = round(err, 2) if not np.isnan(err) else ''
        table.insert("", "end", values=(h, display_a, display_p, display_err))

        if not np.isnan(a):
            sum_actual += a
        if not np.isnan(p):
            sum_predicted += p
        if not np.isnan(err):
            sum_error += err

    table.insert("", "end", values=("總和", round(sum_actual, 2), round(sum_predicted, 2), round(sum_error, 2)))
    table.pack(fill="both", expand=True)

root = tk.Tk()
root.title("用電預測 GUI - XGBoost")
root.geometry("1200x600")

frame_input = tk.Frame(root)
frame_input.pack(pady=10)

tk.Label(frame_input, text="請輸入日期 (YYYY-MM-DD):").pack(side=tk.LEFT)
entry_date = tk.Entry(frame_input)
entry_date.pack(side=tk.LEFT, padx=5)
tk.Button(frame_input, text="比對", command=lambda: plot_and_table(entry_date.get())).pack(side=tk.LEFT)

main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

frame_plot = tk.Frame(main_frame)
frame_plot.pack(side=tk.LEFT, fill="both", expand=True, padx=10, pady=10)

frame_table = tk.Frame(main_frame)
frame_table.pack(side=tk.RIGHT, fill="both", expand=False, padx=10, pady=10)

root.mainloop()
