import os
import pandas as pd
from flask import Flask, render_template, request, jsonify,redirect, url_for,send_from_directory

import xgboost as xgb
from werkzeug.utils import secure_filename
import tempfile
import io

app = Flask(__name__, template_folder="templates")# 資料與模型路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "hourly_aggregated.csv")
# 移除 xgb_model.json 相關程式碼，只保留 xgb_kpt.json
# MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")

# 契約容量設定 (kWh/小時)
CONTRACT_CAPACITY = 785  # 契約容量為 785 kWh/小時

# 全域變數
_df = None
_model = None
latest_upload_payload = None

# 僅載入 use 與 pv 兩個模型
use_model = None
pv_model = None

def load_all_models():
    global use_model, pv_model
    import xgboost as xgb
    use_model = xgb.XGBRegressor()
    use_model.load_model("xgb_use.json")
    pv_model = xgb.XGBRegressor()
    pv_model.load_model("xgb_pv.json")

# 允許的檔案類型
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data():
    global _df
    print(f"正在載入資料檔案: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"錯誤:找不到資料檔案 {DATA_PATH}")
        return False
    
    _df = pd.read_csv(DATA_PATH)
    print(f"成功載入資料，共 {len(_df)} 筆記錄")
    
    # 統一日期格式為 YYYY-MM-DD
    _df["Date_str"] = _df["Date"].apply(lambda d: d.replace("/", "-") if "/" in d else d)
    _df["Date_str"] = pd.to_datetime(_df["Date_str"]).dt.strftime("%Y-%m-%d")
    return True

def load_model():
    pass  # 不再使用 xgb_model.json

def get_features(df, target="use"):
    features_df = df.copy()
    features_df['avg_temperature'] = df['Temperature']
    features_df['avg_humidity'] = df['Humidity']
    features_df['rain_percent'] = df['Rain']
    features_df['weekday'] = df['day_week']
    features_df['hour'] = df['Hour']
    features_df['label'] = df['label']
    if target == "pv":
        features = ['avg_temperature', 'avg_humidity', 'rain_percent', 'hour']
    else:
        features = ['avg_temperature', 'avg_humidity', 'rain_percent', 'weekday', 'hour', 'label']
    return features_df[features]

# 契約容量檢查函數
def check_contract_capacity(use_pred, pv_pred, kpt_pred, battery_charge=None, battery_discharge=None):
    """
    檢查每小時的總用電量是否超過契約容量
    總用電量 = 購電量 + 消耗 - 放電 - 發電量
    其中：購電量 = kpt + 充電量 - 放電量
    """
    contract_violations = []
    total_power_usage = []
    
    # 初始化蓄電池參數
    if battery_charge is None:
        battery_charge = [0] * len(use_pred)
    if battery_discharge is None:
        battery_discharge = [0] * len(use_pred)
    
    for i, (use, pv, kpt) in enumerate(zip(use_pred, pv_pred, kpt_pred)):
        # 計算總用電量（從電網購電）
        # 總用電量 = kpt + 充電量 - 放電量
        charge = battery_charge[i] if i < len(battery_charge) else 0
        discharge = battery_discharge[i] if i < len(battery_discharge) else 0
        
        total_power = kpt + charge - discharge
        
        total_power_usage.append(total_power)
        
        # 檢查是否超過契約容量
        if total_power > CONTRACT_CAPACITY:
            contract_violations.append({
                'hour': i,
                'total_power': total_power,
                'excess': total_power - CONTRACT_CAPACITY,
                'kpt': kpt,
                'charge': charge,
                'discharge': discharge,
                'use': use,
                'pv': pv
            })
    
    return {
        'contract_capacity': CONTRACT_CAPACITY,
        'total_power_usage': total_power_usage,
        'violations': contract_violations,
        'max_power': max(total_power_usage) if total_power_usage else 0,
        'min_power': min(total_power_usage) if total_power_usage else 0,
        'avg_power': sum(total_power_usage) / len(total_power_usage) if total_power_usage else 0
    }

@app.route("/")
def index():
    return render_template("prediction.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/battery")
def battery():
    return render_template("battery.html")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Keep track of the latest uploaded file
latest_file = None

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    global latest_file
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            # Delete old file if it exists
            if latest_file:
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], latest_file))
                except FileNotFoundError:
                    pass
            # Save new file
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
            latest_file = file.filename
            return redirect(url_for("history"))
    return render_template("upload.html")

# --- Serve uploaded files ---
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# --- Make latest_file available in all templates ---
@app.context_processor
def inject_latest_file():
    return dict(latest_file=latest_file)

@app.route("/dates")
def get_dates():
    try:
        if _df is None:
            if not load_data():
                return jsonify([])
        
        if _df is None:
            return jsonify([])
            
        dates = sorted(_df["Date_str"].unique())
        print(f"可查詢日期: {len(dates)} 個")
        return jsonify(dates)
        
    except Exception as e:
        print(f"取得日期列表錯誤: {str(e)}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

@app.route("/latest_predict")
def latest_predict():
    try:
        if _df is None:
            if not load_data():
                return jsonify({"error": "資料未載入"}), 500
        if use_model is None or pv_model is None:
            load_all_models()
        last_24 = _df.tail(24).copy()
        if len(last_24) < 24:
            return jsonify({"error": "資料不足24筆"}), 400
        X_use = get_features(last_24, "use")
        X_pv = get_features(last_24, "pv")
        use_pred = use_model.predict(X_use)
        pv_pred = pv_model.predict(X_pv)
        # 將 pv 負值設為 0
        pv_pred = [max(0, float(v)) for v in pv_pred]
        kpt_pred = [float(v) - float(pv_pred[i]) for i, v in enumerate(use_pred)]
        hours = last_24["Hour"].tolist()
        
        # 檢查契約容量
        contract_check = check_contract_capacity(use_pred, pv_pred, kpt_pred)
        
        return jsonify({
            "hours": hours,
            "use_pred": [round(float(v), 2) for v in use_pred],
            "kpt_pred": [round(float(v), 2) for v in kpt_pred],
            "pv_pred": [round(float(v), 2) for v in pv_pred],
            "contract_check": contract_check
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

@app.route("/predict")
def predict():
    try:
        if _df is None:
            if not load_data():
                return jsonify({"error": "資料未載入"}), 500
        if use_model is None or pv_model is None:
            load_all_models()
        date = request.args.get("date")
        if not date:
            return jsonify({"error": "缺少 date 參數"}), 400
        try:
            date_obj = pd.to_datetime(date)
            date_str = date_obj.strftime("%Y-%m-%d")
        except:
            return jsonify({"error": "日期格式錯誤"}), 400
        day_data = _df[_df["Date_str"] == date_str].copy()
        if day_data.empty:
            return jsonify({"error": f"找不到 {date_str} 的資料"}), 404
        X_use = get_features(day_data, "use")
        X_pv = get_features(day_data, "pv")
        use_pred = use_model.predict(X_use)
        pv_pred = pv_model.predict(X_pv)
        # 將 pv 負值設為 0
        pv_pred = [max(0, float(v)) for v in pv_pred]
        kpt_pred = [float(v) - float(pv_pred[i]) for i, v in enumerate(use_pred)]
        hours = day_data["Hour"].tolist()
        actual_use = [round(float(v), 2) for v in (day_data["kpt"] + day_data["pv"]).tolist()]
        actual_kpt = [round(float(v), 2) for v in day_data["kpt"].tolist()]
        actual_pv = [round(float(v), 2) for v in day_data["pv"].tolist()]
        
        # 檢查契約容量
        contract_check = check_contract_capacity(use_pred, pv_pred, kpt_pred)
        
        return jsonify({
            "hours": hours,
            "actual_use": actual_use,
            "actual_kpt": actual_kpt,
            "actual_pv": actual_pv,
            "use_pred": [round(float(v), 2) for v in use_pred],
            "kpt_pred": [round(float(v), 2) for v in kpt_pred],
            "pv_pred": [round(float(v), 2) for v in pv_pred],
            "contract_check": contract_check
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

@app.route("/real_data")
def get_real_data():
    try:
        # 確保資料已載入
        if _df is None:
            if not load_data():
                return jsonify({"error": "資料未載入"}), 500
        
        if _df is None:
            return jsonify({"error": "資料未載入"}), 500
        
        # 取得日期參數
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        
        if not start_date or not end_date:
            return jsonify({"error": "缺少 start_date 或 end_date 參數"}), 400
        
        # 轉換日期格式
        try:
            start_obj = pd.to_datetime(start_date)
            end_obj = pd.to_datetime(end_date)
            start_str = start_obj.strftime("%Y-%m-%d")
            end_str = end_obj.strftime("%Y-%m-%d")
        except:
            return jsonify({"error": "日期格式錯誤"}), 400
        
        # 查詢該日期範圍的資料
        mask = (_df["Date_str"] >= start_str) & (_df["Date_str"] <= end_str)
        range_data = _df[mask].copy()
        
        if range_data.empty:
            return jsonify({"error": f"找不到 {start_str} 到 {end_str} 的資料"}), 404
        
        # 準備回傳資料
        results = []
        for _, row in range_data.iterrows():
            results.append({
                "date": row["Date_str"],
                "hour": int(row["Hour"]),
                "temperature": float(row["Temperature"]),
                "humidity": float(row["Humidity"]),
                "rain": float(row["Rain"]),
                "day_week": int(row["day_week"]),
                "actual_kpt": float(row["kpt"]),
                "actual_pv": float(row["pv"]),
                "actual_use": float(row["use"])
            })
        
        print(f"真實資料查詢 {start_str} 到 {end_str}: 回傳 {len(results)} 筆資料")
        
        return jsonify({
            "success": True,
            "message": f"成功取得 {len(results)} 筆真實資料",
            "results": results
        })
        
    except Exception as e:
        print(f"真實資料查詢錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

@app.route("/upload_predict", methods=['POST'])
def upload_predict():
    try:
        if use_model is None or pv_model is None:
            load_all_models()
        if 'file' not in request.files:
            return jsonify({"error": "沒有上傳檔案"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "沒有選擇檔案"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "只允許上傳 CSV 檔案"}), 400
        import io
        uploaded_df = pd.read_csv(io.BytesIO(file.read()))
        # 自動補齊缺少欄位
        for col, default in zip(['label'], [63]):
            if col not in uploaded_df.columns:
                uploaded_df[col] = default
        # 統一日期格式
        uploaded_df["Date_str"] = uploaded_df["Date"].apply(lambda d: d.replace("/", "-") if "/" in str(d) else d)
        uploaded_df["Date_str"] = pd.to_datetime(uploaded_df["Date_str"]).dt.strftime("%Y-%m-%d")
        # 預測
        X_use = get_features(uploaded_df, "use")
        X_pv = get_features(uploaded_df, "pv")
        use_pred = use_model.predict(X_use)
        pv_pred = pv_model.predict(X_pv)
        # 將 pv 負值設為 0
        pv_pred = [max(0, float(v)) for v in pv_pred]
        kpt_pred = [float(v) - float(pv_pred[i]) for i, v in enumerate(use_pred)]
        
        # 檢查契約容量
        contract_check = check_contract_capacity(use_pred, pv_pred, kpt_pred)
        results = []


        #-------
        # Persist the uploaded file and cache the latest predictions
        global latest_upload_payload, latest_file
        try:
            # make sure we can re-read the uploaded stream (we read it once for pandas)
            file.stream.seek(0)

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_as = 'latest.csv'
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_as)

            # replace the old saved file
            try:
                os.remove(save_path)
            except FileNotFoundError:
                pass

            file.save(save_path)
            latest_file = save_as
        except Exception as save_err:
            # This isn't fatal for predictions; just log it
            print(f"Warning: failed to persist uploaded file: {save_err}")

        # Cache predictions for other pages
        latest_upload_payload = {
            "results": results,
            "contract_check": contract_check
        }
        #--------

        
        for i, row in uploaded_df.iterrows():
            results.append({
                "date": row["Date_str"],
                "hour": int(row["Hour"]),
                "use_pred": round(float(use_pred[i]), 2),
                "pv_pred": round(float(pv_pred[i]), 2),
                "kpt_pred": round(float(kpt_pred[i]), 2)
            })
        return jsonify({
            "success": True,
            "message": f"成功預測 {len(results)} 筆資料",
            "results": results,
            "contract_check": contract_check
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

#Add a small endpoint so any page (including battery.html) can fetch the last upload:
@app.route("/latest_upload", methods=["GET"])
def latest_upload():
    global latest_upload_payload
    if not latest_upload_payload:
        return jsonify({"success": False, "error": "尚未上傳任何檔案"}), 404
    return jsonify({"success": True, **latest_upload_payload})

@app.route("/contract_info")
def get_contract_info():
    """取得契約容量相關資訊"""
    return jsonify({
        'contract_capacity': CONTRACT_CAPACITY,
        'description': f'契約容量為 {CONTRACT_CAPACITY} kWh/小時，表示每小時從電網購電量不能超過此限制'
    })

if __name__ == "__main__":
    # 啟動時載入資料和模型
    load_data()
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False) 