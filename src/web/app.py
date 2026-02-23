import json
import os
import pickle

import pandas as pd
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'time_series_country.csv')
print("Đường dẫn dữ liệu:", DATA_PATH)

try:
    df = pd.read_csv(DATA_PATH)
    print("Đã đọc file CSV. Số dòng:", len(df))
    print("Các cột:", df.columns.tolist())
except Exception as e:
    print("LỖI đọc file:", e)
    df = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data-table')
def data_table():
    return render_template('data_table.html')

@app.route('/event-detail')
def event_detail():
    return render_template('event_detail.html')

@app.route('/forecast-modal')
def forecast_modal():
    return render_template('forecast_modal.html')

@app.route('/country-panel')
def country_panel():
    return render_template('country_panel.html')

@app.route('/error')
def error():
    return render_template('error.html')

# API: Lấy danh sách các năm có dữ liệu
@app.route('/api/years')
def get_years():
    if df.empty:
        return jsonify([])
    years = sorted(df['Year'].unique())
    # Chuyển từ int64 sang int
    years = [int(y) for y in years]
    return jsonify(years)

# API: Lấy tỷ lệ tử vong theo năm (đã sửa để trả về Country Code)
@app.route('/api/mortality-by-year/<int:year>')
def get_mortality_by_year(year):
    if df.empty:
        return jsonify({'error': 'No data'}), 500
    data = df[df['Year'] == year]
    if data.empty:
        return jsonify({'error': 'Year not found'}), 404
    # Lấy các cột cần thiết: Country, Country Code, Mortality_Rate
    result = data[['Country', 'Country Code', 'Mortality_Rate']].to_dict(orient='records')
    # Chuyển các giá trị float64 sang float
    for item in result:
        item['Mortality_Rate'] = float(item['Mortality_Rate'])
    return jsonify(result)

# API: Lấy dữ liệu lịch sử của một quốc gia
@app.route('/api/historical/<country>')
def get_historical(country):
    if df.empty:
        return jsonify({'error': 'No data'}), 500
    data = df[df['Country'].str.strip().str.lower() == country.strip().lower()]
    if data.empty:
        return jsonify({'error': 'Country not found'}), 404
    # Chuyển đổi toàn bộ DataFrame thành dict và đảm bảo kiểu dữ liệu Python
    records = data.to_dict(orient='records')
    for record in records:
        for key, value in record.items():
            if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype, pd.Int64Dtype.type, pd.Float64Dtype.type)):
                # Nếu là kiểu pandas, chuyển thành kiểu Python tương ứng
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (int, float, pd.Int64Dtype, pd.Float64Dtype)):
                    record[key] = value.item() if hasattr(value, 'item') else value
    return jsonify(records)

METADATA_PATH = os.path.join(BASE_DIR, "saved", "metadata.json")

@app.route("/api/forecast", methods=["POST"])
def run_forecast():
    try:
        data = request.get_json()
        n_years = int(data.get("n_years", 1))

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        forecast_results = {}

        for country in df["Country"].unique():

            model_path = metadata["countries"].get(country)

            # Nếu có model riêng
            if model_path and not model_path.startswith("fallback"):
                full_path = os.path.join(BASE_DIR, model_path.replace("..\\", ""))
            else:
                # fallback region
                region = metadata["fallback_models"].get(country)
                if not region:
                    continue
                region_path = metadata["regions"].get(region)
                full_path = os.path.join(BASE_DIR, region_path.replace("..\\", ""))

            if not os.path.exists(full_path):
                continue

            with open(full_path, "rb") as f_model:
                model = pickle.load(f_model)

            future = model.make_future_dataframe(periods=n_years, freq="YE")
            forecast = model.predict(future)

            value = forecast.iloc[-1]["yhat"]
            country_code = df[df["Country"] == country]["Country Code"].iloc[0]

            forecast_results[country_code] = float(value)

        return jsonify(forecast_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)