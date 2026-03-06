import os
import io

import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'time_series_country.csv')
FORECAST_CSV_PATH = os.path.join(BASE_DIR, 'saved', 'forecast_data.csv')

print("Đường dẫn dữ liệu lịch sử:", DATA_PATH)
print("Đường dẫn dữ liệu dự báo:", FORECAST_CSV_PATH)

# =========================
# LOAD HISTORICAL DATA
# =========================
try:
    df = pd.read_csv(DATA_PATH)
    print("Đã đọc file lịch sử CSV. Số dòng:", len(df))
    print("Các cột lịch sử:", df.columns.tolist())
except Exception as e:
    print("LỖI đọc file lịch sử:", e)
    df = pd.DataFrame()

# =========================
# LOAD FORECAST CSV
# =========================
try:
    forecast_df = pd.read_csv(FORECAST_CSV_PATH)
    print("Đã đọc file forecast CSV. Số dòng:", len(forecast_df))
    print("Các cột forecast:", forecast_df.columns.tolist())
except Exception as e:
    print("LỖI đọc file forecast:", e)
    forecast_df = pd.DataFrame()


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
    years = sorted(df['Year'].dropna().unique())
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

    result = data[['Country', 'Country Code', 'Mortality_Rate']].to_dict(orient='records')

    for item in result:
        if pd.notna(item['Mortality_Rate']):
            item['Mortality_Rate'] = float(item['Mortality_Rate'])
        else:
            item['Mortality_Rate'] = None

    return jsonify(result)


# API: Lấy dữ liệu lịch sử của một quốc gia theo tên
@app.route('/api/historical/<country>')
def get_historical(country):
    if df.empty:
        return jsonify({'error': 'No data'}), 500

    data = df[df['Country'].str.strip().str.lower() == country.strip().lower()]
    if data.empty:
        return jsonify({'error': 'Country not found'}), 404

    data = data.sort_values('Year')
    records = data.to_dict(orient='records')

    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, 'item'):
                record[key] = value.item()

    return jsonify(records)


# API: Lấy dữ liệu lịch sử của một quốc gia theo mã ISO3
@app.route('/api/historical-by-code/<country_code>')
def get_historical_by_code(country_code):
    if df.empty:
        return jsonify({'error': 'No data'}), 500

    data = df[df['Country Code'].astype(str).str.strip().str.upper() == country_code.strip().upper()]
    if data.empty:
        return jsonify({'error': 'Country not found'}), 404

    data = data.sort_values('Year')
    records = data.to_dict(orient='records')

    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, 'item'):
                record[key] = value.item()

    return jsonify(records)


# API: Forecast đọc từ CSV đã precompute sẵn
@app.route("/api/forecast", methods=["POST"])
def run_forecast():
    try:
        if forecast_df.empty:
            return jsonify({"error": "Forecast CSV is empty or not loaded"}), 500

        data = request.get_json(silent=True) or {}
        n_years = int(data.get("n_years", 1))

        target_year = 2024 + n_years

        filtered = forecast_df[forecast_df["Year"] == target_year].copy()
        if filtered.empty:
            return jsonify({"error": f"No forecast data for year {target_year}"}), 404

        result = {
            "year": int(target_year),
            "values": {},
            "intervals": {}
        }

        for _, row in filtered.iterrows():
            country_code = row["Country Code"]
            if pd.isna(country_code):
                continue

            code = str(country_code)

            result["values"][code] = (
                float(row["Mortality_Rate"])
                if pd.notna(row["Mortality_Rate"])
                else None
            )

            result["intervals"][code] = {
                "lo": float(row["yhat_lower"]) if "yhat_lower" in filtered.columns and pd.notna(row["yhat_lower"]) else None,
                "hi": float(row["yhat_upper"]) if "yhat_upper" in filtered.columns and pd.notna(row["yhat_upper"]) else None,
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def build_download_filename(export_type: str, country_code: str | None, target_year: int | None):
    scope = country_code.upper() if country_code else "world"
    if export_type == "history":
        return f"historical_{scope}.csv"
    if export_type == "forecast" and target_year:
        return f"forecast_{scope}_through_{target_year}.csv"
    return f"export_{scope}.csv"


def dataframe_to_csv_response(df_export: pd.DataFrame, filename: str):
    output = io.BytesIO()
    csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    output.write(csv_bytes)
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/export", methods=["GET"])
def export_data():
    try:
        export_type = request.args.get("type", "history").strip().lower()
        country_code = request.args.get("country_code", default=None, type=str)
        target_year = request.args.get("year", default=None, type=int)

        if export_type not in {"history", "forecast"}:
            return jsonify({"error": "Invalid export type"}), 400

        hist_df = df.copy()
        if hist_df.empty:
            return jsonify({"error": "Historical data is empty"}), 500

        if country_code:
            country_code = country_code.strip().upper()
            hist_df = hist_df[
                hist_df["Country Code"].astype(str).str.strip().str.upper() == country_code
            ]

        if hist_df.empty:
            return jsonify({"error": "No historical data found"}), 404

        hist_df = hist_df.sort_values(["Country", "Year"]).reset_index(drop=True)

        if export_type == "history":
            export_df = hist_df.copy()
            filename = build_download_filename("history", country_code, None)
            return dataframe_to_csv_response(export_df, filename)

        if forecast_df.empty:
            return jsonify({"error": "Forecast CSV is empty or not loaded"}), 500

        if target_year is None:
            return jsonify({"error": "Missing forecast year"}), 400

        fc_df = forecast_df.copy()

        if country_code:
            fc_df = fc_df[
                fc_df["Country Code"].astype(str).str.strip().str.upper() == country_code
            ]

        fc_df = fc_df[fc_df["Year"] <= target_year]

        if fc_df.empty:
            return jsonify({"error": f"No forecast data found up to year {target_year}"}), 404

        fc_df = fc_df.sort_values(["Country", "Year"]).reset_index(drop=True)

        hist_export = hist_df.copy()
        hist_export["Data_Type"] = "historical"

        forecast_export = fc_df.copy()
        forecast_export["Data_Type"] = "forecast"

        all_cols = [
            "Country",
            "Country Code",
            "Region",
            "Year",
            "Mortality_Rate",
            "yhat_lower",
            "yhat_upper",
            "Model_Source",
            "Model_Path",
            "Data_Type",
        ]

        for col in all_cols:
            if col not in hist_export.columns:
                hist_export[col] = None

        for col in all_cols:
            if col not in forecast_export.columns:
                forecast_export[col] = None

        hist_export = hist_export[all_cols]
        forecast_export = forecast_export[all_cols]

        export_df = pd.concat([hist_export, forecast_export], ignore_index=True)
        export_df = export_df.sort_values(["Country", "Year", "Data_Type"]).reset_index(drop=True)

        filename = build_download_filename("forecast", country_code, target_year)
        return dataframe_to_csv_response(export_df, filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)