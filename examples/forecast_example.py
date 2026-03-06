import json
import os
import pickle
from pathlib import Path

import pandas as pd


def normalize_saved_path(base_dir: str, saved_path: str) -> str:
    """
    Chuẩn hóa path lấy từ metadata.json để chạy được trên cả Windows/Linux.
    Ví dụ:
      ..\\saved\\country_Viet_Nam.pkl
      ../saved/country_Viet_Nam.pkl
    """
    if not saved_path:
        return ""

    cleaned = saved_path.replace("\\", "/").replace("../", "").lstrip("/")
    return os.path.join(base_dir, cleaned)


def to_relative_project_path(base_dir: str, abs_path: str) -> str:
    """
    Chuyển absolute path thành path tương đối so với thư mục gốc project.
    """
    return os.path.relpath(abs_path, base_dir).replace("\\", "/")


def load_pickle_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    CURRENT_FILE = Path(__file__).resolve()
    BASE_DIR = CURRENT_FILE.parents[1]
    DATA_PATH = BASE_DIR / "data" / "time_series_country.csv"
    SAVED_DIR = BASE_DIR / "saved"
    METADATA_PATH = SAVED_DIR / "metadata.json"
    OUTPUT_PATH = SAVED_DIR / "forecast_data.csv"

    print("=" * 80)
    print("GENERATE FORECAST CSV FROM PICKLE MODELS")
    print("=" * 80)
    print(f"BASE_DIR      : {BASE_DIR}")
    print(f"DATA_PATH     : {DATA_PATH}")
    print(f"METADATA_PATH : {METADATA_PATH}")
    print(f"OUTPUT_PATH   : {OUTPUT_PATH}")
    print()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {METADATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_cols = {"Country", "Country Code", "Region", "Year"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu cột trong CSV: {missing_cols}")

    country_master = (
        df[["Country", "Country Code", "Region"]]
        .dropna(subset=["Country", "Country Code"])
        .drop_duplicates(subset=["Country"])
        .reset_index(drop=True)
    )

    forecast_years = list(range(2025, 2031))
    periods = len(forecast_years)

    print(f"Số quốc gia: {len(country_master)}")
    print(f"Năm dự báo: {forecast_years}")
    print()

    results = []
    loaded_model_cache = {}
    success_count = 0
    error_count = 0

    for idx, row in country_master.iterrows():
        country = row["Country"]
        country_code = row["Country Code"]
        region = row["Region"] if pd.notna(row["Region"]) else None

        try:
            country_model_path = metadata.get("countries", {}).get(country)
            model_source = None
            resolved_model_path = None

            if country_model_path and not str(country_model_path).startswith("fallback"):
                resolved_model_path = normalize_saved_path(str(BASE_DIR), str(country_model_path))
                model_source = "country"
            else:
                fallback_region = metadata.get("fallback_models", {}).get(country, region)
                region_model_path = metadata.get("regions", {}).get(fallback_region)

                if not region_model_path:
                    print(f"[WARN] Không có region model cho: {country} ({fallback_region})")
                    error_count += 1
                    continue

                resolved_model_path = normalize_saved_path(str(BASE_DIR), str(region_model_path))
                model_source = f"fallback_region:{fallback_region}"

            if not resolved_model_path or not os.path.exists(resolved_model_path):
                print(f"[WARN] Không tìm thấy model file cho {country}: {resolved_model_path}")
                error_count += 1
                continue

            relative_model_path = to_relative_project_path(str(BASE_DIR), resolved_model_path)

            if resolved_model_path in loaded_model_cache:
                model = loaded_model_cache[resolved_model_path]
            else:
                model = load_pickle_model(resolved_model_path)
                loaded_model_cache[resolved_model_path] = model

            future = model.make_future_dataframe(periods=periods, freq="YE")
            forecast = model.predict(future).copy()

            if "ds" not in forecast.columns:
                print(f"[WARN] Forecast output không có cột 'ds' cho {country}")
                error_count += 1
                continue

            forecast["Year"] = pd.to_datetime(forecast["ds"]).dt.year
            forecast_filtered = forecast[forecast["Year"].isin(forecast_years)].copy()

            if forecast_filtered.empty:
                print(f"[WARN] Không có dòng forecast hợp lệ cho {country}")
                error_count += 1
                continue

            for _, fc_row in forecast_filtered.iterrows():
                results.append({
                    "Country": country,
                    "Country Code": country_code,
                    "Region": region,
                    "Year": int(fc_row["Year"]),
                    "Mortality_Rate": float(fc_row["yhat"]),
                    "yhat_lower": float(fc_row["yhat_lower"]) if "yhat_lower" in forecast_filtered.columns and pd.notna(fc_row.get("yhat_lower")) else None,
                    "yhat_upper": float(fc_row["yhat_upper"]) if "yhat_upper" in forecast_filtered.columns and pd.notna(fc_row.get("yhat_upper")) else None,
                    "Model_Source": model_source,
                    "Model_Path": relative_model_path,
                })

            success_count += 1

            if (idx + 1) % 25 == 0:
                print(f"Đã xử lý {idx + 1}/{len(country_master)} quốc gia...")

        except Exception as e:
            print(f"[ERROR] {country}: {e}")
            error_count += 1

    if not results:
        raise RuntimeError("Không tạo được dữ liệu forecast nào.")

    forecast_df = pd.DataFrame(results)
    forecast_df = forecast_df.sort_values(["Year", "Country"]).reset_index(drop=True)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Quốc gia dự báo thành công : {success_count}")
    print(f"Lỗi / bỏ qua              : {error_count}")
    print(f"Số dòng output            : {len(forecast_df)}")
    print(f"File đã lưu               : {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()