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


def append_null_forecast(results, country, country_code, region, forecast_years, model_source, model_path=None):
    """
    Append các dòng forecast NULL cho country không có model/fallback.
    """
    for year in forecast_years:
        results.append({
            "Country": country,
            "Country Code": country_code,
            "Region": region,
            "Year": int(year),
            "Mortality_Rate": None,
            "yhat_lower": None,
            "yhat_upper": None,
            "Model_Source": model_source,
            "Model_Path": model_path,
        })


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
    null_count = 0
    error_count = 0

    metadata_countries = metadata.get("countries", {})
    metadata_regions = metadata.get("regions", {})
    fallback_models = metadata.get("fallback_models", {})

    for idx, row in country_master.iterrows():
        country = row["Country"]
        country_code = row["Country Code"]
        region = row["Region"] if pd.notna(row["Region"]) else None

        try:
            country_meta = metadata_countries.get(country)
            model_source = None
            resolved_model_path = None

            # -----------------------------------------------------------------
            # CASE 1: metadata kiểu mới dạng dict
            # -----------------------------------------------------------------
            if isinstance(country_meta, dict):
                status = country_meta.get("status")

                if status == "trained_country":
                    country_model_path = country_meta.get("model_path")
                    if country_model_path:
                        resolved_model_path = normalize_saved_path(str(BASE_DIR), str(country_model_path))
                        model_source = "country"

                elif status == "fallback_region":
                    fallback_region = country_meta.get("region", region)
                    region_model_path = metadata_regions.get(fallback_region)

                    if region_model_path:
                        resolved_model_path = normalize_saved_path(str(BASE_DIR), str(region_model_path))
                        model_source = f"fallback_region:{fallback_region}"
                    else:
                        append_null_forecast(
                            results=results,
                            country=country,
                            country_code=country_code,
                            region=region,
                            forecast_years=forecast_years,
                            model_source=f"no_region_model:{fallback_region}",
                            model_path=None,
                        )
                        null_count += 1
                        print(f"[WARN] Không có region model cho {country} ({fallback_region}) -> forecast NULL")
                        continue

                elif status == "skipped":
                    append_null_forecast(
                        results=results,
                        country=country,
                        country_code=country_code,
                        region=region,
                        forecast_years=forecast_years,
                        model_source="no_model_insufficient_data",
                        model_path=None,
                    )
                    null_count += 1
                    continue

                elif status == "error":
                    append_null_forecast(
                        results=results,
                        country=country,
                        country_code=country_code,
                        region=region,
                        forecast_years=forecast_years,
                        model_source="no_model_training_error",
                        model_path=None,
                    )
                    null_count += 1
                    continue

                else:
                    append_null_forecast(
                        results=results,
                        country=country,
                        country_code=country_code,
                        region=region,
                        forecast_years=forecast_years,
                        model_source="no_model_unknown_status",
                        model_path=None,
                    )
                    null_count += 1
                    continue

            # -----------------------------------------------------------------
            # CASE 2: metadata kiểu cũ dạng string
            # -----------------------------------------------------------------
            elif isinstance(country_meta, str):
                # country model trực tiếp
                if not country_meta.startswith("fallback"):
                    resolved_model_path = normalize_saved_path(str(BASE_DIR), str(country_meta))
                    model_source = "country"
                else:
                    # fallback theo metadata cũ
                    fallback_region = fallback_models.get(country, region)
                    region_model_path = metadata_regions.get(fallback_region)

                    if region_model_path:
                        resolved_model_path = normalize_saved_path(str(BASE_DIR), str(region_model_path))
                        model_source = f"fallback_region:{fallback_region}"
                    else:
                        append_null_forecast(
                            results=results,
                            country=country,
                            country_code=country_code,
                            region=region,
                            forecast_years=forecast_years,
                            model_source=f"no_region_model:{fallback_region}",
                            model_path=None,
                        )
                        null_count += 1
                        print(f"[WARN] Không có region model cho {country} ({fallback_region}) -> forecast NULL")
                        continue

            # -----------------------------------------------------------------
            # CASE 3: không có entry trong metadata["countries"]
            # => nhóm < 13 năm dữ liệu hoặc missing metadata
            # => forecast NULL
            # -----------------------------------------------------------------
            else:
                append_null_forecast(
                    results=results,
                    country=country,
                    country_code=country_code,
                    region=region,
                    forecast_years=forecast_years,
                    model_source="no_model_no_metadata",
                    model_path=None,
                )
                null_count += 1
                continue

            # -----------------------------------------------------------------
            # Nếu path model không hợp lệ -> forecast NULL
            # -----------------------------------------------------------------
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                append_null_forecast(
                    results=results,
                    country=country,
                    country_code=country_code,
                    region=region,
                    forecast_years=forecast_years,
                    model_source="no_model_missing_pickle",
                    model_path=None,
                )
                null_count += 1
                print(f"[WARN] Không tìm thấy model file cho {country}: {resolved_model_path} -> forecast NULL")
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
                append_null_forecast(
                    results=results,
                    country=country,
                    country_code=country_code,
                    region=region,
                    forecast_years=forecast_years,
                    model_source="no_model_invalid_forecast_output",
                    model_path=relative_model_path,
                )
                null_count += 1
                print(f"[WARN] Forecast output không có cột 'ds' cho {country} -> forecast NULL")
                continue

            forecast["Year"] = pd.to_datetime(forecast["ds"]).dt.year
            forecast_filtered = forecast[forecast["Year"].isin(forecast_years)].copy()

            if forecast_filtered.empty:
                append_null_forecast(
                    results=results,
                    country=country,
                    country_code=country_code,
                    region=region,
                    forecast_years=forecast_years,
                    model_source="no_model_empty_forecast",
                    model_path=relative_model_path,
                )
                null_count += 1
                print(f"[WARN] Không có dòng forecast hợp lệ cho {country} -> forecast NULL")
                continue

            for _, fc_row in forecast_filtered.iterrows():
                results.append({
                    "Country": country,
                    "Country Code": country_code,
                    "Region": region,
                    "Year": int(fc_row["Year"]),
                    "Mortality_Rate": float(fc_row["yhat"]) if pd.notna(fc_row.get("yhat")) else None,
                    "yhat_lower": float(fc_row["yhat_lower"]) if "yhat_lower" in forecast_filtered.columns and pd.notna(fc_row.get("yhat_lower")) else None,
                    "yhat_upper": float(fc_row["yhat_upper"]) if "yhat_upper" in forecast_filtered.columns and pd.notna(fc_row.get("yhat_upper")) else None,
                    "Model_Source": model_source,
                    "Model_Path": relative_model_path,
                })

            success_count += 1

            if (idx + 1) % 25 == 0:
                print(f"Đã xử lý {idx + 1}/{len(country_master)} quốc gia...")

        except Exception as e:
            append_null_forecast(
                results=results,
                country=country,
                country_code=country_code,
                region=region,
                forecast_years=forecast_years,
                model_source="no_model_runtime_error",
                model_path=None,
            )
            print(f"[ERROR] {country}: {e} -> forecast NULL")
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
    print(f"Quốc gia forecast thành công : {success_count}")
    print(f"Quốc gia forecast NULL       : {null_count}")
    print(f"Lỗi runtime                 : {error_count}")
    print(f"Số dòng output              : {len(forecast_df)}")
    print(f"File đã lưu                 : {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()