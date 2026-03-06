"""
TrainModel: Hệ thống huấn luyện Prophet model theo cấp độ Global, Region, Country

Cấu trúc:
- Global: 1 model cho toàn bộ dữ liệu
- Region: 1 model cho mỗi region
- Country:
    - Nếu country có >= 21 năm dữ liệu  -> train country model
    - Nếu country có từ 13 đến 20 năm   -> fallback sang region chứa country đó
    - Nếu country có < 13 năm dữ liệu   -> không train, không fallback

Output:
- model/global.pkl
- model/region_{region_name}.pkl
- model/country_{country_name}.pkl
- model/metadata.json
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.model.prophet import ProphetModel
from src.processing.aggregate import prepare_level_data


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = "TrainModel") -> logging.Logger:
    """Cấu hình logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Tránh add handler trùng nhiều lần
    if logger.handlers:
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


# ============================================================================
# MAIN CLASS
# ============================================================================

class TrainModel:
    """
    Hệ thống huấn luyện Prophet model cho Global, Region, Country.

    Logic country:
    - years_with_data >= min_years_for_country_model: train country model
    - min_years_for_region_fallback <= years_with_data < min_years_for_country_model:
        fallback sang region
    - years_with_data < min_years_for_region_fallback:
        không train, không fallback
    """

    def __init__(
        self,
        df: pd.DataFrame,
        param_grid: Dict = None,
        model_dir: str = "../saved",
        year_start: int = 2000,
        year_end: int = 2024,
        min_years_for_region_fallback: int = 13,
        min_years_for_country_model: int = 20,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame gốc, kỳ vọng có các cột:
            Year, Country, Region, Total_Deaths, Population
        param_grid : Dict, optional
            Grid search params cho Prophet
        model_dir : str
            Thư mục lưu model
        year_start : int
            Năm bắt đầu train
        year_end : int
            Năm kết thúc train
        min_years_for_region_fallback : int
            Tối thiểu số năm dữ liệu để fallback sang region
        min_years_for_country_model : int
            Tối thiểu số năm dữ liệu để train country model
        """
        self.df = df.copy()

        if param_grid is None:
            param_grid = {
                "changepoint_prior_scale": [0.001, 0.01],
                "seasonality_prior_scale": [0.01, 0.1],
            }
        self.param_grid = param_grid

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.year_start = year_start
        self.year_end = year_end
        self.min_years_for_region_fallback = min_years_for_region_fallback
        self.min_years_for_country_model = min_years_for_country_model

        self.metadata = {
            "global": None,
            "regions": {},
            "countries": {},
        }

        # country -> region
        self.fallback_models: Dict[str, str] = {}

        # cache region đã train
        self.trained_regions = set()

        logger.info(f"TrainModel initialized with model_dir={self.model_dir}")
        logger.info(f"Year range: {self.year_start}-{self.year_end}")
        logger.info(
            "Country logic: "
            f">= {self.min_years_for_country_model} years -> train country, "
            f"{self.min_years_for_region_fallback}-{self.min_years_for_country_model - 1} years -> fallback region, "
            f"< {self.min_years_for_region_fallback} years -> skip"
        )

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def train_global_model(self) -> bool:
        """
        Huấn luyện Global model trên toàn bộ dữ liệu.

        Returns
        -------
        bool
            True nếu train thành công, ngược lại False
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING GLOBAL MODEL")
        logger.info("=" * 70)

        try:
            logger.info("Preparing global data...")
            data_global = prepare_level_data(
                df=self.df,
                level="global",
                year_start=self.year_start,
                year_end=self.year_end,
                fill_missing_years=True,
            )

            logger.info(f"  - Data shape: {data_global.shape}")
            logger.info(
                f"  - Years: {data_global['Year'].min()} - {data_global['Year'].max()}"
            )

            if len(data_global) < 5:
                logger.warning("❌ Not enough data for global model")
                return False

            logger.info("Initializing ProphetModel...")
            model = ProphetModel(
                df=data_global,
                value_col="Mortality_Rate",
                date_col="Year",
            )

            logger.info("Running Prophet pipeline (grid search + cross-validation)...")
            result = model.run_pipeline_prophet(
                param_grid=self.param_grid,
                min_train_periods=min(5, len(data_global) - 2),
                horizon=1,
                step=1,
                alpha=0.10,
                forecast_steps=5,
            )

            if result is None:
                logger.error("❌ Pipeline failed for global model")
                return False

            final_model = model.train_final_model(
                best_params=result["best_params"],
                holiday_years=result["holiday_years"],
            )

            model_path = self._save_model(final_model, "global")
            self.metadata["global"] = str(model_path)

            logger.info("✅ Global model trained successfully")
            logger.info(f"   Saved to: {model_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error training global model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def train_region_models(self) -> Dict[str, bool]:
        """
        Huấn luyện model cho mỗi region.

        Returns
        -------
        Dict[str, bool]
            {region_name: success_flag}
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING REGION MODELS")
        logger.info("=" * 70)

        regions = sorted(self.df["Region"].dropna().unique())
        logger.info(f"Found {len(regions)} regions: {regions}")

        results: Dict[str, bool] = {}

        for region_name in regions:
            logger.info(f"\n--- Region: {region_name} ---")

            try:
                if region_name in self.trained_regions:
                    logger.info("⚠️ Already trained, skipping...")
                    results[region_name] = True
                    continue

                logger.info(f"Preparing data for region {region_name}...")
                data_region = prepare_level_data(
                    df=self.df,
                    level="region",
                    filter=region_name,
                    year_start=self.year_start,
                    year_end=self.year_end,
                    fill_missing_years=True,
                )

                logger.info(f"  - Data shape: {data_region.shape}")
                logger.info(
                    f"  - Years: {data_region['Year'].min()} - {data_region['Year'].max()}"
                )

                if len(data_region) < 5:
                    logger.warning(f"❌ Not enough data for region {region_name}")
                    results[region_name] = False
                    continue

                logger.info("Initializing ProphetModel...")
                model = ProphetModel(
                    df=data_region,
                    value_col="Mortality_Rate",
                    date_col="Year",
                )

                logger.info("Running Prophet pipeline...")
                result = model.run_pipeline_prophet(
                    param_grid=self.param_grid,
                    min_train_periods=min(5, len(data_region) - 2),
                    horizon=1,
                    step=1,
                    alpha=0.10,
                    forecast_steps=5,
                )

                if result is None:
                    logger.error(f"❌ Pipeline failed for region {region_name}")
                    results[region_name] = False
                    continue

                final_model = model.train_final_model(
                    best_params=result["best_params"],
                    holiday_years=result["holiday_years"],
                )

                model_path = self._save_model(
                    model=final_model,
                    model_name=f"region_{self._safe_name(region_name)}",
                )

                self.metadata["regions"][region_name] = str(model_path)
                self.trained_regions.add(region_name)

                logger.info("✅ Region model trained successfully")
                logger.info(f"   Saved to: {model_path}")

                results[region_name] = True

            except Exception as e:
                logger.error(f"❌ Error training region {region_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                results[region_name] = False

        return results

    def train_country_models(self) -> Tuple[Dict[str, bool], Dict[str, str]]:
        """
        Huấn luyện model cho mỗi country theo logic:

        - years_with_data >= 21: train country model
        - 13 <= years_with_data <= 20: fallback sang region
        - years_with_data < 13: skip

        Returns
        -------
        Tuple[Dict[str, bool], Dict[str, str]]
            - train_results: {country_name: success_flag}
            - fallback_mapping: {country_name: region_name}
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COUNTRY MODELS")
        logger.info("=" * 70)

        countries = sorted(self.df["Country"].dropna().unique())
        logger.info(f"Found {len(countries)} countries")

        train_results: Dict[str, bool] = {}
        fallback_mapping: Dict[str, str] = {}

        for country_name in countries:
            logger.info(f"\n--- Country: {country_name} ---")

            try:
                country_data_all = self.df[self.df["Country"] == country_name]

                if country_data_all.empty:
                    logger.warning(f"❌ No raw data found for country {country_name}")
                    self.metadata["countries"][country_name] = {
                        "status": "skipped",
                        "reason": "no_raw_data",
                    }
                    train_results[country_name] = False
                    continue

                region_name = country_data_all["Region"].iloc[0]
                logger.info(f"Region: {region_name}")

                logger.info(f"Preparing data for country {country_name}...")
                data_country = prepare_level_data(
                    df=self.df,
                    level="country",
                    filter=country_name,
                    year_start=self.year_start,
                    year_end=self.year_end,
                    fill_missing_years=True,
                )

                logger.info(f"  - Data shape: {data_country.shape}")
                logger.info(
                    f"  - Years available: {data_country['Year'].min()} - {data_country['Year'].max()}"
                )

                # Đếm số năm có dữ liệu thực sự
                # Ở đây dùng Total_Deaths > 0 theo logic hiện tại của bạn
                years_with_data = int((data_country["Total_Deaths"] > 0).sum())
                logger.info(f"  - Years with Total_Deaths > 0: {years_with_data}")

                # CASE 1: < 13 năm -> skip
                if years_with_data < self.min_years_for_region_fallback:
                    logger.info(
                        f"⚠️ Too little data ({years_with_data} < {self.min_years_for_region_fallback})"
                    )
                    logger.info("    → Skip model training, no fallback")

                    self.metadata["countries"][country_name] = {
                        "status": "skipped",
                        "reason": f"insufficient_data_lt_{self.min_years_for_region_fallback}",
                        "years_with_data": years_with_data,
                        "region": region_name,
                    }

                    train_results[country_name] = False
                    continue

                # CASE 2: 13-20 năm -> fallback region
                if years_with_data < self.min_years_for_country_model:
                    logger.info(
                        f"⚠️ Moderate data ({years_with_data} years) "
                        f"→ fallback to region: {region_name}"
                    )

                    fallback_mapping[country_name] = region_name
                    self.fallback_models[country_name] = region_name

                    self.metadata["countries"][country_name] = {
                        "status": "fallback_region",
                        "region": region_name,
                        "years_with_data": years_with_data,
                    }

                    train_results[country_name] = False
                    continue

                # CASE 3: >= 21 năm -> train country
                if len(data_country) < 5:
                    logger.warning(f"❌ Not enough data points for country {country_name}")

                    self.metadata["countries"][country_name] = {
                        "status": "skipped",
                        "reason": "not_enough_datapoints",
                        "years_with_data": years_with_data,
                        "region": region_name,
                    }

                    train_results[country_name] = False
                    continue

                logger.info("Initializing ProphetModel...")
                model = ProphetModel(
                    df=data_country,
                    value_col="Mortality_Rate",
                    date_col="Year",
                )

                logger.info("Running Prophet pipeline...")
                result = model.run_pipeline_prophet(
                    param_grid=self.param_grid,
                    min_train_periods=min(5, len(data_country) - 2),
                    horizon=1,
                    step=1,
                    alpha=0.10,
                    forecast_steps=5,
                )

                if result is None:
                    logger.error(f"❌ Pipeline failed for country {country_name}")

                    self.metadata["countries"][country_name] = {
                        "status": "skipped",
                        "reason": "pipeline_failed",
                        "years_with_data": years_with_data,
                        "region": region_name,
                    }

                    train_results[country_name] = False
                    continue

                final_model = model.train_final_model(
                    best_params=result["best_params"],
                    holiday_years=result["holiday_years"],
                )

                model_path = self._save_model(
                    model=final_model,
                    model_name=f"country_{self._safe_name(country_name)}",
                )

                self.metadata["countries"][country_name] = {
                    "status": "trained_country",
                    "model_path": str(model_path),
                    "years_with_data": years_with_data,
                    "region": region_name,
                }

                logger.info("✅ Country model trained successfully")
                logger.info(f"   Saved to: {model_path}")

                train_results[country_name] = True

            except Exception as e:
                logger.error(f"❌ Error training country {country_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

                try:
                    region_name = self.df[self.df["Country"] == country_name]["Region"].iloc[0]
                except Exception:
                    region_name = None

                self.metadata["countries"][country_name] = {
                    "status": "error",
                    "reason": str(e),
                    "region": region_name,
                }
                train_results[country_name] = False

        return train_results, fallback_mapping

    def train_all(self) -> Dict:
        """
        Huấn luyện toàn bộ pipeline: Global -> Region -> Country.

        Returns
        -------
        Dict
            Kết quả tổng hợp
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE TRAINING PIPELINE: GLOBAL -> REGION -> COUNTRY")
        logger.info("=" * 80)

        results = {
            "global": None,
            "regions": {},
            "countries": {},
            "fallback_count": 0,
            "skipped_count": 0,
        }

        # 1. Global
        global_success = self.train_global_model()
        results["global"] = "SUCCESS" if global_success else "FAILED"

        # 2. Region
        region_results = self.train_region_models()
        successful_regions = sum(1 for v in region_results.values() if v)
        results["regions"] = region_results

        logger.info(
            f"\nRegion Summary: {successful_regions}/{len(region_results)} successful"
        )

        # 3. Country
        country_results, fallback_mapping = self.train_country_models()
        successful_countries = sum(1 for v in country_results.values() if v)
        fallback_count = len(fallback_mapping)

        skipped_count = sum(
            1
            for country_meta in self.metadata["countries"].values()
            if isinstance(country_meta, dict) and country_meta.get("status") == "skipped"
        )

        results["countries"] = country_results
        results["fallback_count"] = fallback_count
        results["skipped_count"] = skipped_count

        logger.info(
            f"\nCountry Summary: {successful_countries}/{len(country_results)} trained"
        )
        logger.info(f"                 {fallback_count} using fallback to region")
        logger.info(f"                 {skipped_count} skipped (no model)")

        # 4. Save metadata
        self._save_metadata()

        # 5. Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Global model:        {results['global']}")
        logger.info(
            f"Region models:       {successful_regions}/{len(region_results)} trained"
        )
        logger.info(
            f"Country models:      {successful_countries}/{len(country_results)} trained"
        )
        logger.info(f"Fallback countries:  {fallback_count}")
        logger.info(f"Skipped countries:   {skipped_count}")
        logger.info(f"Metadata saved to:   {self.model_dir / 'metadata.json'}")
        logger.info(f"All models saved to: {self.model_dir}/")
        logger.info("=" * 80 + "\n")

        return results

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _save_model(self, model, model_name: str) -> Path:
        """
        Lưu model vào file pickle.
        """
        model_path = self.model_dir / f"{model_name}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved model to {model_path} ({file_size_mb:.2f} MB)")

        return model_path

    def _save_metadata(self) -> None:
        """
        Lưu metadata vào JSON file.
        """
        metadata_path = self.model_dir / "metadata.json"

        metadata_json = {
            "global": self.metadata["global"],
            "regions": self.metadata["regions"],
            "countries": self.metadata["countries"],
            "fallback_models": self.fallback_models,
            "training_config": {
                "year_start": self.year_start,
                "year_end": self.year_end,
                "min_years_for_region_fallback": self.min_years_for_region_fallback,
                "min_years_for_country_model": self.min_years_for_country_model,
                "param_grid": self.param_grid,
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata to {metadata_path}")

    @staticmethod
    def _safe_name(name: str) -> str:
        """
        Chuẩn hóa tên để dùng làm file name.
        """
        return (
            str(name)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_from_pkl(model_path: str):
    """
    Load model từ file pickle.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_metadata(model_dir: str = "model") -> Dict:
    """
    Load metadata từ JSON file.
    """
    metadata_path = Path(model_dir) / "metadata.json"

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


# ============================================================================
# MAIN (EXAMPLE USAGE)
# ============================================================================

if __name__ == "__main__":
    """
    Ví dụ sử dụng:

    import pandas as pd

    df = pd.read_csv("data/time_series_country.csv")

    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.05],
        "seasonality_prior_scale": [0.01, 0.1, 1.0],
    }

    trainer = TrainModel(
        df=df,
        param_grid=param_grid,
        model_dir="model",
        year_start=2000,
        year_end=2024,
        min_years_for_region_fallback=13,
        min_years_for_country_model=21,
    )

    results = trainer.train_all()
    print(results)

    metadata = load_metadata("model")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    """
    pass