"""
TrainModel: Hệ thống huấn luyện Prophet model theo cấp độ Global, Region, Country

Cấu trúc:
- Global: 1 model cho toàn bộ dữ liệu
- Region: 1 model cho mỗi region (7 regions)
- Country: 1 model cho mỗi country (nếu đủ dữ liệu >= 23 năm)
           - Nếu không đủ dữ liệu: Fallback sang region model

Output:
- model/global.pkl
- model/region_{region_name}.pkl
- model/country_{country_name}.pkl
- model/metadata.json (mapping & fallback info)
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.model.prophet import ProphetModel
from src.processing.aggregate import prepare_level_data


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = "TrainModel") -> logging.Logger:
    """Cấu hình logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)

    # Clear existing handlers
    logger.handlers = []
    logger.addHandler(ch)

    return logger


logger = setup_logger()


# ============================================================================
# MAIN CLASS
# ============================================================================

class TrainModel:
    """
    Hệ thống huấn luyện Prophet saved cho Global, Region, Country

    Attributes:
        df: DataFrame gốc chứa dữ liệu tất cả countries
        param_grid: Dictionary tham số grid search cho Prophet
        model_dir: Đường dẫn thư mục lưu saved
        year_start: Năm bắt đầu training
        year_end: Năm kết thúc training
        min_years_for_country_model: Số năm tối thiểu để train country model

    Methods:
        train_global_model()
        train_region_models()
        train_country_models()
        train_all()
        _load_trained_regions()
        _save_model()
        _save_metadata()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        param_grid: Dict = None,
        model_dir: str = "../saved",
        year_start: int = 2000,
        year_end: int = 2024,
        min_years_for_country_model: int = 23
    ):
        """
        Constructor

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame gốc chứa columns: Year, Country, Region, Total_Deaths, Population
        param_grid : Dict, optional
            Tham số grid search cho Prophet
            Mặc định: {'changepoint_prior_scale': [0.001, 0.01],
                       'seasonality_prior_scale': [0.01, 0.1]}
        model_dir : str, default='model'
            Đường dẫn thư mục lưu saved
        year_start : int, default=2000
            Năm bắt đầu training
        year_end : int, default=2024
            Năm kết thúc training
        min_years_for_country_model : int, default=23
            Số năm tối thiểu có Total_Deaths > 0 để train country model
        """
        self.df = df.copy()

        # Tham số grid search mặc định
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01],
                'seasonality_prior_scale': [0.01, 0.1],
            }
        self.param_grid = param_grid

        # Cấu hình thư mục
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Cấu hình dữ liệu
        self.year_start = year_start
        self.year_end = year_end
        self.min_years_for_country_model = min_years_for_country_model

        # Metadata
        self.metadata = {
            "global": None,
            "regions": {},
            "countries": {}
        }

        # Fallback mapping: country -> region
        self.fallback_models: Dict[str, str] = {}

        # Trained regions (để tránh train lại)
        self.trained_regions: set = set()

        logger.info(f"TrainModel initialized with model_dir={self.model_dir}")
        logger.info(f"Year range: {year_start}-{year_end}")
        logger.info(f"Min years for country model: {min_years_for_country_model}")

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def train_global_model(self) -> bool:
        """
        Huấn luyện Global model trên dữ liệu toàn bộ

        Returns:
        --------
        bool : True nếu thành công, False nếu lỗi
        """
        logger.info("\n" + "="*70)
        logger.info("TRAINING GLOBAL MODEL")
        logger.info("="*70)

        try:
            # 1. Chuẩn bị dữ liệu Global
            logger.info("Preparing global data...")
            data_global = prepare_level_data(
                df=self.df,
                level="global",
                year_start=self.year_start,
                year_end=self.year_end,
                fill_missing_years=True
            )

            logger.info(f"  - Data shape: {data_global.shape}")
            logger.info(f"  - Years: {data_global['Year'].min()} - {data_global['Year'].max()}")

            # 2. Kiểm tra đủ dữ liệu
            if len(data_global) < 5:
                logger.warning("❌ Not enough data for global model")
                return False

            # 3. Khởi tạo ProphetModel
            logger.info("Initializing ProphetModel...")
            model = ProphetModel(
                df=data_global,
                value_col="Mortality_Rate",
                date_col="Year"
            )

            # 4. Chạy pipeline
            logger.info("Running Prophet pipeline (grid search + cross-validation)...")
            result = model.run_pipeline_prophet(
                param_grid=self.param_grid,
                min_train_periods=min(5, len(data_global) - 2),
                horizon=1,
                step=1,
                alpha=0.10,
                forecast_steps=5
            )

            if result is None:
                logger.error("❌ Pipeline failed")
                return False

            # 5. Lưu model
            final_model = model.train_final_model(
                best_params=result["best_params"],
                holiday_years=result["holiday_years"]
            )

            model_path = self._save_model(
                model=final_model,
                model_name="global"
            )

            # 6. Cập nhật metadata
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
        Huấn luyện saved cho mỗi region

        Returns:
        --------
        Dict[str, bool] : {region_name: success_flag}
        """
        logger.info("\n" + "="*70)
        logger.info("TRAINING REGION MODELS")
        logger.info("="*70)

        # Lấy danh sách regions
        regions = sorted(self.df['Region'].dropna().unique())
        logger.info(f"Found {len(regions)} regions: {regions}")

        results = {}

        for region_name in regions:
            logger.info(f"\n--- Region: {region_name} ---")

            try:
                # 1. Kiểm tra đã train rồi?
                if region_name in self.trained_regions:
                    logger.info(f"⚠️  Already trained, skipping...")
                    results[region_name] = True
                    continue

                # 2. Chuẩn bị dữ liệu
                logger.info(f"Preparing data for region {region_name}...")
                data_region = prepare_level_data(
                    df=self.df,
                    level="region",
                    filter=region_name,
                    year_start=self.year_start,
                    year_end=self.year_end,
                    fill_missing_years=True
                )

                logger.info(f"  - Data shape: {data_region.shape}")
                logger.info(f"  - Years: {data_region['Year'].min()} - {data_region['Year'].max()}")

                # 3. Kiểm tra đủ dữ liệu
                if len(data_region) < 5:
                    logger.warning(f"❌ Not enough data for region {region_name}")
                    results[region_name] = False
                    continue

                # 4. Khởi tạo ProphetModel
                logger.info("Initializing ProphetModel...")
                model = ProphetModel(
                    df=data_region,
                    value_col="Mortality_Rate",
                    date_col="Year"
                )

                # 5. Chạy pipeline
                logger.info("Running Prophet pipeline...")
                result = model.run_pipeline_prophet(
                    param_grid=self.param_grid,
                    min_train_periods=min(5, len(data_region) - 2),
                    horizon=1,
                    step=1,
                    alpha=0.10,
                    forecast_steps=5
                )

                if result is None:
                    logger.error(f"❌ Pipeline failed for region {region_name}")
                    results[region_name] = False
                    continue

                # 6. Lưu model
                final_model = model.train_final_model(
                    best_params=result["best_params"],
                    holiday_years=result["holiday_years"]
                )

                final_model = model.train_final_model(
                    best_params=result["best_params"],
                    holiday_years=result["holiday_years"]
                )

                model_path = self._save_model(
                    model=final_model,
                    model_name=f"region_{region_name.replace(' ', '_')}"
                )

                # 7. Cập nhật metadata & trained_regions
                self.metadata["regions"][region_name] = str(model_path)
                self.trained_regions.add(region_name)

                logger.info(f"✅ Region model trained successfully")
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
        Huấn luyện saved cho mỗi country (nếu đủ điều kiện)

        Logic:
        - Đếm số năm có Total_Deaths > 0
        - Nếu >= min_years_for_country_model (23): Train riêng
        - Nếu < 23: Fallback sang region model

        Returns:
        --------
        Tuple[Dict[str, bool], Dict[str, str]] :
            (train_results, fallback_mapping)
            - train_results: {country_name: success_flag}
            - fallback_mapping: {country_name: region_name}
        """
        logger.info("\n" + "="*70)
        logger.info("TRAINING COUNTRY MODELS")
        logger.info("="*70)

        # Lấy danh sách countries
        countries = sorted(self.df['Country'].dropna().unique())
        logger.info(f"Found {len(countries)} countries")

        train_results = {}
        fallback_mapping = {}

        for country_name in countries:
            logger.info(f"\n--- Country: {country_name} ---")

            try:
                # 0. Lấy region của country
                country_data_all = self.df[self.df['Country'] == country_name]
                region_name = country_data_all['Region'].iloc[0]
                logger.info(f"Region: {region_name}")

                # 1. Chuẩn bị dữ liệu
                logger.info(f"Preparing data for country {country_name}...")
                data_country = prepare_level_data(
                    df=self.df,
                    level="country",
                    filter=country_name,
                    year_start=self.year_start,
                    year_end=self.year_end,
                    fill_missing_years=True
                )

                logger.info(f"  - Data shape: {data_country.shape}")
                logger.info(f"  - Years available: {data_country['Year'].min()} - {data_country['Year'].max()}")

                # 2. Đếm số năm có Total_Deaths > 0
                years_with_deaths = (data_country['Total_Deaths'] > 0).sum()
                logger.info(f"  - Years with Total_Deaths > 0: {years_with_deaths}")

                # 3. Kiểm tra điều kiện
                if years_with_deaths < self.min_years_for_country_model:
                    logger.info(
                        f"⚠️  Not enough years ({years_with_deaths} < {self.min_years_for_country_model})"
                    )
                    logger.info(f"    → Will use fallback to region: {region_name}")

                    # Lưu fallback mapping
                    fallback_mapping[country_name] = region_name
                    self.fallback_models[country_name] = region_name

                    # Cập nhật metadata
                    self.metadata["countries"][country_name] = f"fallback_{region_name}"

                    train_results[country_name] = False
                    continue

                # 4. Train model riêng
                if len(data_country) < 5:
                    logger.warning(f"❌ Not enough data points for country {country_name}")

                    fallback_mapping[country_name] = region_name
                    self.fallback_models[country_name] = region_name
                    self.metadata["countries"][country_name] = f"fallback_{region_name}"

                    train_results[country_name] = False
                    continue

                # 5. Khởi tạo ProphetModel
                logger.info("Initializing ProphetModel...")
                model = ProphetModel(
                    df=data_country,
                    value_col="Mortality_Rate",
                    date_col="Year"
                )

                # 6. Chạy pipeline
                logger.info("Running Prophet pipeline...")
                result = model.run_pipeline_prophet(
                    param_grid=self.param_grid,
                    min_train_periods=min(5, len(data_country) - 2),
                    horizon=1,
                    step=1,
                    alpha=0.10,
                    forecast_steps=5
                )

                if result is None:
                    logger.error(f"❌ Pipeline failed for country {country_name}")

                    fallback_mapping[country_name] = region_name
                    self.fallback_models[country_name] = region_name
                    self.metadata["countries"][country_name] = f"fallback_{region_name}"

                    train_results[country_name] = False
                    continue

                # 7. Lưu model
                final_model = model.train_final_model(
                    best_params=result["best_params"],
                    holiday_years=result["holiday_years"]
                )

                model_path = self._save_model(
                    model=final_model,
                    model_name=f"country_{country_name.replace(' ', '_')}"
                )

                # 8. Cập nhật metadata
                self.metadata["countries"][country_name] = str(model_path)

                logger.info(f"✅ Country model trained successfully")
                logger.info(f"   Saved to: {model_path}")

                train_results[country_name] = True

            except Exception as e:
                logger.error(f"❌ Error training country {country_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

                try:
                    region_name = self.df[self.df['Country'] == country_name]['Region'].iloc[0]
                    fallback_mapping[country_name] = region_name
                    self.fallback_models[country_name] = region_name
                    self.metadata["countries"][country_name] = f"fallback_{region_name}"
                except:
                    pass

                train_results[country_name] = False

        return train_results, fallback_mapping

    def train_all(self) -> Dict:
        """
        Huấn luyện tất cả: Global → Region → Country

        Returns:
        --------
        Dict : Kết quả tổng hợp
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE TRAINING PIPELINE: GLOBAL → REGION → COUNTRY")
        logger.info("="*80)

        results = {
            "global": None,
            "regions": {},
            "countries": {},
            "fallback_count": 0
        }

        # 1. Train Global
        global_success = self.train_global_model()
        results["global"] = "✅ SUCCESS" if global_success else "❌ FAILED"

        # 2. Train Regions
        region_results = self.train_region_models()
        successful_regions = sum(1 for v in region_results.values() if v)
        results["regions"] = region_results
        logger.info(f"\nRegion Summary: {successful_regions}/{len(region_results)} successful")

        # 3. Train Countries
        country_results, fallback_mapping = self.train_country_models()
        successful_countries = sum(1 for v in country_results.values() if v)
        fallback_count = len(fallback_mapping)

        results["countries"] = country_results
        results["fallback_count"] = fallback_count

        logger.info(f"\nCountry Summary: {successful_countries}/{len(country_results)} trained")
        logger.info(f"                 {fallback_count} using fallback to region saved")

        # 4. Lưu metadata
        self._save_metadata()

        # 5. In tóm tắt
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED - SUMMARY")
        logger.info("="*80)
        logger.info(f"Global model:       {results['global']}")
        logger.info(f"Region saved:      {successful_regions}/{len(region_results)} trained")
        logger.info(f"Country saved:     {successful_countries}/{len(country_results)} trained")
        logger.info(f"Fallback countries: {fallback_count}")
        logger.info(f"Metadata saved to:  {self.model_dir}/metadata.json")
        logger.info(f"All saved saved to: {self.model_dir}/")
        logger.info("="*80 + "\n")

        return results

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _save_model(self, model, model_name: str) -> Path:
        """
        Lưu model vào file pickle

        Parameters:
        -----------
        model : ProphetModel hoặc object
            Model object cần lưu
        model_name : str
            Tên model (sẽ thành {model_name}.pkl)

        Returns:
        --------
        Path : Đường dẫn file đã lưu
        """
        model_path = self.model_dir / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        logger.debug(f"Saved model to {model_path} ({file_size:.2f} MB)")

        return model_path

    def _save_metadata(self) -> None:
        """
        Lưu metadata (mapping & fallback info) vào JSON file

        Structure:
        {
            "global": "path/to/global.pkl",
            "regions": {
                "region_name": "path/to/region_region_name.pkl",
                ...
            },
            "countries": {
                "country_name": "path/to/country_country_name.pkl"
                           OR
                "country_name": "fallback_region_name"
            }
        }
        """
        metadata_path = self.model_dir / "metadata.json"

        # Chuyển metadata để JSON serializable
        metadata_json = {
            "global": self.metadata["global"],
            "regions": self.metadata["regions"],
            "countries": self.metadata["countries"],
            "fallback_models": self.fallback_models,
            "training_config": {
                "year_start": self.year_start,
                "year_end": self.year_end,
                "min_years_for_country_model": self.min_years_for_country_model,
                "param_grid": self.param_grid
            }
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata to {metadata_path}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_from_pkl(model_path: str):
    """
    Load model từ file pickle

    Parameters:
    -----------
    model_path : str
        Đường dẫn tới file pickle

    Returns:
    --------
    object : Model object
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_metadata(model_dir: str = "model") -> Dict:
    """
    Load metadata từ JSON file

    Parameters:
    -----------
    model_dir : str
        Đường dẫn thư mục saved

    Returns:
    --------
    Dict : Metadata
    """
    metadata_path = Path(model_dir) / "metadata.json"

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


# ============================================================================
# MAIN (EXAMPLE USAGE)
# ============================================================================

if __name__ == "__main__":
    # Ví dụ sử dụng (giả sử df đã được load)
    """
    from src.processing.aggregate import prepare_level_data
    
    # Load dữ liệu
    df = pd.read_csv('data/time_series_country.csv')
    
    # Định nghĩa param grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05],
        'seasonality_prior_scale': [0.01, 0.1, 1.0],
    }
    
    # Tạo TrainModel instance
    trainer = TrainModel(
        df=df,
        param_grid=param_grid,
        model_dir="model",
        year_start=2000,
        year_end=2024,
        min_years_for_country_model=23
    )
    
    # Train all
    results = trainer.train_all()
    
    # Load metadata sau
    metadata = load_metadata("model")
    print(json.dumps(metadata, indent=2))
    """
    pass

