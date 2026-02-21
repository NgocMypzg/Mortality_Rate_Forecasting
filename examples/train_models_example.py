"""
Script ví dụ sử dụng TrainModel class

Chạy: python examples/train_models_example.py
"""

import pandas as pd
import json
from pathlib import Path
from src.model.train_model import TrainModel, load_model_from_pkl, load_metadata


def main():
    """
    Ví dụ hoàn chỉnh: Load dữ liệu → Train saved → Load metadata
    """

    print("\n" + "="*80)
    print("EXAMPLE: Using TrainModel Class")
    print("="*80 + "\n")

    # ========================================================================
    # STEP 1: Load dữ liệu
    # ========================================================================

    print("[STEP 1] Loading data...")

    # Giả sử bạn có file CSV
    data_path = "../data/time_series_country.csv"

    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Countries: {df['Country'].nunique()}")
        print(f"   Regions: {df['Region'].nunique()}")
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        print("   Please provide a CSV file with columns: Year, Country, Region, Total_Deaths, Population")
        return

    # ========================================================================
    # STEP 2: Định nghĩa param grid
    # ========================================================================

    print("\n[STEP 2] Defining param grid for Prophet...")

    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3, 0.5],  # Removed 0
        'n_changepoints': [15, 20, 25],
        'changepoint_range': [0.8, 0.9],
        'holidays_prior_scale': [10.0, 15.0]
    }

    print(f"✅ Param grid defined:")
    print("Param grid defined:")
    for key, value in param_grid.items():
        print(f"   {key}: {value}")
    # ========================================================================
    # STEP 3: Tạo TrainModel instance
    # ========================================================================

    print("\n[STEP 3] Creating TrainModel instance...")

    trainer = TrainModel(
        df=df,
        param_grid=param_grid,
        model_dir="../saved",
        year_start=2000,
        year_end=2024,
        min_years_for_country_model=23
    )

    print("✅ TrainModel instance created")
    print(f"   Model directory: {trainer.model_dir}")
    print(f"   Year range: {trainer.year_start}-{trainer.year_end}")
    print(f"   Min years for country model: {trainer.min_years_for_country_model}")

    # ========================================================================
    # STEP 4: Train all saved
    # ========================================================================

    print("\n[STEP 4] Starting training...")
    print("This may take 30-60 minutes depending on data size\n")

    results = trainer.train_all()

    # ========================================================================
    # STEP 5: Hiển thị kết quả
    # ========================================================================

    print("\n[STEP 5] Training results:")
    print(f"   Global: {results['global']}")
    print(f"   Regions: {sum(1 for v in results['regions'].values() if v)}/{len(results['regions'])} trained")
    print(f"   Countries: {sum(1 for v in results['countries'].values() if v)}/{len(results['countries'])} trained")
    print(f"   Fallback: {results['fallback_count']} countries using region saved")

    # ========================================================================
    # STEP 6: Load và check metadata
    # ========================================================================

    print("\n[STEP 6] Loading metadata...")

    metadata = load_metadata("model")

    print("✅ Metadata loaded")
    print(f"   Global model: {metadata['global']}")
    print(f"   Region saved: {len(metadata['regions'])} regions")
    print(f"   Country saved: {len(metadata['countries'])} countries")

    # ========================================================================
    # STEP 7: Ví dụ load một model
    # ========================================================================

    print("\n[STEP 7] Example: Loading a model...")

    if metadata['global']:
        try:
            global_model = load_model_from_pkl(metadata['global'])
            print(f"✅ Global model loaded successfully")
            print(f"   Type: {type(global_model)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    # ========================================================================
    # STEP 8: In fallback mapping
    # ========================================================================

    print("\n[STEP 8] Fallback mapping (countries using region saved):")

    fallback = metadata.get('fallback_models', {})
    if fallback:
        for country, region in list(fallback.items())[:10]:  # Show first 10
            print(f"   {country:20s} → {region}")
        if len(fallback) > 10:
            print(f"   ... and {len(fallback) - 10} more")
    else:
        print("   No fallbacks")

    # ========================================================================
    # STEP 9: In training config
    # ========================================================================

    print("\n[STEP 9] Training configuration:")
    config = metadata.get('training_config', {})
    print(f"   Year range: {config.get('year_start')}-{config.get('year_end')}")
    print(f"   Min years for country model: {config.get('min_years_for_country_model')}")
    print(f"   Param grid: {config.get('param_grid')}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"\nModels saved in: {Path('../saved').absolute()}")
    print(f"Metadata saved in: {Path('../saved/metadata.json').absolute()}")
    print("\nYou can now use these saved for predictions!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

