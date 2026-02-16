import numpy as np
import pandas as pd
from typing import Tuple, Iterator, Optional, Dict

def expanding_splits(
    df: pd.DataFrame,
    min_train_periods: int = 15,
    horizon: int = 1,
    step: int = 1
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yield nhiều cặp (train_df, test_df) theo expanding window.

    - Train bắt đầu từ năm đầu và mở rộng dần.
    - Test là horizon năm ngay sau train.
    - Mỗi fold nhảy thêm step năm.

    Output mỗi fold:
    - train_df: chỉ gồm ['Year', 'Mortality_Rate']
    - test_df : chỉ gồm ['Year', 'Mortality_Rate']
    """

    # Copy + chuẩn hóa năm + sort
    df2 = df.copy()
    df2["Year"] = pd.to_numeric(df2["Year"], errors="coerce").astype("Int64")
    df2 = df2.dropna(subset=["Year"]).sort_values("Year")

    # Trục thời gian (các năm duy nhất)
    years = df2["Year"].astype(int).unique()
    n_years = len(years)

    if n_years < (min_train_periods + horizon):
        raise ValueError(
            f"Không đủ số năm ({n_years}). Cần ít nhất {min_train_periods + horizon} năm."
        )

    # Chạy expanding: train_end_pos chạy từ (min_train_periods-1) đến tối đa
    train_end_pos = min_train_periods - 1
    max_train_end_pos = n_years - horizon - 1

    while train_end_pos <= max_train_end_pos:
        train_end_year = years[train_end_pos]
        test_start_year = years[train_end_pos + 1]
        test_end_year = years[train_end_pos + horizon]

        train_df = df2[df2["Year"] <= train_end_year].copy()
        test_df = df2[(df2["Year"] >= test_start_year) & (df2["Year"] <= test_end_year)].copy()
        yield train_df, test_df

        train_end_pos += step