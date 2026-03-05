import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.split import expanding_splits


class ProphetModel:

    def __init__(self,
                 df: pd.DataFrame,
                 value_col: str = "Mortality_Rate",
                 date_col: str = "Year"):

        self.df = df.copy()
        self.value_col = value_col
        self.date_col = date_col

        # ==================================================
        # 1. VISUALIZE HOLIDAY (IQR)
        # ==================================================

    def plot_dashboard(self, df_plot=None, target_name="", holiday_years=None,
                       year_col=None, deaths_col="Total_Deaths", pop_col="Population"):

        import matplotlib.pyplot as plt

        if df_plot is None:
            df_plot = self.df.copy()

        if year_col is None:
            year_col = self.date_col

        holiday_years = set(holiday_years or [])

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # 1) Total deaths
        if deaths_col in df_plot.columns:
            axs[0, 0].bar(df_plot[year_col], df_plot[deaths_col])
            axs[0, 0].set_title(f"Total Deaths in {target_name}".strip())
            for y in holiday_years:
                axs[0, 0].axvline(y, linestyle="--", alpha=0.6)

        # 2) Mortality rate
        axs[0, 1].plot(df_plot[year_col], df_plot[self.value_col], marker="o")
        axs[0, 1].set_title(f"Mortality Rate in {target_name}".strip())
        if holiday_years:
            hol = df_plot[df_plot[year_col].isin(holiday_years)]
            axs[0, 1].scatter(hol[year_col], hol[self.value_col], color="red", s=80, zorder=3)

        # 3) Population
        if pop_col in df_plot.columns:
            axs[1, 0].bar(df_plot[year_col], df_plot[pop_col])
            axs[1, 0].set_title(f"Population Trend in {target_name}".strip())
            for y in holiday_years:
                axs[1, 0].axvline(y, linestyle="--", alpha=0.6)

        # 4) Pop vs mortality
        if pop_col in df_plot.columns:
            axs[1, 1].scatter(df_plot[pop_col], df_plot[self.value_col], alpha=0.7)
            axs[1, 1].set_title(f"Population vs Mortality Rate in {target_name}".strip())
            if holiday_years:
                axs[1, 1].scatter(hol[pop_col], hol[self.value_col], color="red", s=120)

        for ax in axs.flat:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    # ==================================================
    # 1. CREATE HOLIDAY FROM YEAR LIST
    # ==================================================

    def _build_holidays(self, holiday_years):

        if holiday_years is None:
            return None

        return pd.DataFrame({
            "holiday": "custom_event",
            "ds": pd.to_datetime(
                [f"{y}-12-31" for y in holiday_years]
            ),
            "lower_window": 0,
            "upper_window": 0
        })

    # ==================================================
    # 2. TO PROPHET FORMAT
    # ==================================================

    def _to_prophet(self, df):

        df_p = df.copy()
        df_p["ds"] = pd.to_datetime(
            df_p[self.date_col].astype(str) + "-12-31"
        )
        df_p["y"] = df_p[self.value_col]

        return df_p[["ds", "y"]]

    # ==================================================
    # 3. EXPANDING CROSS VALIDATION
    # ==================================================

    def evaluate_expanding_cv(self,
                              min_train_periods: int = 15,
                              horizon: int = 1,
                              step: int = 1,
                              alpha: float = 0.10,
                              holiday_years=None,
                              **prophet_params):

        holidays = self._build_holidays(holiday_years)

        all_actuals = []
        all_forecasts = []
        all_lowers = []
        all_uppers = []

        for train_df, test_df in expanding_splits(
                self.df,
                min_train_periods=min_train_periods,
                horizon=horizon,
                step=step):

            train_p = self._to_prophet(train_df)
            test_p = self._to_prophet(test_df)

            try:
                model = Prophet(
                    growth="linear",
                    seasonality_mode="additive",
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    holidays=holidays,
                    interval_width=1 - alpha,
                    **prophet_params
                )

                model.fit(train_p)

                forecast = model.predict(test_p[["ds"]])

                all_actuals.extend(test_p["y"].values)
                all_forecasts.extend(forecast["yhat"].values)
                all_lowers.extend(forecast["yhat_lower"].values)
                all_uppers.extend(forecast["yhat_upper"].values)

            except:
                continue

        if len(all_actuals) == 0:
            return pd.DataFrame([{
                "Model": "Prophet",
                "MAE": None,
                "RMSE": None,
                "MASE": None,
                f"Coverage_{int((1 - alpha) * 100)}%": None,
                "n_forecasts": 0
            }])

        all_actuals = np.array(all_actuals)
        all_forecasts = np.array(all_forecasts)
        all_lowers = np.array(all_lowers)
        all_uppers = np.array(all_uppers)

        errors = all_actuals - all_forecasts

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        r2 = r2_score(all_actuals, all_forecasts)

        # MASE (naive lag-1)
        naive = self.df[self.value_col].shift(1).dropna()
        mae_naive = np.mean(
            np.abs(self.df[self.value_col][1:] - naive)
        )
        mase = mae / mae_naive if mae_naive != 0 else None

        inside = np.logical_and(
            all_actuals >= all_lowers,
            all_actuals <= all_uppers
        )
        coverage = np.mean(inside)

        return pd.DataFrame([{
            "Model": "Prophet",
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MASE": round(mase, 4) if mase is not None else None,
            f"Coverage_{int((1 - alpha) * 100)}%": round(coverage, 4),
            "n_forecasts": len(all_actuals)
        }])

    # ==================================================
    # 4. GRID SEARCH
    # ==================================================

    def grid_search_expanding(self,
                              param_grid,
                              min_train_periods: int = 15,
                              horizon: int = 1,
                              step: int = 1,
                              alpha: float = 0.10,
                              holiday_years=None):

        keys, values = zip(*param_grid.items())
        combinations = [
            dict(zip(keys, v))
            for v in itertools.product(*values)
        ]

        results = []

        for params in combinations:

            df_metrics = self.evaluate_expanding_cv(
                min_train_periods=min_train_periods,
                horizon=horizon,
                step=step,
                alpha=alpha,
                holiday_years=holiday_years,
                **params
            )

            row = {**params, **df_metrics.iloc[0].to_dict()}
            results.append(row)

        return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

    # ==================================================
    # 5. FORECAST FUTURE
    # ==================================================

    def prophet_forecast(self,
                         steps: int = 6,
                         holiday_years=None,
                         **prophet_params):

        holidays = self._build_holidays(holiday_years)

        df_p = self._to_prophet(self.df)

        model = Prophet(
            growth="linear",
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            **prophet_params
        )

        model.fit(df_p)

        future = model.make_future_dataframe(
            periods=steps,
            freq="YE"
        )

        forecast = model.predict(future)
        df= forecast.tail(steps)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        return df

    def run_pipeline_prophet(self,
                             param_grid: dict,
                             holiday_years=None,
                             min_train_periods: int = 15,
                             horizon: int = 1,
                             step: int = 1,
                             alpha: float = 0.10,
                             forecast_steps: int = 5):
        """
        Pipeline Prophet:
        1. Detect outlier bằng visualize pick
        2. Lấy year outlier làm holiday_years
        3. Grid search
        4. Chọn best model theo MAE
        5. Xuất kết quả best model (summary DataFrame)
        """

        print("========== PROPHET PIPELINE START ==========")

        # 1. Holidays pick
        if holiday_years is not None:
            print("Manual holiday years:", holiday_years)
        else:
            print("No holiday years provided (manual).")

        # 2. Grid search
        print("Running grid search...")

        cv_results = self.grid_search_expanding(
            param_grid=param_grid,
            min_train_periods=min_train_periods,
            horizon=horizon,
            step=step,
            alpha=alpha,
            holiday_years=holiday_years
        )

        if cv_results.empty:
            print("No valid saved found.")
            return None

        # 3. Best model theo MAE
        cv_results = cv_results.sort_values("MAE").reset_index(drop=True)
        best_row = cv_results.iloc[0]

        print("\nBest model based on MAE:")
        print(best_row)

        # 4. Extract best params (loại metric columns)
        coverage_col = f"Coverage_{int((1 - alpha) * 100)}%"
        metric_cols = [
            "Model", "MAE", "RMSE", "MASE",
            coverage_col, "n_forecasts"
        ]

        best_params = {
            k: v for k, v in best_row.items()
            if k not in metric_cols
        }

        print("\nBest Prophet parameters:")
        print(best_params)

        # 5. Train best model full data + forecast
        forecast_future = self.prophet_forecast(
            steps=forecast_steps,
            holiday_years=holiday_years,
            **best_params
        )

        print("========== PIPELINE FINISHED ==========")

        return {
            "cv_results": cv_results,
            "best_params": best_params,
            "best_model_summary": pd.DataFrame([best_row]),
            "future_forecast": forecast_future,
            "holiday_years": holiday_years
        }

    def train_final_model(self,
                          best_params: dict,
                          holiday_years=None):
        """
        Train Prophet model cuối cùng trên toàn bộ dữ liệu
        (dùng để save production model)

        Parameters
        ----------
        best_params : dict
            Tham số tối ưu sau grid search
        holiday_years : list, optional
            Danh sách năm outlier để tạo holiday

        Returns
        -------
        Prophet object (đã fit)
        """

        holidays = self._build_holidays(holiday_years)

        df_p = self._to_prophet(self.df)

        final_model = Prophet(
            growth="linear",
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            **best_params
        )

        final_model.fit(df_p)

        return final_model

    # ==================================================
    # 6. PLOT FORECAST (FINAL CLEAN VERSION)
    # ==================================================

    def plot_forecast(self,
                      steps: int = 5,
                      holiday_years=None,
                      **prophet_params):
        """
        Plot gồm:
        - Observed data points
        - Historical fitted line
        - Continuous uncertainty interval
        - Forecast line (nối từ observed cuối)
        - Forecast confidence interval (đậm hơn)
        """

        holidays = self._build_holidays(holiday_years)
        df_p = self._to_prophet(self.df)

        model = Prophet(
            growth="linear",
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            **prophet_params
        )

        model.fit(df_p)

        future = model.make_future_dataframe(
            periods=steps,
            freq="YE"
        )

        forecast = model.predict(future)

        forecast_future = forecast.tail(steps)
        historical_df = forecast.iloc[:-steps]

        # ====== PLOT (reworked) ======
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # 1) Uncertainty interval toàn bộ (nhạt, liền mạch)
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            alpha=0.12,
            label="Uncertainty interval",
            zorder=1,
        )

        # 2) Historical fitted line (yhat trên lịch sử)
        ax.plot(
            historical_df["ds"],
            historical_df["yhat"],
            linewidth=2,
            label="Historical fitted",
            zorder=2,
        )

        # 3) Forecast line nối mượt từ yhat lịch sử cuối (không nối từ y thật để tránh gãy)
        last_hist_ds = historical_df["ds"].iloc[-1]
        last_hist_yhat = historical_df["yhat"].iloc[-1]

        extended_x = pd.concat([pd.Series([last_hist_ds]), forecast_future["ds"]], ignore_index=True)
        extended_y = pd.concat([pd.Series([last_hist_yhat]), forecast_future["yhat"]], ignore_index=True)

        ax.plot(
            extended_x,
            extended_y,
            linewidth=2,
            label="Forecast",
            zorder=3,
        )

        # 4) Forecast confidence interval (đậm hơn thật)
        ax.fill_between(
            forecast_future["ds"],
            forecast_future["yhat_lower"],
            forecast_future["yhat_upper"],
            alpha=0.35,  # <-- đậm hơn interval nền
            label="Forecast confidence interval",
            zorder=2,
        )

        # 5) Observed points (vẽ sau cùng cho nổi lên)
        ax.scatter(
            df_p["ds"],
            df_p["y"],
            color="black",
            s=25,
            label="Observed",
            zorder=4,
        )

        # 6) Formatting + legend dedupe
        ax.set_title("Mortality Rate Forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mortality Rate")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    # ==================================================
    # 7. POST-PROCESSING & BENCHMARKS
    # ==================================================

    def process_and_print_forecast(self, forecast_df):
        """
        Xử lý yhat_lower < 0 thành 0, tính mean cho 7 dòng cuối và in kết quả.
        """
        # Tạo bản sao để tránh SettingWithCopyWarning
        df_cleaned = forecast_df.copy()

        # 1. Chuyển các giá trị âm ở yhat_lower về 0
        df_cleaned['yhat_lower'] = df_cleaned['yhat_lower'].clip(lower=0)

        # 2. Tính toán mean của yhat_lower và yhat_upper cho 7 dòng cuối
        # .iloc[-7:] lấy 7 dòng cuối cùng có trong DataFrame
        mean_lower_last_6 = df_cleaned['yhat_lower'].iloc[-6:].mean()
        mean_upper_last_6 = df_cleaned['yhat_upper'].iloc[-6:].mean()

        # 3. In bảng kết quả giống format yêu cầu
        print("\n Mortality Rate Forecast (2025-2030) - cắt âm ")
        pd.options.display.max_columns = None
        print(df_cleaned[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # 4. In thêm giá trị trung bình CI theo yêu cầu của bạn
        print(f"Mean yhat_lower (last 6): {mean_lower_last_6:.6f}")
        print(f"Mean yhat_upper (last 6): {mean_upper_last_6:.6f}")
        return df_cleaned

    # ==================================================
    # 8. BENCHMARK STRATEGY 1: MEAN COMPARISON
    # ==================================================
    def evaluate_benchmark_mean(self, forecast_df):
        """
        Benchmark Strategy 1: So sánh giá trị trung bình 2025-2030 với lịch sử.
        """
        # 1. Tính toán lịch sử (toàn bộ dữ liệu train)
        historical_mean = self.df[self.value_col].mean()
        print(f"Historical Mean Mortality Rate: {historical_mean:.4f}")

        # 2. Lọc dữ liệu dự báo 2025 - 2030
        # Đảm bảo ds là datetime để lọc theo năm
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        forecast_25_30 = forecast_df[(forecast_df['ds'].dt.year >= 2025) & (forecast_df['ds'].dt.year <= 2030)]

        forecast_mean = forecast_25_30['yhat'].mean()
        print(f"Forecast Mean Mortality Rate (2025-2030): {forecast_mean:.4f}")

        # 3. Ngưỡng cắt giảm
        reduction_25 = historical_mean * 0.75
        reduction_50 = historical_mean * 0.50
        print(f"25% Reduction Threshold (Target <= {reduction_25:.4f})")
        print(f"50% Reduction Threshold (Target <= {reduction_50:.4f})")

        # 4. Kiểm tra
        print(f"Yes: 25% reduction" if forecast_mean <= reduction_25 else "No: 25% reduction")
        print(f"Yes: 50% reduction" if forecast_mean <= reduction_50 else "No: 50% reduction")

    # ==================================================
    # 9.Benchmark Strategy 2: So sánh tốc độ thay đổi hàng năm (YoY).
    # ==================================================
    def evaluate_benchmark_yoy(self, forecast_df):
        """
        Benchmark Strategy 2: So sánh tốc độ thay đổi hàng năm (YoY).
        Chỉ in Benchmark và Average Forecast, KHÔNG CÓ kết luận.
        """
        # 1. Benchmark lịch sử: Trung bình thay đổi hàng năm
        hist_yoy_change = self.df[self.value_col].diff().dropna().mean()
        print(f"Historical Average YoY Change (Benchmark): {hist_yoy_change:.6f}")

        # 2. Dự báo YoY: Tính từ 2024 để có bước nhảy sang 2025
        f_data = forecast_df.copy()
        f_data['ds'] = pd.to_datetime(f_data['ds'])

        # Lọc lấy từ 2024 đến 2030 để tính diff
        f_data = f_data[(f_data['ds'].dt.year >= 2024) & (f_data['ds'].dt.year <= 2030)]
        f_data['year_on_year_change'] = f_data['yhat'].diff()

        print("\nForecasted YoY Changes:")
        print(f_data[['ds', 'yhat', 'year_on_year_change']])

        # 3. Tính trung bình thay đổi của dự báo (bỏ dòng đầu vì là NaN)
        average_forecast_change = f_data['year_on_year_change'].dropna().mean()
        print(f"\nAverage Forecasted YoY Change: {average_forecast_change:.6f}")

