import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.split import expanding_splits


class ARIMAModel:

    def __init__(self, df: pd.DataFrame, value_col: str = "Mortality_Rate"):
        self.df = df.copy()
        self.value_col = value_col
        self.series = self.df[value_col].dropna()

    # ==================================================
    # 1. ADF TEST
    # ==================================================

    def adf_test(self, d: int = 0, verbose: bool = True):
        series_test = self.series.diff(d).dropna() if d > 0 else self.series

        adf_stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(series_test)

        result = {
            "ADF Statistic": float(adf_stat),
            "p-value": float(p_value),
            "Critical Values": {k: float(v) for k, v in critical_values.items()},
            "Is Stationary": p_value <= 0.05
        }

        if verbose:
            print("\n========== ADF TEST RESULT ==========")
            print(f"Differencing (d): {d}")
            print(f"ADF Statistic  : {adf_stat:.6f}")
            print(f"p-value        : {p_value:.6e}")
            print(f"Stationary     : {p_value <= 0.05}")
            print("\nCritical Values:")
            for key, value in critical_values.items():
                print(f"   {key} : {value:.6f}")
            print("=====================================\n")

        return result

    # ==================================================
    # 2. ACF & PACF
    # ==================================================

    def plot_acf_pacf(self, lags: int = 10):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(self.series, lags=lags, ax=axes[0])
        plot_pacf(self.series, lags=lags, ax=axes[1])
        axes[0].set_title("ACF")
        axes[1].set_title("PACF")
        plt.tight_layout()
        plt.show()

    # ==================================================
    # 3. EXPANDING CROSS-VALIDATION
    # ==================================================

    def evaluate_expanding_cv(self,
                              p: int,
                              d: int,
                              q: int,
                              min_train_periods: int = 15,
                              horizon: int = 1,
                              step: int = 1,
                              alpha: float = 0.10):

        all_actuals = []
        all_forecasts = []
        all_lowers = []
        all_uppers = []

        for train_df, test_df in expanding_splits(
                self.df,
                min_train_periods=min_train_periods,
                horizon=horizon,
                step=step):

            train_series = train_df[self.value_col]
            test_series = test_df[self.value_col]

            try:
                model = ARIMA(train_series, order=(p, d, q))
                result = model.fit()

                forecast_res = result.get_forecast(steps=len(test_series))
                forecast = forecast_res.predicted_mean
                conf_int = forecast_res.conf_int(alpha=alpha)

                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values

                all_actuals.extend(test_series.values)
                all_forecasts.extend(forecast.values)
                all_lowers.extend(lower)
                all_uppers.extend(upper)

            except:
                continue

        if len(all_actuals) == 0:
            return {
                "Model": (p, d, q),
                "MAE": None,
                "RMSE": None,
                "MASE": None,
                f"Coverage_{int((1 - alpha) * 100)}%": None,
                "n_forecasts": 0
            }

        all_actuals = np.array(all_actuals)
        all_forecasts = np.array(all_forecasts)
        all_lowers = np.array(all_lowers)
        all_uppers = np.array(all_uppers)

        errors = all_actuals - all_forecasts

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        r2 = r2_score(all_actuals, all_forecasts)

        naive = self.series.shift(1).dropna()
        mae_naive = np.mean(np.abs(self.series[1:] - naive))
        mase = mae / mae_naive if mae_naive != 0 else None

        inside = np.logical_and(all_actuals >= all_lowers,
                                all_actuals <= all_uppers)
        coverage = np.mean(inside)

        return {
            "Model": (p, d, q),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MASE": round(mase, 4) if mase else None,
            f"Coverage_{int((1 - alpha) * 100)}%": round(coverage, 4),
            "n_forecasts": len(all_actuals)
        }

    # ==================================================
    # 4. GRID SEARCH (EXPANDING BASED)
    # ==================================================

    def grid_search_expanding(self,
                              p_list,
                              d_list,
                              q_list,
                              min_train_periods: int = 15,
                              horizon: int = 1,
                              step: int = 1,
                              alpha: float = 0.10):

        results = []

        for p in p_list:
            for d in d_list:
                for q in q_list:

                    cv_metrics = self.evaluate_expanding_cv(
                        p=p,
                        d=d,
                        q=q,
                        min_train_periods=min_train_periods,
                        horizon=horizon,
                        step=step,
                        alpha=alpha
                    )

                    if cv_metrics["n_forecasts"] == 0:
                        continue

                    try:
                        full_model = ARIMA(self.series, order=(p, d, q))
                        full_result = full_model.fit()
                        aic_value = full_result.aic
                    except:
                        continue

                    results.append({
                        "Order": (p, d, q),
                        "AIC": round(aic_value, 2),
                        "MAE": cv_metrics["MAE"],
                        "RMSE": cv_metrics["RMSE"],
                        "MASE": cv_metrics["MASE"],
                        f"Coverage_{int((1 - alpha) * 100)}%":
                            cv_metrics[f"Coverage_{int((1 - alpha) * 100)}%"],
                        "n_forecasts": cv_metrics["n_forecasts"]
                    })

        df_results = pd.DataFrame(results)

        if not df_results.empty:
            df_results = df_results.sort_values("MAE").reset_index(drop=True)

        return df_results

    # ==================================================
    # 5. FORECAST FUTURE
    # ==================================================

    def arima_forecast(self, p, d, q, steps: int = 5):

        model = ARIMA(self.series, order=(p, d, q))
        result = model.fit()

        forecast = result.get_forecast(steps=steps)
        return forecast.summary_frame(alpha=0.05)

    # ==================================================
    # 6. PLOT FORECAST
    # ==================================================

    def plot_forecast(self, p, d, q, steps: int = 5):

        model = ARIMA(self.series, order=(p, d, q))
        result = model.fit()

        forecast_res = result.get_forecast(steps=steps)
        forecast_df = forecast_res.summary_frame(alpha=0.05)

        # Nếu có cột Year thì dùng Year làm trục x
        if "Year" in self.df.columns:
            historical_x = self.df["Year"].values
            last_year = historical_x[-1]
            forecast_x = np.arange(last_year + 1, last_year + 1 + steps)
        else:
            historical_x = np.arange(len(self.series))
            forecast_x = np.arange(len(self.series), len(self.series) + steps)

        plt.figure(figsize=(10, 5))

        # Historical
        plt.plot(historical_x,
                 self.series.values,
                 label="Historical Data")

        # Forecast mean
        plt.plot(forecast_x,
                 forecast_df["mean"].values,
                 label="Forecast")

        # Confidence interval
        plt.fill_between(
            forecast_x,
            forecast_df["mean_ci_lower"].values,
            forecast_df["mean_ci_upper"].values,
            alpha=0.2
        )

        plt.title(f"ARIMA({p},{d},{q}) Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()

