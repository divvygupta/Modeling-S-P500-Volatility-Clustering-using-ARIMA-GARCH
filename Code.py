import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import datetime as dt
import warnings
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

FIGURE_DIR = "report_figures/report_figures"
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

tickers = ['^GSPC', '^VIX']
start_date = '2004-01-01'

try:
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError("No data fetched. Check ticker symbols or date range.")

    prices_gspc = data['Adj Close']['^GSPC'].dropna()
    volume_gspc = data['Volume']['^GSPC'].dropna()
    prices_vix = data['Close']['^VIX'].dropna()

    common_index = prices_gspc.index.intersection(volume_gspc.index).intersection(prices_vix.index)
    prices_gspc = prices_gspc.loc[common_index]
    volume_gspc = volume_gspc.loc[common_index]
    prices_vix = prices_vix.loc[common_index]

    if prices_gspc.empty or volume_gspc.empty or prices_vix.empty:
         raise ValueError("Data alignment resulted in empty series. Check date overlaps.")

except Exception as e:
    print(f"Error fetching or processing data: {e}")
    exit()

log_returns_gspc = np.log(prices_gspc / prices_gspc.shift(1)).dropna() * 100
common_index_final = log_returns_gspc.index
prices_gspc = prices_gspc.loc[common_index_final]
volume_gspc = volume_gspc.loc[common_index_final]
prices_vix = prices_vix.loc[common_index_final]

print("\n--- Data Summary ---")
print(f"Aligned data points: {len(prices_gspc)}")
print(f"S&P 500 Log Returns calculated: {len(log_returns_gspc)}")
print(f"Latest data date: {prices_gspc.index[-1].date()}")

print("\n--- Performing EDA ---")

# S&P 500 Price Plot
plt.figure(figsize=(12, 6))
plt.plot(prices_gspc.index, prices_gspc)
plt.title('Figure 1: S&P 500 Adjusted Close Price (2004-2025)')
plt.xlabel('Date'); plt.ylabel('Price')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(FIGURE_DIR, 'figure_1_gspc_price.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# S&P 500 Log Returns Plot
plt.figure(figsize=(12, 6))
plt.plot(log_returns_gspc.index, log_returns_gspc)
plt.title('Figure 2: S&P 500 Daily Log Returns (%) (2004-2025)')
plt.xlabel('Date'); plt.ylabel('Log Return (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(FIGURE_DIR, 'figure_2_gspc_log_returns.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# S&P 500 Squared Log Returns Plot
plt.figure(figsize=(12, 6))
plt.plot(log_returns_gspc.index, log_returns_gspc**2)
plt.title('S&P 500 Daily Squared Log Returns (Volatility Proxy)')
plt.xlabel('Date'); plt.ylabel('Squared Log Return (%)^2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(FIGURE_DIR, 'figure_A1_gspc_sq_log_returns.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# VIX Plot
plt.figure(figsize=(12, 6))
plt.plot(prices_vix.index, prices_vix)
plt.title('VIX Index Level (2004-2025)')
plt.xlabel('Date'); plt.ylabel('VIX Level')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(FIGURE_DIR, 'figure_A2_vix_level.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# S&P 500 Volume Plot
plt.figure(figsize=(12, 6))
plt.plot(volume_gspc.index, volume_gspc)
plt.title('S&P 500 Trading Volume (Log Scale)')
plt.xlabel('Date'); plt.ylabel('Volume')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(FIGURE_DIR, 'figure_A3_gspc_volume.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Stationarity Test (ADF) on S&P 500 Log Returns
print("\n--- Stationarity Test (ADF) on S&P 500 Log Returns ---")
adf_test_gspc = adfuller(log_returns_gspc)
print(f'ADF Statistic: {adf_test_gspc[0]:.4f}, p-value: {adf_test_gspc[1]:.4f}')
if adf_test_gspc[1] <= 0.05: print("Conclusion: ^GSPC Log returns likely stationary.")
else: print("Conclusion: ^GSPC Log returns likely non-stationary.")

# Autocorrelation Analysis on S&P 500 Returns
log_returns_gspc2 = log_returns_gspc**2
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(log_returns_gspc, lags=40, ax=axes[0], title='ACF of Daily Log Returns (^GSPC)')
plot_pacf(log_returns_gspc, lags=40, method='ywm', ax=axes[1], title='PACF of Daily Log Returns (^GSPC)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_3_gspc_log_returns_acf_pacf.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(log_returns_gspc2, lags=40, ax=axes[0], title='ACF of Squared Log Returns (^GSPC)')
plot_pacf(log_returns_gspc2, lags=40, method='ywm', ax=axes[1], title='PACF of Squared Log Returns (^GSPC)')
plt.suptitle('Figure 4: Autocorrelation of Squared S&P 500 Log Returns', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_4_gspc_sq_log_returns_acf_pacf.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

# GSPC ARMA(1,1) + GARCH(1,1) Modeling & Validation
p_order_gspc, q_order_gspc = 1, 1
arma_model_gspc = None
garch_results_gspc = None
conditional_vol_gspc = None
std_resid_gspc = None

try:
    arma_model_gspc = ARIMA(log_returns_gspc, order=(p_order_gspc, 0, q_order_gspc)).fit()
    arma_residuals_gspc = arma_model_gspc.resid.loc[log_returns_gspc.index]

    garch_model_spec_gspc = arch_model(arma_residuals_gspc, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
    garch_results_gspc = garch_model_spec_gspc.fit(last_obs=arma_residuals_gspc.index[-1], update_freq=0, disp='off')
    conditional_vol_gspc = garch_results_gspc.conditional_volatility
    std_resid_gspc = garch_results_gspc.std_resid

except Exception as e:
    print(f"Error fitting ^GSPC ARMA/GARCH model: {e}")

if std_resid_gspc is not None:
    std_resid_gspc_clean = std_resid_gspc.dropna()
    std_resid_gspc_sq_clean = std_resid_gspc_clean**2

    # Plotting Standardized Residuals
    plt.figure(figsize=(12, 6))
    plt.plot(std_resid_gspc_clean.index, std_resid_gspc_clean)
    plt.title('S&P 500 GARCH Standardized Residuals')
    plt.xlabel('Date'); plt.ylabel('Standardized Residual')
    plt.axhline(0, color='grey', linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURE_DIR, 'figure_A4_gspc_garch_std_resid.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # ACF/PACF Plots (Std Residuals)
    std_resid_gspc_np = std_resid_gspc_clean.to_numpy()
    std_resid_gspc_sq_np = std_resid_gspc_sq_clean.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    if np.var(std_resid_gspc_np) > 1e-10:
         plot_acf(std_resid_gspc_np, lags=40, ax=axes[0], title='ACF of Standardized Residuals (^GSPC)')
         plot_pacf(std_resid_gspc_np, lags=40, method='ywm', ax=axes[1], title='PACF of Standardized Residuals (^GSPC)')
         plt.tight_layout()
         plt.savefig(os.path.join(FIGURE_DIR, 'figure_A5_gspc_garch_std_resid_acf_pacf.png'), dpi=300, bbox_inches='tight')
         plt.show()
    plt.close(fig)

    # ACF/PACF Plots (Squared Std Residuals - ESSENTIAL VALIDATION)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    if np.var(std_resid_gspc_sq_np) > 1e-10:
        plot_acf(std_resid_gspc_sq_np, lags=40, ax=axes[0], title='ACF of Squared Standardized Residuals (^GSPC)')
        plot_pacf(std_resid_gspc_sq_np, lags=40, method='ywm', ax=axes[1], title='PACF of Squared Standardized Residuals (^GSPC)')
        plt.suptitle('Figure 10: Autocorrelation of Squared Standardized Residuals (S&P 500 GARCH)', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'figure_10_gspc_garch_sq_std_resid_acf_pacf.png'), dpi=300, bbox_inches='tight')
        plt.show()
    plt.close(fig)
else:
    print("Skipping ^GSPC GARCH validation due to model fit error.")

# GSPC Volatility Forecasting
forecast_horizon = 30
forecast_series_gspc = None
if garch_results_gspc and conditional_vol_gspc is not None:
    try:
        forecasts_gspc = garch_results_gspc.forecast(horizon=forecast_horizon, reindex=False)
        forecast_volatility_gspc = np.sqrt(forecasts_gspc.variance.iloc[-1])
        last_hist_date_gspc = conditional_vol_gspc.index[-1]
        forecast_start_date_gspc = last_hist_date_gspc + pd.tseries.offsets.BDay(1)
        forecast_dates_gspc = pd.date_range(start=forecast_start_date_gspc, periods=forecast_horizon, freq='B')
        if len(forecast_volatility_gspc) == len(forecast_dates_gspc):
            forecast_series_gspc = pd.Series(forecast_volatility_gspc.values, index=forecast_dates_gspc)
        else:
            forecast_series_gspc = pd.Series(forecast_volatility_gspc.values[:forecast_horizon], index=forecast_dates_gspc)

        last_hist_vol = conditional_vol_gspc.iloc[-1]
        last_hist_date = conditional_vol_gspc.index[-1]
        stitch_point = pd.Series([last_hist_vol], index=[last_hist_date])
        plot_forecast_series_gspc = pd.concat([stitch_point, forecast_series_gspc])

        # Plotting the forecast
        plt.figure(figsize=(14, 7))
        plt.plot(conditional_vol_gspc.iloc[-200:].index, conditional_vol_gspc.iloc[-200:], label='Fitted GARCH Volatility (^GSPC, Recent History)')
        plt.plot(plot_forecast_series_gspc.index, plot_forecast_series_gspc, label=f'Forecasted Volatility ({forecast_horizon} days)', color='red', marker='.', linestyle='--')
        plt.title('Figure 7: S&P 500 Conditional Volatility Forecast')
        plt.xlabel('Date'); plt.ylabel('Conditional Volatility (%)')
        plot_start_date_fc = conditional_vol_gspc.iloc[-200:].index[0]
        plot_end_date_fc = plot_forecast_series_gspc.index[-1]
        plt.xlim(plot_start_date_fc - pd.Timedelta(days=10), plot_end_date_fc + pd.Timedelta(days=10))
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURE_DIR, 'figure_7_gspc_vol_forecast.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Error during ^GSPC forecasting: {e}")
else:
    print("Skipping ^GSPC Volatility forecasting due to model fit error.")

# VIX EDA (Stationarity, ACF/PACF)
print("\n--- Stationarity Test (ADF) on VIX Level ---")
adf_test_vix = adfuller(prices_vix)
print(f'ADF Statistic: {adf_test_vix[0]:.4f}, p-value: {adf_test_vix[1]:.4f}')
d_vix = 0
vix_series_to_model = None
if adf_test_vix[1] <= 0.05:
    print("Conclusion: VIX level likely stationary (mean-reverting). Setting d=0.")
    vix_series_to_model = prices_vix
else:
    print("Warning: VIX level appears non-stationary. Check data or implement differencing. Skipping VIX analysis.")
    vix_series_to_model = None

use_garch_for_vix = False
if vix_series_to_model is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(vix_series_to_model, lags=40, ax=axes[0], title=f'ACF of {"VIX Level" if d_vix == 0 else "Differenced VIX"}')
    plot_pacf(vix_series_to_model, lags=40, method='ywm', ax=axes[1], title=f'PACF of {"VIX Level" if d_vix == 0 else "Differenced VIX"}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure_A6_vix_level_acf_pacf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print("\n--- ARCH Effect Test (Squared VIX Level) ---")
    series_for_arch_test = prices_vix if d_vix == 0 else prices_vix.diff().dropna()
    series_for_arch_test = series_for_arch_test.loc[vix_series_to_model.index]

    lb_test_sq = acorr_ljungbox(series_for_arch_test**2, lags=[10], return_df=True)
    use_garch_for_vix = bool(lb_test_sq['lb_pvalue'].iloc[0] <= 0.05) if not lb_test_sq.empty else False
    print(f"Ljung-Box test p-value on squared series (lag 10): {lb_test_sq['lb_pvalue'].iloc[0]:.4f}")
    print(f"Conclusion: {'GARCH model likely appropriate' if use_garch_for_vix else 'GARCH model may not be needed'} for VIX.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series_for_arch_test**2, lags=40, ax=axes[0], title=f'ACF of Squared {"VIX Level" if d_vix == 0 else "Differenced VIX"}')
    plot_pacf(series_for_arch_test**2, lags=40, method='ywm', ax=axes[1], title=f'PACF of Squared {"VIX Level" if d_vix == 0 else "Differenced VIX"}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure_A7_vix_sq_level_acf_pacf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# VIX Modeling (ARMA or ARMA+GARCH) & Validation
print("\n--- VIX Modeling ---")
p_vix, q_vix = 1, 1
arima_model_vix = None
garch_results_vix = None
conditional_vol_vix = None
std_resid_vix = None

if vix_series_to_model is not None:
    try:
        print(f"Fitting ARIMA({p_vix}, {d_vix}, {q_vix}) for VIX...")
        arima_model_vix = ARIMA(prices_vix, order=(p_vix, d_vix, q_vix)).fit()
        print("ARIMA fit successful.")
        arima_residuals_vix = arima_model_vix.resid.dropna()

        if use_garch_for_vix:
            print(f"\nFitting GARCH(1,1) to ARIMA residuals for VIX...")
            arima_residuals_vix = arima_residuals_vix.loc[prices_vix.index.intersection(arima_residuals_vix.index)]
            garch_model_spec_vix = arch_model(arima_residuals_vix, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
            garch_results_vix = garch_model_spec_vix.fit(update_freq=0, disp='off')
            print("\n--- GARCH Model Summary (VIX Residuals) ---")
            conditional_vol_vix = garch_results_vix.conditional_volatility
            std_resid_vix = garch_results_vix.std_resid

            print("\n--- GARCH Model Validation (VIX) ---")
            if std_resid_vix is not None:
                std_resid_vix_clean = std_resid_vix.dropna()
                std_resid_vix_sq_clean = std_resid_vix_clean**2
                std_resid_vix_np = std_resid_vix_clean.to_numpy()
                std_resid_vix_sq_np = std_resid_vix_sq_clean.to_numpy()
                print("Plotting ACF/PACF of Standardized Residuals (VIX GARCH)...")
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                if np.var(std_resid_vix_np) > 1e-10:
                     plot_acf(std_resid_vix_np, lags=40, ax=axes[0], title='ACF of Std Residuals (VIX GARCH)')
                     plot_pacf(std_resid_vix_np, lags=40, method='ywm', ax=axes[1], title='PACF of Std Residuals (VIX GARCH)')
                     plt.tight_layout()
                     plt.savefig(os.path.join(FIGURE_DIR, 'figure_A8_vix_garch_std_resid_acf_pacf.png'), dpi=300, bbox_inches='tight')
                     plt.show()
                plt.close(fig)
                print("Plotting ACF/PACF of Squared Standardized Residuals (VIX GARCH)...")
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                if np.var(std_resid_vix_sq_np) > 1e-10:
                    plot_acf(std_resid_vix_sq_np, lags=40, ax=axes[0], title='ACF of Sq. Std Residuals (VIX GARCH)')
                    plot_pacf(std_resid_vix_sq_np, lags=40, method='ywm', ax=axes[1], title='PACF of Sq. Std Residuals (VIX GARCH)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(FIGURE_DIR, 'figure_A9_vix_garch_sq_std_resid_acf_pacf.png'), dpi=300, bbox_inches='tight')
                    plt.show()
                plt.close(fig)
            else:
                 print("Skipping VIX GARCH validation plots.")
        else:
            print("\nUsing only ARIMA model for VIX (GARCH not indicated).")

    except Exception as e:
        print(f"Error during VIX modeling: {e}")
else:
    print("Skipping VIX modeling due to stationarity issues.")

# VIX Forecasting (Level and Conditional Volatility)
print("\n--- Forecasting Future VIX Level ---")
if arima_model_vix:
    try:
        forecast_obj_vix_level = arima_model_vix.get_forecast(steps=forecast_horizon)
        forecast_values = forecast_obj_vix_level.predicted_mean.values
        conf_int_values = forecast_obj_vix_level.conf_int(alpha=0.05).values
        last_hist_date_vix = prices_vix.index[-1]
        forecast_start_date_vix = last_hist_date_vix + pd.tseries.offsets.BDay(1)
        forecast_dates_vix = pd.date_range(start=forecast_start_date_vix, periods=forecast_horizon, freq='B')
        forecast_series_vix_level = pd.Series(forecast_values, index=forecast_dates_vix)
        forecast_ci_vix_level_df = pd.DataFrame(conf_int_values, index=forecast_dates_vix, columns=['lower', 'upper'])

        print(f"\nForecasted VIX Level for next {forecast_horizon} business days:")
        print(forecast_series_vix_level.head())

        # Plotting the VIX level forecast
        plt.figure(figsize=(14, 7))
        plt.plot(prices_vix.iloc[-200:].index, prices_vix.iloc[-200:], label='Historical VIX (Recent History)')
        plt.plot(forecast_series_vix_level.index, forecast_series_vix_level, label=f'Forecasted VIX ({forecast_horizon} days)', color='red', marker='.')
        plt.fill_between(forecast_ci_vix_level_df.index, forecast_ci_vix_level_df['lower'], forecast_ci_vix_level_df['upper'], color='pink', alpha=0.5, label='95% Confidence Interval')
        plt.title('VIX Index Forecast')
        plt.xlabel('Date'); plt.ylabel('VIX Level')
        plot_start_date_vix_fc = prices_vix.iloc[-200:].index[0]
        plot_end_date_vix_fc = forecast_series_vix_level.index[-1]
        plt.xlim(plot_start_date_vix_fc - pd.Timedelta(days=10), plot_end_date_vix_fc + pd.Timedelta(days=10))
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURE_DIR, 'figure_8_vix_level_forecast.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error during VIX Level forecasting: {e}")
else:
    print("Skipping VIX Level forecasting as ARIMA model failed or was not fitted.")

print("\n--- Forecasting Future VIX Conditional Volatility ---")
forecast_series_vix_vol = None
if garch_results_vix and conditional_vol_vix is not None:
    try:
        forecasts_vix_vol = garch_results_vix.forecast(horizon=forecast_horizon, reindex=False)
        forecast_volatility_vix = np.sqrt(forecasts_vix_vol.variance.iloc[-1])
        last_hist_date_vix_vol = conditional_vol_vix.index[-1]
        forecast_start_date_vix_vol = last_hist_date_vix_vol + pd.tseries.offsets.BDay(1)
        forecast_dates_vix_vol = pd.date_range(start=forecast_start_date_vix_vol, periods=forecast_horizon, freq='B')
        if len(forecast_volatility_vix) == len(forecast_dates_vix_vol):
            forecast_series_vix_vol = pd.Series(forecast_volatility_vix.values, index=forecast_dates_vix_vol)
        else:
            forecast_series_vix_vol = pd.Series(forecast_volatility_vix.values[:forecast_horizon], index=forecast_dates_vix_vol)

        print(f"\nForecasted VIX Conditional Volatility for next {forecast_horizon} business days:")
        print(forecast_series_vix_vol.head())

        last_hist_vol_vix = conditional_vol_vix.iloc[-1]
        last_hist_date_vix = conditional_vol_vix.index[-1]
        stitch_point_vix = pd.Series([last_hist_vol_vix], index=[last_hist_date_vix])
        plot_forecast_series_vix_vol = pd.concat([stitch_point_vix, forecast_series_vix_vol])

        # Plotting the VIX conditional volatility forecast
        plt.figure(figsize=(14, 7))
        plt.plot(conditional_vol_vix.iloc[-200:].index, conditional_vol_vix.iloc[-200:], label='Fitted VIX Cond. Volatility (Recent History)')
        plt.plot(plot_forecast_series_vix_vol.index, plot_forecast_series_vix_vol, label=f'Forecasted VIX Cond. Volatility ({forecast_horizon} days)', color='green', marker='.', linestyle='--')
        plt.title('VIX Conditional Volatility Forecast')
        plt.xlabel('Date'); plt.ylabel('Cond. Volatility of VIX Residuals')
        plot_start_date_vix_vol_fc = conditional_vol_vix.iloc[-200:].index[0]
        plot_end_date_vix_vol_fc = plot_forecast_series_vix_vol.index[-1]
        plt.xlim(plot_start_date_vix_vol_fc - pd.Timedelta(days=10), plot_end_date_vix_vol_fc + pd.Timedelta(days=10))
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURE_DIR, 'figure_9_vix_cond_vol_forecast.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error during VIX Conditional Volatility forecasting: {e}")
else:
    print("Skipping VIX Conditional Volatility forecasting as GARCH model was not fitted or failed.")

# Comparison: GSPC GARCH Volatility vs. VIX Index
print("\n--- Comparison: GSPC GARCH Volatility vs. VIX Index ---")
if conditional_vol_gspc is not None:
    try:
        aligned_garch_vol_gspc = conditional_vol_gspc.reindex(prices_vix.index).dropna()
        aligned_vix_for_comp = prices_vix.loc[aligned_garch_vol_gspc.index]
        annualized_garch_vol_gspc = aligned_garch_vol_gspc * np.sqrt(252)
        garch_vix_comp_df = pd.DataFrame({'garch_ann': annualized_garch_vol_gspc,'vix': aligned_vix_for_comp}).dropna()

        if not garch_vix_comp_df.empty:
            corr_garch_vix, pval_garch_vix = stats.pearsonr(garch_vix_comp_df['garch_ann'], garch_vix_comp_df['vix'])
            print(f"Correlation between Annualized GARCH Volatility (^GSPC) and VIX: {corr_garch_vix:.4f} (p-value: {pval_garch_vix:.4f})")

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(garch_vix_comp_df.index, garch_vix_comp_df['garch_ann'], label='Annualized GARCH(1,1) Volatility (^GSPC)', alpha=0.8)
            plt.plot(garch_vix_comp_df.index, garch_vix_comp_df['vix'], label='VIX Index (Implied Volatility)', alpha=0.8)
            plt.title('Figure 5: GARCH Estimated Volatility vs. VIX Implied Volatility')
            plt.xlabel('Date'); plt.ylabel('Annualized Volatility (%)')
            plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
            text_str_vix = f'Pearson R = {corr_garch_vix:.3f}\np-value = {pval_garch_vix:.3f}'
            plt.text(0.05, 0.95, text_str_vix, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
            plt.savefig(os.path.join(FIGURE_DIR, 'figure_5_gspc_garch_vs_vix.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
        else:
            print("Could not compare GARCH Volatility and VIX due to data issues after alignment/cleaning.")
    except Exception as e:
        print(f"Error during VIX comparison: {e}")
else:
    print("Skipping VIX comparison as ^GSPC Volatility was not calculated.")

# Correlation: GSPC GARCH Volatility vs. Trading Volume
print("\n--- Correlation: GSPC GARCH Volatility vs. Trading Volume ---")
if conditional_vol_gspc is not None:
    try:
        aligned_garch_vol_daily = conditional_vol_gspc.reindex(volume_gspc.index).dropna()
        aligned_volume_gspc = volume_gspc.loc[aligned_garch_vol_daily.index]
        aligned_log_volume_gspc = np.log(aligned_volume_gspc + 1)

        vol_corr_df = pd.DataFrame({
            'garch_vol': aligned_garch_vol_daily,
            'volume': aligned_volume_gspc,
            'log_volume': aligned_log_volume_gspc
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if not vol_corr_df.empty and len(vol_corr_df) > 1:
            corr_vol_raw, pval_vol_raw = stats.pearsonr(vol_corr_df['garch_vol'], vol_corr_df['volume'])
            print(f"Correlation between Daily GARCH Volatility (^GSPC) and Raw Volume: {corr_vol_raw:.4f} (p-value: {pval_vol_raw:.4f})")
            corr_vol_log, pval_vol_log = stats.pearsonr(vol_corr_df['garch_vol'], vol_corr_df['log_volume'])
            print(f"Correlation between Daily GARCH Volatility (^GSPC) and Log Volume: {corr_vol_log:.4f} (p-value: {pval_vol_log:.4f})")

            # Use jointplot hexbin for better visualization of density
            jp = sns.jointplot(data=vol_corr_df, x='garch_vol', y='log_volume', kind='hex',
                               height=7, gridsize=50, cmap='viridis', mincnt=1)
            jp.fig.suptitle('Figure 6: Daily GARCH Volatility vs. Log Trading Volume (^GSPC)', y=1.02)
            jp.set_axis_labels('Daily Conditional Volatility (%)', 'Log(Volume)')
            text_str_vol = f'Pearson R (Log Vol) = {corr_vol_log:.3f}\np-value = {pval_vol_log:.3f}'
            jp.ax_joint.text(0.05, 0.95, text_str_vol, transform=jp.ax_joint.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
            jp.fig.savefig(os.path.join(FIGURE_DIR, 'figure_6_gspc_vol_vs_logvolume_hexbin.png'), dpi=300, bbox_inches='tight')
            plt.show()

        else:
            print("Insufficient data points after cleaning to calculate volume correlations.")
    except Exception as e:
        print(f"Error during Volume correlation analysis: {e}")
else:
    print("Skipping Volume correlation as ^GSPC Volatility was not calculated.")
