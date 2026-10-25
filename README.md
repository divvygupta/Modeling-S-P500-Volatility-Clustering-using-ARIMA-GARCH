# Modelling S&P 500 Volatility Clustering with ARIMA-GARCH

Do large moves in the S&P 500 cluster together, and how persistent is that volatility? This project models daily S&P 500 returns with an ARMA-GARCH framework and compares the estimated volatility with the VIX and trading volume.

## Data

- Daily S&P 500 (`^GSPC`) adjusted close and volume, and the VIX (`^VIX`), from Yahoo Finance via `yfinance`
- January 2004 to 2025, with 5,360 aligned daily observations
- Returns are daily log returns, in percent

## Method

1. **Exploratory analysis.** Price, return, squared-return, VIX and volume plots. ADF stationarity test, and ACF/PACF of returns and squared returns to detect ARCH effects (Ljung-Box).
2. **Mean model.** ARMA(1,1) on S&P 500 log returns to filter out linear dependence.
3. **Volatility model.** GARCH(1,1) on the ARMA residuals.
4. **Diagnostics.** ACF/PACF of standardized residuals and squared standardized residuals.
5. **Comparisons and forecasts.** Annualized GARCH volatility compared with the VIX and log trading volume (Pearson correlation). 30-day out-of-sample volatility and VIX forecasts.

## Key results

| Result | Value |
|---|---|
| GARCH(1,1) parameters (all p < 0.001) | ω = 0.0261, α = 0.1270, β = 0.8517 |
| Volatility persistence (α + β) | **0.979**: shocks to volatility decay slowly |
| Correlation of GARCH volatility with the VIX | **R = 0.89** (p < 0.001) |
| Correlation with log trading volume | R = 0.46 (p < 0.001) |

- S&P 500 returns are stationary (ADF p < 0.001), but squared returns show strong, persistent autocorrelation, which is clear evidence of volatility clustering.
- After fitting the GARCH model, squared standardized residuals show no remaining autocorrelation, so the model captures the clustering.
- The VIX itself shows near-integrated volatility (α + β ≈ 1), which suggests very long-lived shocks.

## Files

- `Code.py`: full analysis pipeline (data download, tests, models, forecasts, figures)
- `Project_Report.pdf`: written report with methodology, figures and discussion

## Tools

Python: pandas, NumPy, yfinance, statsmodels, arch, SciPy, Matplotlib, Seaborn

## How to run

```bash
pip install pandas numpy yfinance statsmodels arch scipy matplotlib seaborn
python Code.py
```

Figures are saved to `report_figures/`.
