from Deepalgo import ForecastingModels, RVEstimator, ReturnCalculator, DataPreprocessor
import pandas as pd
import sys

# Allow CSV path override via bash argument
csv_path = 'data/SPY.csv'
for i, arg in enumerate(sys.argv):
    if arg == "--data" and i + 1 < len(sys.argv):
        csv_path = sys.argv[i + 1]

# Load and process your data
df = DataPreprocessor.load_and_clean(csv_path, {'market_hours': {'start': '13:30', 'end': '20:00'}})
returns = ReturnCalculator.compute_log_returns(df)
rv_series = RVEstimator.naive_rv(returns)

# Prepare features and train model
features_df = ForecastingModels.prepare_har_features(rv_series)
model_results = ForecastingModels.fit_har_model(features_df, validation_split=0.2)

# Get the trained model
model = model_results['model']

# Make next-day forecast (same as Method 1)
latest_daily = rv_series.iloc[-1]
latest_weekly = rv_series.iloc[-5:].mean()
latest_monthly = rv_series.iloc[-22:].mean()

next_day_features = pd.DataFrame({
    'rv_daily_lag1': [latest_daily],
    'rv_weekly_lag1': [latest_weekly], 
    'rv_monthly_lag1': [latest_monthly]
})

forecast = model.predict(next_day_features)[0]
import math

def convert_rv(forecast, trading_days=252):
    """
    Converts daily variance into daily and annualized volatility (percentage terms).

    Args:
        rv_variance (float): The forecasted realized variance (e.g., 0.000106).
        trading_days (int): Number of trading days in a year (default: 252).

    Returns:
        dict: Dictionary with daily and annualized volatility in percent.
    """
    daily_vol = math.sqrt(forecast)
    annualized_vol = math.sqrt(forecast * trading_days)
    
    return {
        "daily_volatility_%": round(daily_vol * 100, 4),
        "annualized_volatility_%": round(annualized_vol * 100, 4)
    }

def app_log(msg):
    print(f"APPLOG: {msg}")
# Example
converted = convert_rv(forecast)
RUNNERResult = converted["annualized_volatility_%"] ##RR

print("Forecasted Realized Variance:", forecast)
print("Daily Volatility (%):", converted["daily_volatility_%"])
print("Annualized Volatility (%):", converted["annualized_volatility_%"])
app_log(f"Annualized Volatility (%): {converted['annualized_volatility_%']}")
