import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === SETTINGS ===
csv_path = 'data\\SPY.csv'  # Your 1-minute OHLCV CSV file
iv_threshold = 0.1855
trading_days = 252

# === LOAD DATA ===
df = pd.read_csv(csv_path, parse_dates=['datetime'])
df.sort_values('datetime', inplace=True)

# === EXTRACT DATE FROM TIMESTAMP ===
df['date'] = df['datetime'].dt.date

# === CALCULATE 1-MIN LOG RETURNS ===
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df.dropna(inplace=True)

# === CALCULATE DAILY REALIZED VARIANCE ===
# Sum of squared log returns per day (realized variance)
daily_var = df.groupby('date')['log_return'].agg(lambda x: np.sum(x**2))

# === CALCULATE DAILY REALIZED VOLATILITY ===
daily_vol = np.sqrt(daily_var)

# === ANNUALIZE VOLATILITY ===
daily_vol_annual = daily_vol * np.sqrt(trading_days)

# === COUNT DAYS OVER/UNDER THRESHOLD ===
over = (daily_vol_annual > iv_threshold).sum()
under = (daily_vol_annual <= iv_threshold).sum()

print(f"Days with annualized realized volatility > {iv_threshold*100:.2f}%: {over}")
print(f"Days with annualized realized volatility <= {iv_threshold*100:.2f}%: {under}")

# === PLOT ===
plt.figure(figsize=(12, 6))
plt.plot(daily_vol_annual.index, daily_vol_annual, label='Annualized Realized Volatility (1-min returns)', color='blue')
plt.axhline(iv_threshold, color='red', linestyle='--', label=f'Threshold ({iv_threshold*100:.2f}%)')

percentile_90 = np.percentile(daily_vol_annual, 90)
plt.axhline(percentile_90, color='orange', linestyle=':', label=f'90th Percentile ({percentile_90*100:.2f}%)')

plt.title('Daily Annualized Realized Volatility from 1-Minute Returns')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
