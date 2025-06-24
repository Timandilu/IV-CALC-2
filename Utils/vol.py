import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === SETTINGS ===
csv_path = 'data\\SPY.USUSD_Candlestick_1_D_BID_28.03.2017-14.06.2025.csv'  # Replace with your CSV file path
trading_days = 252

df = pd.read_csv(csv_path, parse_dates=['datetime'])


df['vol_daily'] = np.log(df['high'] / df['low'])
df['vol_annual'] = df['vol_daily'] * np.sqrt(trading_days)

df['price_change'] = df['close'].pct_change().abs()
df['VOL/PC'] = df['vol_annual'] / df['price_change']
"""# Normalize PC and PC/VOL
def scale_to_reference(series_to_scale, reference_series):
    s_mean, s_std = series_to_scale.mean(), series_to_scale.std()
    ref_mean, ref_std = reference_series.mean(), reference_series.std()
    zscore = (series_to_scale - s_mean) / s_std
    return zscore * ref_std + ref_mean

# Usage:
df['price_change'] = scale_to_reference(df['price_change'], df['vol_annual'])
df['PC/VOL'] = scale_to_reference(df['PC/VOL'], df['vol_annual'])"""

def calculate_score(vol_annual: pd.Series, price_change: pd.Series, alpha=1.0) -> pd.Series:
    # Take absolute value of price_change
    pc_abs = price_change.abs()
    
    # Normalize vol and price change to 0–1
    vol_norm = (vol_annual - vol_annual.min()) / (vol_annual.max() - vol_annual.min())
    pc_norm = (pc_abs - pc_abs.min()) / (pc_abs.max() - pc_abs.min())
    
    # Base: high vol & low pc
    base = vol_norm * (1 - pc_norm)
    
    # Amplify by max of normalized vol or price change
    amplify = 1 + alpha * np.maximum(vol_norm, pc_norm)
    
    score = base * amplify
    return score
# === Example usage ===
df['result'] = calculate_score(df['vol_annual'], df['price_change'])


### === PLOT ===
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['vol_annual'], label='Annualized Volatility (High-Low Range)')
plt.plot(df['datetime'], df['price_change'], label='Price Change', alpha = 0.2, color='orange')
plt.plot(df['datetime'], df['result'], label='Price Change / Volatility', alpha = 0.5, color='green')
plt.title('Daily Annualized Volatility from OHLC High-Low Range')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.text(f"Annualized Volatility Mean: {df['vol_annual'].mean() * 100:.2f}%")

print()