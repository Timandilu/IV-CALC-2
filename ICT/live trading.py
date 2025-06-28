
import os
import time
import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta

# === CONFIG ===
API_KEY = 'PK5DB3EBE7MXR5QZGXSD'
API_SECRET = 'eSjsTwn5EpQjQ9E5ttnhARy5x0uon51jfoLmPqh5'
USE_PAPER = True  # True = paper trading

ASSET = 'AAPL'
INTERVAL = TimeFrame.Minute
WINDOW = 20
RISK_PER_TRADE = 0.01
BASE_BALANCE = 100_000
MINUTES_LOOKBACK = 300

# === INIT ALPACA ===
trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# === STRATEGY STATE ===
balance = BASE_BALANCE
positions = []
last_bias = None

# === FETCH DATA ===
def fetch_bars(symbol, minutes=300):
    end = datetime.utcnow()
    start = end - timedelta(minutes=minutes)
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=INTERVAL,
        start=start,
        end=end,
    )
    bars = data_client.get_stock_bars(request_params).df
    bars = bars[bars['symbol'] == symbol].drop(columns='symbol')
    bars.index.name = 'datetime'
    return bars

# === STRATEGY LOGIC ===
def determine_bias(htf_df):
    if len(htf_df) < 3:
        return 'neutral'
    if htf_df['high'].iloc[-1] > htf_df['high'].iloc[-2] and htf_df['low'].iloc[-1] > htf_df['low'].iloc[-2]:
        return 'bullish'
    elif htf_df['high'].iloc[-1] < htf_df['high'].iloc[-2] and htf_df['low'].iloc[-1] < htf_df['low'].iloc[-2]:
        return 'bearish'
    else:
        return 'neutral'

def resample_to_4h(df):
    return df.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

def simulate_trade(df, bias):
    global balance

    if len(df) < WINDOW:
        return

    window_df = df[-WINDOW:]
    current = df.iloc[-1]
    previous = df.iloc[-2]
    price = current['close']
    direction, stop, tp = None, None, None

    if bias == 'bullish' and current['low'] < window_df['low'].min():
        direction = 'buy'
        stop = current['low']
        tp = price + 2 * (price - stop)
    elif bias == 'bearish' and current['high'] > window_df['high'].max():
        direction = 'sell'
        stop = current['high']
        tp = price - 2 * (stop - price)

    if direction:
        stop_distance = abs(price - stop)
        if stop_distance == 0:
            return
        risk = balance * RISK_PER_TRADE
        qty = round(risk / stop_distance, 2)

        place_market_order(direction, qty)
        print(f"{datetime.utcnow()} - {direction.upper()} {ASSET} | Entry: {price:.2f} | TP: {tp:.2f} | SL: {stop:.2f}")

def place_market_order(side, qty):
    order = MarketOrderRequest(
        symbol=ASSET,
        qty=qty,
        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    trading_client.submit_order(order)

# === MAIN LOOP ===
while True:
    try:
        df = fetch_bars(ASSET, minutes=MINUTES_LOOKBACK)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        htf = resample_to_4h(df)
        bias = determine_bias(htf)

        now = datetime.utcnow()
        if (bias in ['bullish', 'bearish']) and (7 <= now.hour < 10 or 12 <= now.hour < 15):
            simulate_trade(df, bias)

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(60)
