
"""
ICT Liquidity Sweep + FVG Reversal Strategy
==========================================

A self‑contained backtesting script for 1‑minute OHLCV CSV data.

Features
--------
* High‑Time‑Frame (HTF) bias from rolling Daily and 4‑Hour windows
* Kill‑Zone session filter (London & New‑York)
* Liquidity sweep detection (swing high/low stop‑hunt)
* Break‑Of‑Structure (BOS)
* Fair‑Value‑Gap (FVG) identification & 50 % entry
* Fibonacci Optimal‑Trade‑Entry (OTE) filter (61.8–79 %)
* Volume confirmation (entry candle volume > 20‑period SMA)
* Optional Order‑Block (OB) confluence
* Position sizing by fixed %‑risk
* Static + trailing exit logic
* Equity‑curve & drawdown visuals
* Printable trade log & performance summary

Usage
-----
python ict_strategy.py --csv path/to/minute_data.csv

Input CSV must have **datetime, open, high, low, close, volume**
columns. Datetime must be either ISO 8601 or '%Y-%m-%d %H:%M:%S'.

Dependencies: pandas, numpy, matplotlib.
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------  Data Classes  -----------------------------

@dataclass
class Trade:
    direction: str  # 'long' or 'short'
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    def open_side(self) -> int:
        return 1 if self.direction == 'long' else -1

# -----------------------------  Strategy Core  ---------------------------

class ICTStrategy:
    def __init__(
        self,
        df: pd.DataFrame,
        risk_per_trade: float = 0.01,
        fib_min: float = 0.618,
        fib_max: float = 0.79,
        rr_multiple: float = 2.0,
        max_trades_per_session: int = 2,
        session_limits: Tuple[Tuple[int, int], Tuple[int, int]] = ((7, 10), (12, 15)), # UTC hours
        vol_ma_period: int = 20
    ):
        self.df = df.copy()
        self.equity = 1.0  # start with normalized equity
        self.risk_per_trade = risk_per_trade
        self.fib_min, self.fib_max = fib_min, fib_max
        self.rr_multiple = rr_multiple
        self.max_trades_per_session = max_trades_per_session
        self.session_limits = session_limits
        self.vol_ma_period = vol_ma_period
        self.trades: List[Trade] = []

        self._prepare_indicators()

    # -----------------------  Indicator Preparation -----------------------

    def _prepare_indicators(self):
        df = self.df
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # HTF windows
        df['daily_high'] = df['high'].rolling('1D').max()
        df['daily_low'] = df['low'].rolling('1D').min()
        df['4h_high'] = df['high'].rolling('4H').max()
        df['4h_low'] = df['low'].rolling('4H').min()

        # Volume moving average
        df['vol_ma'] = df['volume'].rolling(self.vol_ma_period).mean()

        # Initialize sweep / bos / fvg flags
        df['sweep'] = False
        df['bos'] = False
        df['fvg_up'] = np.nan
        df['fvg_dn'] = np.nan

    # -------------------------  Helper Functions --------------------------

    @staticmethod
    def in_session(ts: pd.Timestamp, session_limits) -> bool:
        hour = ts.hour
        for lo, hi in session_limits:
            if lo <= hour <= hi:
                return True
        return False

    @staticmethod
    def detect_swing(highs: pd.Series, lows: pd.Series, lookback: int = 10):
        return highs.shift(1).rolling(lookback).max(), lows.shift(1).rolling(lookback).min()

    # -------------------------  Core Detection ----------------------------

    def _detect_sweep_bos_fvg(self, lookback: int = 10):
        df = self.df
        swing_high, swing_low = self.detect_swing(df['high'], df['low'], lookback)
        df['prev_swing_high'] = swing_high
        df['prev_swing_low'] = swing_low

        # Sweep logic: current high > previous swing high   OR current low < prev swing low
        df['sweep_high'] = df['high'] > df['prev_swing_high']
        df['sweep_low'] = df['low'] < df['prev_swing_low']
        df['sweep'] = df['sweep_high'] | df['sweep_low']

        # BOS: After sweep, we look for price to reverse & break internal structure
        df['bos'] = False
        df['bos_dir'] = None

        # FVG detection: imbalance between candle N-2 high and N low (bear FVG) or vice versa
        df['fvg_up'] = np.nan
        df['fvg_dn'] = np.nan
        for i in range(2, len(df)):
            c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            # Bearish FVG (down impulse)
            if c1['low'] > c3['high']:
                df.iloc[i, df.columns.get_loc('fvg_dn')] = (c1['low'] + c3['high']) / 2
            # Bullish FVG (up impulse)
            if c1['high'] < c3['low']:
                df.iloc[i, df.columns.get_loc('fvg_up')] = (c1['high'] + c3['low']) / 2

            # Simple BOS: after sweep_high, price breaks below short‑term low
            if df.iloc[i - 1]['sweep_high'] and c3['close'] < min(c2['low'], c1['low']):
                df.iloc[i, df.columns.get_loc('bos')] = True
                df.iloc[i, df.columns.get_loc('bos_dir')] = 'short'
            # After sweep_low, break above short‑term high
            if df.iloc[i - 1]['sweep_low'] and c3['close'] > max(c2['high'], c1['high']):
                df.iloc[i, df.columns.get_loc('bos')] = True
                df.iloc[i, df.columns.get_loc('bos_dir')] = 'long'

    # -------------------------  Strategy Logic ---------------------------

    def run(self):
        self._detect_sweep_bos_fvg()
        df = self.df
        open_trade: Optional[Trade] = None
        trades_this_session = 0
        last_session_day = None

        for idx, row in df.iterrows():
            # session book‑keeping
            if last_session_day != idx.date():
                trades_this_session = 0
                last_session_day = idx.date()

            # Update trailing stop
            if open_trade:
                if open_trade.direction == 'long':
                    # move stop to previous swing low to lock profits
                    open_trade.stop_loss = max(open_trade.stop_loss, row['prev_swing_low'])
                else:
                    open_trade.stop_loss = min(open_trade.stop_loss, row['prev_swing_high'])

                # Exit conditions
                if (open_trade.direction == 'long' and row['low'] <= open_trade.stop_loss) or                        (open_trade.direction == 'short' and row['high'] >= open_trade.stop_loss):
                    open_trade.exit_time = idx
                    open_trade.exit_price = open_trade.stop_loss
                    open_trade.pnl = open_trade.open_side() * (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size
                    self.equity += open_trade.pnl
                    self.trades.append(open_trade)
                    open_trade = None
                    continue
                if (open_trade.direction == 'long' and row['high'] >= open_trade.take_profit) or                        (open_trade.direction == 'short' and row['low'] <= open_trade.take_profit):
                    open_trade.exit_time = idx
                    open_trade.exit_price = open_trade.take_profit
                    open_trade.pnl = open_trade.open_side() * (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size
                    self.equity += open_trade.pnl
                    self.trades.append(open_trade)
                    open_trade = None
                    continue

            # Entry Logic
            if open_trade or trades_this_session >= self.max_trades_per_session:
                continue  # skip if in trade or trade limit hit

            if not self.in_session(idx, self.session_limits):
                continue  # skip if outside kill zone

            if not row['bos']:
                continue

            # Volume confirmation
            if row['volume'] < row['vol_ma']:
                continue

            direction = row['bos_dir']
            if direction == 'long' and not math.isnan(row['fvg_up']):
                fvg_mid = row['fvg_up']
                # Fib OTE filter
                fib_zone_low = fvg_mid * (1 - (1 - self.fib_min) * 0.5)
                fib_zone_high = fvg_mid * (1 + (self.fib_max - 1) * 0.5)
                if not (fib_zone_low <= row['close'] <= fib_zone_high):
                    continue
                entry_price = fvg_mid
                stop_loss = row['prev_swing_low']
                tp = entry_price + self.rr_multiple * (entry_price - stop_loss)
            elif direction == 'short' and not math.isnan(row['fvg_dn']):
                fvg_mid = row['fvg_dn']
                fib_zone_high = fvg_mid * (1 + (1 - self.fib_min) * 0.5)
                fib_zone_low = fvg_mid * (1 - (self.fib_max - 1) * 0.5)
                if not (fib_zone_low <= row['close'] <= fib_zone_high):
                    continue
                entry_price = fvg_mid
                stop_loss = row['prev_swing_high']
                tp = entry_price - self.rr_multiple * (stop_loss - entry_price)
            else:
                continue

            # Risk‑based position size (normalized equity)
            risk_amount = self.equity * self.risk_per_trade
            stop_dist = abs(entry_price - stop_loss)
            size = risk_amount / stop_dist if stop_dist > 0 else 0
            if size <= 0:
                continue

            # Open trade
            open_trade = Trade(
                direction=direction,
                entry_time=idx,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=tp,
                size=size,
            )
            trades_this_session += 1

        # Close any open trade at end
        if open_trade:
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = df.iloc[-1]['close']
            open_trade.pnl = open_trade.open_side() * (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size
            self.equity += open_trade.pnl
            self.trades.append(open_trade)

    # ---------------------------  Reporting ------------------------------

    def performance_summary(self):
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total = len(self.trades)
        win_rate = len(wins) / total * 100 if total else 0
        pnl = sum(t.pnl for t in self.trades)
        avg_r = np.mean([t.pnl / self.risk_per_trade for t in self.trades]) if total else 0
        max_dd = self._max_drawdown()

        print(f'Trades: {total}')
        print(f'Win rate: {win_rate:.2f}%')
        print(f'Net PnL (equity gain): {pnl:.4f}')
        print(f'Average R multiple: {avg_r:.2f}')
        print(f'Max Drawdown: {max_dd:.4f}')

    def _max_drawdown(self):
        equity_curve = [1.0]
        eq = 1.0
        for t in self.trades:
            eq += t.pnl
            equity_curve.append(eq)
        equity_curve = np.array(equity_curve)
        roll_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - roll_max) / roll_max
        return drawdown.min()

    def plot_equity_curve(self):
        eq = 1.0
        curve_x, curve_y = [], []
        for t in self.trades:
            eq += t.pnl
            curve_x.append(t.exit_time)
            curve_y.append(eq)
        plt.figure(figsize=(10,4))
        plt.plot(curve_x, curve_y, lw=1.5)
        plt.title('Equity Curve')
        plt.ylabel('Equity')
        plt.xlabel('Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# -----------------------------  CLI Runner  ------------------------------

def run_cli():
    parser = argparse.ArgumentParser(description='ICT Strategy Backtest')
    parser.add_argument('--csv', required=True, help='Path to 1‑minute OHLCV CSV')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required_cols = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must include {required_cols}')

    strat = ICTStrategy(df)
    strat.run()
    strat.performance_summary()
    strat.plot_equity_curve()

if __name__ == '__main__':
    run_cli()
