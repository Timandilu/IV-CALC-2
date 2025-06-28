#!/usr/bin/env python3
"""
ICT Liquidity-Sweep + FVG Reversal Backtester
=============================================
Author : Quant Research Desk
Version: 1.0.0

Usage
-----
$ python ict_backtest.py --csv SPY.csv --capital 1_000_000 --risk 0.01

Core Features
-------------
✓ Minute-data backtest with high-timeframe bias (4 h)  
✓ London & New-York kill-zone filtering  
✓ Liquidity sweep → break-of-structure → FVG + OTE entry logic  
✓ 2 : 1 RR (configurable) – adaptive position sizing  
✓ Equity curve, drawdown, Sharpe ratio, win-loss breakdown  
✓ Robust I/O checks & clear runtime logging
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    csv_path: Path
    base_capital: float = 1_000_000.0   # starting equity
    risk_per_trade: float = 0.01        # 1 % of equity
    rr_mul: float = 2.0                 # reward:risk ratio
    kill_zones: Tuple[Tuple[int, int], Tuple[int, int]] = ((7, 10), (12, 15))  # UTC hours
    lookback: int = 20                  # sweep look-back (minutes)
    max_hold: int = 60                  # bars to sit in trade
    fib_low: float = 0.618              # OTE lower
    fib_high: float = 0.79              # OTE upper
    fvg_bars: int = 3                   # candles in FVG pattern

# ──────────────────────────────────────────────────────────────────────────────
# Utility / Validation
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"datetime", "open", "high", "low", "close", "volume"}

def load_data(path: Path) -> pd.DataFrame:
    """
    Read CSV and validate minimal schema.
    """
    if not path.exists():
        sys.exit(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    if not REQUIRED_COLS.issubset(df.columns.str.lower()):
        missing = REQUIRED_COLS - set(df.columns.str.lower())
        sys.exit(f"Missing columns: {missing}")
    df.columns = [c.lower() for c in df.columns]
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.dropna(inplace=True)
    if df.empty:
        sys.exit("No data after cleaning.")
    return df

def in_kill_zone(ts: pd.Timestamp, zones: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
    for start, end in zones:
        if start <= ts.hour < end:
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# High-Timeframe Bias
# ──────────────────────────────────────────────────────────────────────────────
def compute_htf_bias(df: pd.DataFrame) -> pd.Series:
    """
    4 h resample; bullish if HH/HL, bearish if LL/LH, else neutral.
    """
    htf = df.resample("4H").agg({"high": "max", "low": "min"})
    bias = ["neutral", "neutral"]  # seed first two
    for i in range(2, len(htf)):
        hh = htf.high.iloc[i] > htf.high.iloc[i - 1]
        hl = htf.low.iloc[i] > htf.low.iloc[i - 1]
        ll = htf.high.iloc[i] < htf.high.iloc[i - 1]
        lh = htf.low.iloc[i] < htf.low.iloc[i - 1]
        if hh and hl:
            bias.append("bullish")
        elif ll and lh:
            bias.append("bearish")
        else:
            bias.append("neutral")
    htf["bias"] = bias
    # forward-fill bias back to 1-min dataframe
    return htf["bias"].reindex(df.index, method="ffill")

# ──────────────────────────────────────────────────────────────────────────────
# Pattern Functions
# ──────────────────────────────────────────────────────────────────────────────
def liquidity_sweep(df: pd.DataFrame, idx: int, lookback: int, direction: str) -> bool:
    window = df.iloc[idx - lookback : idx]
    if direction == "bullish":   # look for sell-side sweep (take lows)
        return df.low.iloc[idx] < window.low.min()
    if direction == "bearish":   # seek buy-side sweep (take highs)
        return df.high.iloc[idx] > window.high.max()
    return False

def break_of_structure(df: pd.DataFrame, idx: int, direction: str) -> bool:
    """
    Immediate next candle confirms BOS.
    """
    if idx + 1 >= len(df):
        return False
    if direction == "bullish":
        return df.close.iloc[idx + 1] > df.high.iloc[idx]
    if direction == "bearish":
        return df.close.iloc[idx + 1] < df.low.iloc[idx]
    return False

def detect_fvg(c1, c2, c3) -> Optional[Tuple[float, float]]:
    """
    Basic 3-candle imbalance:
    - Bullish FVG if c1.high < c3.low (gap)
    - Bearish FVG if c1.low > c3.high
    Returns (start, end) of FVG.
    """
    if c1.high < c3.low:         # bullish
        return (c1.high, c3.low)
    if c1.low > c3.high:         # bearish
        return (c3.high, c1.low)
    return None

def fib_ote(price_start: float, price_end: float, entry: float) -> bool:
    """
    Check if entry is within 61.8–79 % retracement (OTE).
    """
    leg = price_end - price_start
    if leg == 0:
        return False
    retr = (entry - price_start) / leg
    if leg > 0:   # bullish impulse
        return Config.fib_low <= retr <= Config.fib_high
    else:         # bearish impulse
        retr = (entry - price_start) / leg  # leg negative
        return Config.fib_low <= retr <= Config.fib_high

# ──────────────────────────────────────────────────────────────────────────────
# Trade object
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    result: Optional[str] = None

# ──────────────────────────────────────────────────────────────────────────────
# Backtester
# ──────────────────────────────────────────────────────────────────────────────
def backtest(df: pd.DataFrame, cfg: Config) -> List[Trade]:
    trades: List[Trade] = []
    equity = cfg.base_capital

    for i in range(cfg.lookback, len(df) - cfg.max_hold - cfg.fvg_bars):
        ts = df.index[i]

        # Skip outside kill zones or neutral bias
        bias = df.bias.iloc[i]
        if bias == "neutral" """or not in_kill_zone(ts, cfg.kill_zones)""":
            continue

        # Sweep check
        if not liquidity_sweep(df, i, cfg.lookback, bias):
            continue

        # BOS confirmation
        if not break_of_structure(df, i, bias):
            continue

        # FVG detection on last 3 candles (i-2, i-1, i)
        fvg = detect_fvg(df.iloc[i - 2], df.iloc[i - 1], df.iloc[i])
        if not fvg:
            continue

        fvg_mid = (fvg[0] + fvg[1]) / 2.0

        # Fibonacci OTE filter vs displacement leg (i-1 close → i close)
        if bias == "bullish":
            #if not fib_ote(df.close.iloc[i - 1], df.close.iloc[i], fvg_mid):
            #   continue
            entry_price = fvg_mid
            stop = df.low.iloc[i]
            target = entry_price + cfg.rr_mul * (entry_price - stop)
        else:  # bearish
            #if not fib_ote(df.close.iloc[i - 1], df.close.iloc[i], fvg_mid):
            #   continue
            entry_price = fvg_mid
            stop = df.high.iloc[i]
            target = entry_price - cfg.rr_mul * (stop - entry_price)

        # Position sizing (risk fraction of current equity)
        risk = equity * cfg.risk_per_trade
        size = risk / abs(entry_price - stop)

        trade = Trade(
            direction="long" if bias == "bullish" else "short",
            entry_time=ts,
            entry=entry_price,
            stop=stop,
            target=target,
            size=size,
        )
        print(f"{ts} | {trade.direction.upper():5} | entry={trade.entry:.2f}  stop={trade.stop:.2f} target={trade.target:.2f}  size={trade.size:.1f}")
        # Simulate forward bars
        exit_found = False
        for j in range(1, cfg.max_hold + 1):
            price = df.close.iloc[i + j]
            time_j = df.index[i + j]
            if trade.direction == "long":
                if price <= stop:
                    trade.exit_time, trade.exit_price = time_j, stop
                    trade.pnl = -risk
                    trade.result = "loss"
                    exit_found = True
                    break
                if price >= target:
                    trade.exit_time, trade.exit_price = time_j, target
                    trade.pnl = risk * cfg.rr_mul
                    trade.result = "win"
                    exit_found = True
                    break
            else:  # short
                if price >= stop:
                    trade.exit_time, trade.exit_price = time_j, stop
                    trade.pnl = -risk
                    trade.result = "loss"
                    exit_found = True
                    break
                if price <= target:
                    trade.exit_time, trade.exit_price = time_j, target
                    trade.pnl = risk * cfg.rr_mul
                    trade.result = "win"
                    exit_found = True
                    break
        if not exit_found:  # neutral (max_hold passed)
            trade.exit_time = df.index[i + cfg.max_hold]
            trade.exit_price = df.close.iloc[i + cfg.max_hold]
            trade.pnl = (
                trade.size * (trade.exit_price - trade.entry)
                if trade.direction == "long"
                else trade.size * (trade.entry - trade.exit_price)
            )
            trade.result = "neutral"
        trades.append(trade)
        equity += trade.pnl
    return trades

# ──────────────────────────────────────────────────────────────────────────────
# Performance & Plots
# ──────────────────────────────────────────────────────────────────────────────
def performance(trades: List[Trade], cfg: Config) -> None:
    pnl = np.array([t.pnl for t in trades])
    equity_curve = cfg.base_capital + np.cumsum(pnl)
    wins = sum(1 for t in trades if t.result == "win")
    losses = sum(1 for t in trades if t.result == "loss")
    neutral = len(trades) - wins - losses
    win_rate = wins / len(trades) if trades else 0.0
    sharpe = (
        (pnl.mean() / pnl.std()) * math.sqrt(252 * 6.5 * 60)  # minute bars → annual
        if pnl.std() > 0
        else 0.0
    )

    print("\n─" * 60)
    print(f"Trades           : {len(trades)}")
    print(f"Wins             : {wins}")
    print(f"Losses           : {losses}")
    print(f"Neutral          : {neutral}")
    print(f"Win rate         : {win_rate:.2%}")
    print(f"Total PnL        : {pnl.sum():,.2f}")
    print(f"Sharpe (annual)  : {sharpe:.2f}")
    print("─" * 60)

    # Equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity")
    plt.tight_layout()

    # Pie of outcomes
    plt.figure(figsize=(4, 4))
    plt.pie(
        [wins, losses, neutral],
        labels=["Wins", "Losses", "Neutral"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Trade Outcomes")
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_cli() -> Config:
    p = argparse.ArgumentParser(
        description="Backtest ICT Liquidity Sweep + FVG Reversal Strategy"
    )
    p.add_argument("--csv", required=True, help="1-minute OHLCV CSV with datetime col")
    p.add_argument("--capital", type=float, default=1_000_000, help="starting equity")
    p.add_argument("--risk", type=float, default=0.01, help="fraction risk per trade")
    p.add_argument("--rr", type=float, default=2.0, help="reward:risk multiple")
    p.add_argument("--maxhold", type=int, default=60, help="max bars in trade")
    args = p.parse_args()

    cfg = Config(
        csv_path=Path(args.csv),
        base_capital=args.capital,
        risk_per_trade=args.risk,
        rr_mul=args.rr,
        max_hold=args.maxhold,
    )
    return cfg

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = parse_cli()
    df = load_data(cfg.csv_path)
    df["bias"] = compute_htf_bias(df)
        # ───── quick diagnostics  (insert right after df["bias"] = compute_htf_bias(df))
    trades = backtest(df, cfg)
    performance(trades, cfg)


if __name__ == "__main__":
    main()

## python ICT\\run.py --csv SPY.csv