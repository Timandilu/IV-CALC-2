#!/usr/bin/env python3
"""
ICT Liquidity-Sweep + FVG Reversal Back-tester (enhanced)
=======================================================
• Minute-level back-test with 4-hour bias
• London / New-York kill-zones
• Sweep → BOS → FVG (+ optional OTE) entry
• Adaptive risk-fraction sizing, configurable R-multiple
• Rich trade log (exit-reason, holding-time, realised R)
• Diagnostics counters for every filter stage
• Equity curve, draw-down, PnL histogram, outcome pie
Usage
-----
$ python ICT/run2.py --csv SPY.csv --capital 1_000_000 --risk 0.01
"""
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

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
@dataclass
class Config:
    csv_path: Path
    base_capital: float = 1_000.0
    risk_per_trade: float = 0.01
    rr_mul: float = 2.0
    kill_zones: Tuple[Tuple[int, int], Tuple[int, int]] = ((7, 10), (12, 15))
    lookback: int = 20
    max_hold: int = 60
    fib_low: float = 0.618
    fib_high: float = 0.79
    fvg_bars: int = 3
    use_ote: bool = False               # toggle OTE filter quickly

REQUIRED_COLS = {"datetime", "open", "high", "low", "close", "volume"}

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    if not REQUIRED_COLS.issubset(df.columns.str.lower()):
        missing = REQUIRED_COLS - set(df.columns.str.lower())
        sys.exit(f"Missing columns: {missing}")
    df.columns = [c.lower() for c in df.columns]
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.empty:
        sys.exit("No data after cleaning.")
    return df

def in_kill_zone(ts: pd.Timestamp, zones) -> bool:
    return any(start <= ts.hour < end for start, end in zones)

# ────────────────────────────────────────────────────────────
# HTF Bias
# ────────────────────────────────────────────────────────────

def compute_htf_bias(df: pd.DataFrame) -> pd.Series:
    htf = df.resample("4H").agg({"high": "max", "low": "min"})
    bias = ["neutral", "neutral"]
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
    return htf["bias"].reindex(df.index, method="ffill")

# ────────────────────────────────────────────────────────────
# Pattern logic
# ────────────────────────────────────────────────────────────

def liquidity_sweep(df: pd.DataFrame, idx: int, lookback: int, bias: str) -> bool:
    window = df.iloc[idx - lookback : idx]
    return (
        df.low.iloc[idx] < window.low.min() if bias == "bullish" else df.high.iloc[idx] > window.high.max()
    )

def break_of_structure(df: pd.DataFrame, idx: int, bias: str) -> bool:
    if idx + 1 >= len(df):
        return False
    return (
        df.close.iloc[idx + 1] > df.high.iloc[idx] if bias == "bullish" else df.close.iloc[idx + 1] < df.low.iloc[idx]
    )

def detect_fvg(c1, c2, c3):
    if c1.high < c3.low:              # bullish gap
        return c1.high, c3.low
    if c1.low > c3.high:              # bearish gap
        return c3.high, c1.low
    return None

def fib_ote(start: float, end: float, entry: float, cfg: Config) -> bool:
    leg = end - start
    if leg == 0:
        return False
    retr = (entry - start) / leg
    return cfg.fib_low <= retr <= cfg.fib_high if leg > 0 else cfg.fib_low <= retr <= cfg.fib_high

# ────────────────────────────────────────────────────────────
# Trade container
# ────────────────────────────────────────────────────────────
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
    r_mult: Optional[float] = None
    outcome: Optional[str] = None   # win / loss / timeout
    hold: Optional[int] = None      # bars in trade

# ────────────────────────────────────────────────────────────
# Back-test core
# ────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, cfg: Config):
    equity = cfg.base_capital
    trades: List[Trade] = []

    # diagnostics counters
    diag = dict(kill=0, bias=0, sweep=0, bos=0, fvg=0, ote=0, taken=0)

    for i in range(cfg.lookback, len(df) - cfg.max_hold - cfg.fvg_bars):
        ts = df.index[i]
        #if not in_kill_zone(ts, cfg.kill_zones):
        #    continue
        diag["kill"] += 1

        bias = df.bias.iloc[i]
        if bias == "neutral":
            continue
        diag["bias"] += 1

        if not liquidity_sweep(df, i, cfg.lookback, bias):
            continue
        diag["sweep"] += 1

        if not break_of_structure(df, i, bias):
            continue
        diag["bos"] += 1

        fvg = detect_fvg(df.iloc[i - 2], df.iloc[i - 1], df.iloc[i])
        if not fvg:
            continue
        diag["fvg"] += 1
        fvg_mid = sum(fvg) / 2

        if cfg.use_ote and not fib_ote(df.close.iloc[i - 1], df.close.iloc[i], fvg_mid, cfg):
            continue
        diag["ote"] += 1

        # construct trade
        if bias == "bullish":
            entry, stop = fvg_mid, df.low.iloc[i]
            target = entry + cfg.rr_mul * (entry - stop)
            direction = "long"
        else:
            entry, stop = fvg_mid, df.high.iloc[i]
            target = entry - cfg.rr_mul * (stop - entry)
            direction = "short"

        risk_amt = equity * cfg.risk_per_trade
        size = risk_amt / abs(entry - stop)

        trade = Trade(direction, ts, entry, stop, target, size)
        diag["taken"] += 1

        # simulate forward
        for j in range(1, cfg.max_hold + 1):
            px = df.close.iloc[i + j]
            t_j = df.index[i + j]
            stop_hit = px <= stop if direction == "long" else px >= stop
            tgt_hit  = px >= target if direction == "long" else px <= target

            if stop_hit:
                trade.exit_time, trade.exit_price = t_j, stop
                trade.pnl = -risk_amt
                trade.outcome = "loss"
                trade.hold = j
                break
            if tgt_hit:
                trade.exit_time, trade.exit_price = t_j, target
                trade.pnl = risk_amt * cfg.rr_mul
                trade.outcome = "win"
                trade.hold = j
                break
        else:  # timeout
            trade.exit_time = df.index[i + cfg.max_hold]
            trade.exit_price = df.close.iloc[i + cfg.max_hold]
            trade.pnl = size * (trade.exit_price - entry) if direction == "long" else size * (entry - trade.exit_price)
            trade.outcome = "timeout"
            trade.hold = cfg.max_hold

        trade.r_mult = trade.pnl / risk_amt if risk_amt else 0
        trades.append(trade)
        equity += trade.pnl

    print("Diagnostics:", diag)
    return trades

# ────────────────────────────────────────────────────────────
# Reporting & visuals
# ────────────────────────────────────────────────────────────

def performance(trades: List[Trade], cfg: Config):
    if not trades:
        print("No trades executed.")
        return

    df_trades = pd.DataFrame([t.__dict__ for t in trades])
    pnl = df_trades["pnl"].values
    equity_curve = cfg.base_capital + np.cumsum(pnl)
    dd = equity_curve - np.maximum.accumulate(equity_curve)
    max_dd = dd.min()

    wins = (df_trades.outcome == "win").sum()
    losses = (df_trades.outcome == "loss").sum()
    timeouts = (df_trades.outcome == "timeout").sum()
    win_rate = wins / len(df_trades)

    sharpe = (pnl.mean() / pnl.std() * math.sqrt(252 * 6.5 * 60)) if pnl.std() > 0 else 0

    print("\n" + "─" * 60)
    print(f"Trades           : {len(trades)}")
    print(f"Wins             : {wins}")
    print(f"Losses           : {losses}")
    print(f"Timeouts         : {timeouts}")
    print(f"Win rate         : {win_rate:.2%}")
    print(f"Total PnL        : {pnl.sum():,.2f}")
    print(f"Sharpe (annual)  : {sharpe:.2f}")
    print(f"Max draw-down    : {max_dd:,.2f}")
    print("─" * 60)

    # Equity & drawdown plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(equity_curve, label="Equity")
    ax[0].set_ylabel("Equity")
    ax_dd = ax[0].twinx()
    ax_dd.fill_between(range(len(dd)), dd, color="red", alpha=0.3, label="Drawdown")
    ax[0].legend(loc="upper left")
    ax_dd.legend(loc="upper right")
    ax[1].bar(range(len(pnl)), pnl, color=["green" if x > 0 else "red" for x in pnl])
    ax[1].set_ylabel("Trade PnL")
    ax[1].set_xlabel("Trade #")
    plt.tight_layout()

    # Outcome pie
    plt.figure(figsize=(4, 4))
    plt.pie([wins, losses, timeouts], labels=["Wins", "Losses", "Timeouts"], autopct="%1.1f%%", startangle=90)
    plt.title("Trade Outcomes")
    plt.tight_layout()
    plt.show()

    # save CSV log
    df_trades.to_csv("trade_log.csv", index=False)
    print("Detailed trade log ➜ trade_log.csv")

# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def parse_cli() -> Config:
    p = argparse.ArgumentParser(description="ICT Liquidity Sweep + FVG Back-test")
    p.add_argument("--csv", required=True)
    p.add_argument("--capital", type=float, default=1_000)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--maxhold", type=int, default=60)
    p.add_argument("--ote", action="store_true", help="enable OTE filter")
    args = p.parse_args()
    return Config(Path(args.csv), args.capital, args.risk, args.rr, max_hold=args.maxhold, use_ote=args.ote)

# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────

def main():
    cfg = parse_cli()
    df = load_data(cfg.csv_path)
    df["bias"] = compute_htf_bias(df)
    trades = backtest(df, cfg)
    performance(trades, cfg)

if __name__ == "__main__":
    main()
