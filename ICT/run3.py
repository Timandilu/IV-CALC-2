#!/usr/bin/env python3
"""
ICT Liquidity Sweep + BOS + FVG Reversal Strategy Backtester

Usage example:
    python ict_backtest.py --csv SPY.csv --timezone America/New_York \
       --initial-capital 100000 --risk-per-trade 0.01 --plot

Author: Generated ICT Strategy Backtester
Version: 1.0
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ICTStrategy:
    """ICT Liquidity Sweep + BOS + FVG Strategy Implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data = None
        self.signals = None
        self.trades = []
        self.equity_curve = []
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate OHLCV data with robust parsing"""
        try:
            # Auto-detect delimiter
            with open(csv_path, 'r') as f:
                first_line = f.readline()
                if ';' in first_line:
                    delimiter = ';'
                elif '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','
            
            logger.info(f"Loading data from {csv_path} with delimiter '{delimiter}'")
            
            # Load data
            df = pd.read_csv(csv_path, delimiter=delimiter)
            
            # Validate columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            datetime_col = None
            
            # Find datetime column (case insensitive)
            for col in df.columns:
                if 'datetime' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                    datetime_col = col
                    break
            
            if datetime_col is None:
                raise ValueError("No datetime column found. Expected column containing 'datetime', 'time', or 'date'")
            
            # Rename datetime column for consistency
            if datetime_col != 'datetime':
                df = df.rename(columns={datetime_col: 'datetime'})
            
            # Check for required OHLCV columns
            missing_cols = []
            col_mapping = {}
            for req_col in ['open', 'high', 'low', 'close', 'volume']:
                found = False
                for col in df.columns:
                    if req_col in col.lower():
                        col_mapping[col] = req_col
                        found = True
                        break
                if not found:
                    missing_cols.append(req_col)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Rename columns
            df = df.rename(columns=col_mapping)
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Convert timezone
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize('UTC')
            
            df['datetime'] = df['datetime'].dt.tz_convert(self.config['timezone'])
            df = df.set_index('datetime').sort_index()
            
            # Data validation
            if df.isnull().any().any():
                logger.warning("Found NaN values in data, forward filling...")
                df = df.fillna(method='ffill')
            
            # Check for duplicates
            if df.index.duplicated().any():
                logger.warning("Found duplicate timestamps, removing...")
                df = df[~df.index.duplicated(keep='first')]
            
            logger.info(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_htf_bias(self) -> pd.Series:
        """Calculate higher timeframe bias using HH/HL or LH/LL patterns"""
        try:
            htf_data = self.data.resample(self.config['htf_frame']).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            bias = pd.Series(index=htf_data.index, dtype='object')
            
            for i in range(2, len(htf_data)):
                # Get last 3 candles
                highs = htf_data['high'].iloc[i-2:i+1]
                lows = htf_data['low'].iloc[i-2:i+1]
                
                # Check for HH/HL (bullish) or LH/LL (bearish)
                if (highs.iloc[1] > highs.iloc[0] and highs.iloc[2] > highs.iloc[1] and
                    lows.iloc[1] > lows.iloc[0]):
                    bias.iloc[i] = 'bullish'
                elif (highs.iloc[1] < highs.iloc[0] and highs.iloc[2] < highs.iloc[1] and
                      lows.iloc[1] < lows.iloc[0]):
                    bias.iloc[i] = 'bearish'
                else:
                    bias.iloc[i] = 'neutral'
            
            # Forward fill and align to 1-minute data
            bias = bias.ffill()
            bias_1m = bias.reindex(self.data.index, method='ffill')
            bias_1m = bias_1m.fillna('neutral')
            
            logger.info(f"HTF bias calculated: {bias_1m.value_counts().to_dict()}")
            return bias_1m
            
        except Exception as e:
            logger.error(f"Error calculating HTF bias: {str(e)}")
            raise
    
    def is_in_killzone(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading kill zones"""
        time_str = timestamp.strftime('%H:%M')
        
        for kz in self.config['killzones']:
            start_time, end_time = kz.split('-')
            if start_time <= time_str <= end_time:
                return True
        return False
    
    def detect_liquidity_sweep(self, idx: int, direction: str) -> bool:
        """Detect liquidity sweep based on direction and lookback window"""
        if idx < self.config['lookback']:
            return False
        
        current_bar = self.data.iloc[idx]
        lookback_window = self.data.iloc[idx-self.config['lookback']:idx]
        
        if direction == 'bullish':
            # Sweep of lows
            return current_bar['low'] < lookback_window['low'].min()
        else:  # bearish
            # Sweep of highs
            return current_bar['high'] > lookback_window['high'].max()
    
    def detect_bos(
            self,
            sweep_idx: int,
            direction: str,
            *,
            fractal_len: int = 2,
            pullback_pct: float = 0.0005,       # ≈ 0.08 % of price
            max_bars_after_swing: int = 15,
            require_close: bool = True
        ) -> Optional[Tuple[int, float]]:
        """
        Detect the first Break-of-Structure (BOS) after a liquidity sweep,
        without relying on ATR.

        Parameters
        ----------
        sweep_idx : int
            Index of the bar that performed the sweep.
        direction : {'bullish', 'bearish'}
            Bias direction *before* the sweep.  A bullish bias implies we expect
            a long setup *after* a sell-side sweep, and vice versa.
        fractal_len : int, optional
            Bars on each side for fractal pivot confirmation.
        pullback_pct : float, optional
            Minimum pull-back size expressed as a fraction of pivot price
            (0.0008 = 0.08 %).  Prevents micro-swings being treated as pivots.
        max_bars_after_swing : int, optional
            Abort search if no BOS occurs within this many bars after the swing.
        require_close : bool, optional
            If True a BOS needs the *close* beyond the pivot.  If False,
            a wick break is enough.

        Returns
        -------
        (bos_idx, pivot_price) or None
        """
        df = self.data
        pivot_idx = pivot_price = None

        # ---------- 1. Locate the first valid internal swing ----------
        look_ahead = range(
            sweep_idx + 1,
            min(sweep_idx + 30, len(df) - fractal_len)
        )

        for idx in look_ahead:
            window = df.iloc[idx - fractal_len: idx + fractal_len + 1]

            if direction == 'bullish':
                # need a swing-high pivot
                is_pivot = window['high'].iloc[fractal_len] == window['high'].max()
                # pull-back must retrace at least pullback_pct of pivot
                min_retrace = window['high'].iloc[fractal_len] * (1 - pullback_pct)
                retrace_ok = df['low'].iloc[idx + 1: idx + 4].min() < min_retrace
                if is_pivot and retrace_ok:
                    pivot_idx = idx
                    pivot_price = window['high'].iloc[fractal_len]
                    break
            else:  # bearish bias → swing-low
                is_pivot = window['low'].iloc[fractal_len] == window['low'].min()
                max_retrace = window['low'].iloc[fractal_len] * (1 + pullback_pct)
                retrace_ok = df['high'].iloc[idx + 1: idx + 4].max() > max_retrace
                if is_pivot and retrace_ok:
                    pivot_idx = idx
                    pivot_price = window['low'].iloc[fractal_len]
                    break

        if pivot_idx is None:
            return None  # no qualified swing found

        # ---------- 2. Search for BOS after the swing ----------
        search_end = min(pivot_idx + max_bars_after_swing, len(df))
        for idx in range(pivot_idx + 1, search_end):
            bar = df.iloc[idx]
            if direction == 'bullish':
                cond = bar['close'] > pivot_price if require_close else bar['high'] > pivot_price
            else:
                cond = bar['close'] < pivot_price if require_close else bar['low'] < pivot_price

            if cond:
                return idx, pivot_price

        return None  # no BOS within time window    
    
    def detect_fvg(self,bos_idx: int,direction: str,max_lookahead: int = 4,tolerance: float = 0,use_bodies: bool = True) -> Optional[Tuple[float, float]]:
        """
        Detect the first 3-bar Fair Value Gap (FVG) that occurs *after* the BOS candle.

        Parameters
        ----------
        bos_idx       : Index of the BOS bar in self.data.
        direction     : 'bullish' or 'bearish' (i.e. trade direction *with* HTF bias).
        max_lookahead : How many bars after BOS to search for a qualifying 3-bar sequence.
        tolerance     : Fractional overlap allowed between gap edges (e.g. 0.001 = 0.1 %).
        use_bodies    : If True compare candle bodies (open/close); else use full wicks (high/low).

        Returns
        -------
        (gap_low, gap_high) tuple if a gap is found, otherwise None.
        """
        price_cols = ('open', 'close') if use_bodies else ('low', 'high')

        # ensure we don’t step off the dataframe
        last_needed_row = bos_idx + max_lookahead + 2     # +2 for third candle
        if last_needed_row >= len(self.data):
            return None

        # walk forward through possible three-bar windows
        for offset in range(1, max_lookahead + 1):
            bars = self.data.iloc[bos_idx + offset : bos_idx + offset + 3]
            if len(bars) < 3:
                break

            bar1, _, bar3 = bars.iloc[0], bars.iloc[1], bars.iloc[2]

            if direction == 'bullish':
                high1 = bar1[price_cols[1]]  # high or close
                low3  = bar3[price_cols[0]]  # low  or open
                if high1 < low3 * (1 - tolerance):
                    return (high1, low3)     # (gap_low, gap_high) order for longs

            else:  # bearish
                low1  = bar1[price_cols[0]]  # low  or open
                high3 = bar3[price_cols[1]]  # high or close
                if low1 > high3 * (1 + tolerance):
                    return (high3, low1)     # (gap_high, gap_low) order for shorts

        return None
    
    def check_fibonacci_ote(self, entry: float, sweep_extreme: float, bos_pivot: float, direction: str) -> bool:
        """Check if entry is within Fibonacci OTE zone (61.8%-79%) of the impulse leg"""
        # Calculate impulse leg (sweep extreme to BOS pivot)
        impulse_range = abs(bos_pivot - sweep_extreme)
        
        if impulse_range == 0:
            return False
        
        if direction == 'bullish':
            # For bullish: BOS pivot is high, sweep extreme is low
            # OTE zone is 61.8% to 79% pullback from BOS pivot toward sweep extreme
            fib_618 = bos_pivot - 0.618 * impulse_range  # 61.8% pullback
            fib_79 = bos_pivot - 0.79 * impulse_range    # 79% pullback
            return fib_79 <= entry <= fib_618
        else:  # bearish
            # For bearish: BOS pivot is low, sweep extreme is high  
            # OTE zone is 61.8% to 79% pullback from BOS pivot toward sweep extreme
            fib_618 = bos_pivot + 0.618 * impulse_range  # 61.8% pullback
            fib_79 = bos_pivot + 0.79 * impulse_range    # 79% pullback
            return fib_618 <= entry <= fib_79
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals based on ICT strategy"""
        try:
            logger.info("Generating trading signals...")
            
            # Calculate HTF bias
            htf_bias = self.calculate_htf_bias()
            
            signals = []
            session_trades = 0
            current_date = None
            
            skipbias = 0
            skipliqsweep = 0
            skipsession = 0
            skipbos = 0
            skipfvg = 0
            skipfib = 0
            for i in tqdm(range(self.config['lookback'], len(self.data) - 10), desc="Processing signals"):
                timestamp = self.data.index[i]
                
                # Reset session trade counter daily
                if current_date != timestamp.date():
                    current_date = timestamp.date()
                    session_trades = 0
                
                # Skip if not in kill zone
                #if not self.is_in_killzone(timestamp):
                #    continue
                
                # Skip if bias is neutral
                bias = htf_bias.iloc[i]
                if bias == 'neutral':
                    continue
                skipbias += 1
                
                # Skip if max trades per session reached
                if session_trades >= self.config['max_trades_per_session']:
                    continue
                skipsession += 1
                # Detect liquidity sweep
                direction = 'bullish' if bias == 'bullish' else 'bearish'
                if not self.detect_liquidity_sweep(i, direction):
                    continue
                skipliqsweep +=1
                
                # Wait for BOS
                bos_result = self.detect_bos(i, direction)
                if bos_result is None:
                    continue
                skipbos += 1
                bos_idx, bos_pivot = bos_result
                
                # Detect FVG
                fvg = self.detect_fvg(bos_idx, direction)
                if fvg is None:
                    continue
                skipfvg += 1
                # Calculate entry (50% of FVG)
                entry_price = (fvg[0] + fvg[1]) / 2
                
                # Calculate stop loss
                sweep_extreme = (self.data.iloc[i]['low'] if direction == 'bullish' 
                               else self.data.iloc[i]['high'])
                spread = (self.data.iloc[i]['high'] - self.data.iloc[i]['low']) * 0.1
                stop_loss = sweep_extreme - spread if direction == 'bullish' else sweep_extreme + spread
                
                # Calculate take profit
                risk = abs(entry_price - stop_loss)
                
                # Add validation for risk calculation
                if risk <= 0:
                    logger.warning(f"Invalid risk calculation at {timestamp}: risk={risk}, entry={entry_price}, sl={stop_loss}")
                    continue
                
                reward_ratio = self.config['reward_risk_ratio']
                take_profit = (entry_price + risk * reward_ratio if direction == 'bullish'
                             else entry_price - risk * reward_ratio)
                
                # Optional Fibonacci OTE check
                if self.config['use_fibonacci_ote']:
                    if not self.check_fibonacci_ote(entry_price, sweep_extreme, bos_pivot, direction):
                        continue
                skipfib += 1
                # Use BOS timestamp as entry time (first possible entry point)
                entry_timestamp = self.data.index[bos_idx]
                
                signals.append({
                    'timestamp': entry_timestamp,  # ✅ Use BOS time, not sweep time
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'sweep_idx': i,
                    'bos_idx': bos_idx,
                    'bos_pivot': bos_pivot,
                    'fvg': fvg
                })
                
                session_trades += 1
            
            logger.info(f"Diagnositcs: Bias: {skipbias}, Session max: {skipsession}, Liquidity Sweep: {skipliqsweep}, BOS: {skipbos}, FVG: {skipfvg}, Fib OTE: {skipfib} ")
            signals_df = pd.DataFrame(signals)
            logger.info(f"Generated {len(signals_df)} trading signals")
            
            # Diagnostic check for look-ahead bias
            if len(signals_df) > 0:
                invalid_signals = 0
                for _, signal in signals_df.iterrows():
                    sweep_time = self.data.index[signal['sweep_idx']]
                    if signal['timestamp'] < sweep_time:
                        invalid_signals += 1
                
                if invalid_signals > 0:
                    logger.warning(f"{invalid_signals} signals still suffer from look-ahead timing")
                else:
                    logger.info(f"All signals pass look-ahead validation")
            
            self.signals = signals_df
            return signals_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def backtest(self) -> Dict:
        """Execute backtest with proper position sizing and risk management"""
        try:
            logger.info("Starting backtest execution...")
            
            if self.signals is None or len(self.signals) == 0:
                logger.warning("No signals to backtest")
                return {}
            
            capital = self.config['initial_capital']
            trades = []
            equity_curve = [capital]
            equity_dates = [self.data.index[0]]
            
            for _, signal in tqdm(self.signals.iterrows(), total=len(self.signals), desc="Backtesting"):
                entry_time = signal['timestamp']
                direction = signal['direction']
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                
                # Position sizing based on risk per trade
                risk_amount = capital * self.config['risk_per_trade']
                risk_per_share = abs(entry_price - stop_loss)
                
                if risk_per_share <= 0:
                    logger.warning(f"Zero risk per share at {entry_time}: entry={entry_price}, sl={stop_loss}")
                    continue
                    
                position_size = risk_amount / risk_per_share
                if position_size > capital * self.config['risk_per_trade'] * 2:
                    position_size = 1
                # Apply slippage and commission
                slippage_cost = entry_price * self.config['slippage'] / 10000
                commission_cost = position_size * entry_price * self.config['commission'] / 10000
                
                entry_price_adj = (entry_price + slippage_cost if direction == 'bullish' 
                                 else entry_price - slippage_cost)
                
                # Find exit point
                exit_price = None
                exit_time = None
                exit_reason = 'timeout'
                
                # Look for exit in subsequent bars
                entry_idx = self.data.index.get_loc(entry_time)
                
                # Convert max holding time to number of bars
                bar_interval_minutes = 1  # Assuming 1-minute bars
                max_hold_bars = self.config['max_holding_minutes'] // bar_interval_minutes
                
                for j in range(1, min(max_hold_bars + 1, len(self.data) - entry_idx)):
                    current_idx = entry_idx + j
                    current_bar = self.data.iloc[current_idx]
                    current_time = self.data.index[current_idx]
                    
                    if direction == 'bullish':
                        # Check for stop loss hit
                        if current_bar['low'] <= stop_loss:
                            exit_price = stop_loss
                            exit_time = current_time
                            exit_reason = 'stop_loss'
                            break
                        # Check for take profit hit
                        elif current_bar['high'] >= take_profit:
                            exit_price = take_profit
                            exit_time = current_time
                            exit_reason = 'take_profit'
                            break
                    else:  # bearish
                        # Check for stop loss hit
                        if current_bar['high'] >= stop_loss:
                            exit_price = stop_loss
                            exit_time = current_time
                            exit_reason = 'stop_loss'
                            break
                        # Check for take profit hit
                        elif current_bar['low'] <= take_profit:
                            exit_price = take_profit
                            exit_time = current_time
                            exit_reason = 'take_profit'
                            break
                
                # If no exit found, use last available price
                if exit_price is None:
                    max_idx = min(entry_idx + max_hold_bars, len(self.data) - 1)
                    exit_price = self.data.iloc[max_idx]['close']
                    exit_time = self.data.index[max_idx]
                
                # Apply exit slippage and commission
                exit_slippage = exit_price * self.config['slippage'] / 10000
                exit_price_adj = (exit_price - exit_slippage if direction == 'bullish' 
                                else exit_price + exit_slippage)
                
                # Calculate P&L
                if direction == 'bullish':
                    pnl = position_size * (exit_price_adj - entry_price_adj)
                else:
                    pnl = position_size * (entry_price_adj - exit_price_adj)
                
                pnl -= 2 * commission_cost  # Entry and exit commissions
                
                # Update capital
                capital += pnl
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry_price': entry_price_adj,
                    'exit_price': exit_price_adj,
                    'position_size': position_size,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price_adj)) * 100,
                    'exit_reason': exit_reason,
                    'capital': capital
                })
                
                equity_curve.append(capital)
                equity_dates.append(exit_time)
            
            self.trades = trades
            self.equity_curve = pd.Series(equity_curve, index=equity_dates)
            
            # Calculate performance metrics
            results = self.calculate_performance_metrics()
            logger.info("Backtest completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = (self.equity_curve.iloc[-1] - self.config['initial_capital']) / self.config['initial_capital']
        
        # Calculate drawdown
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate CAGR
        days_traded = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years_traded = days_traded / 365.25
        cagr = (self.equity_curve.iloc[-1] / self.config['initial_capital']) ** (1/years_traded) - 1 if years_traded > 0 else 0
        
        # Calculate Sharpe ratio (assuming daily returns)
        daily_returns = self.equity_curve.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 9999.0  # Cap infinity at large number
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'final_capital': self.equity_curve.iloc[-1],
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }


def create_visualizations(strategy: ICTStrategy, results: Dict, config: Dict):
    """Create and save visualization plots"""
    try:
        # Create figures directory
        Path('./figures').mkdir(exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Equity curve
        ax1.plot(strategy.equity_curve.index, strategy.equity_curve.values)
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = strategy.equity_curve.expanding().max()
        drawdown = (strategy.equity_curve - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Trade distribution
        if strategy.trades:
            trades_df = pd.DataFrame(strategy.trades)
            win_count = len(trades_df[trades_df['pnl'] > 0])
            loss_count = len(trades_df[trades_df['pnl'] < 0])
            neutral_count = len(trades_df[trades_df['pnl'] == 0])
            
            ax3.pie([win_count, loss_count, neutral_count], 
                   labels=['Wins', 'Losses', 'Neutral'],
                   colors=['green', 'red', 'gray'],
                   autopct='%1.1f%%')
            ax3.set_title('Trade Distribution')
        
        # Monthly returns heatmap (simplified)
        if len(strategy.equity_curve) > 30:
            monthly_returns = strategy.equity_curve.resample('M').last().pct_change().dropna()
            ax4.bar(range(len(monthly_returns)), monthly_returns.values * 100)
            ax4.set_title('Monthly Returns')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Return (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figures/backtest_results.png', dpi=300, bbox_inches='tight')
        
        if config['plot']:
            plt.show()
        else:
            plt.close()
            
        logger.info("Visualizations saved to ./figures/")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")


def save_trades_csv(trades: List[Dict]):
    """Save trades to CSV file"""
    try:
        if not trades:
            logger.warning("No trades to save")
            return
        
        trades_df = pd.DataFrame(trades)
        filename = f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"Trades saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving trades CSV: {str(e)}")


def print_results_table(results: Dict, config: Dict):
    """Print formatted results table to console"""
    print("\n" + "="*60)
    print("ICT STRATEGY BACKTEST RESULTS")
    print("="*60)
    print(f"{'Metric':<25} {'Value':<20} {'Format':<15}")
    print("-"*60)
    print(f"{'Initial Capital':<25} ${config['initial_capital']:,.2f}")
    print(f"{'Final Capital':<25} ${results.get('final_capital', 0):,.2f}")
    print(f"{'Total Return':<25} {results.get('total_return', 0)*100:.2f}%")
    print(f"{'CAGR':<25} {results.get('cagr', 0)*100:.2f}%")
    print(f"{'Max Drawdown':<25} {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"{'Sharpe Ratio':<25} {results.get('sharpe_ratio', 0):.2f}")
    print(f"{'Profit Factor':<25} {results.get('profit_factor', 0):.2f}")
    print("-"*60)
    print(f"{'Total Trades':<25} {results.get('total_trades', 0)}")
    print(f"{'Winning Trades':<25} {results.get('winning_trades', 0)}")
    print(f"{'Losing Trades':<25} {results.get('losing_trades', 0)}")
    print(f"{'Win Rate':<25} {results.get('win_rate', 0)*100:.2f}%")
    print(f"{'Gross Profit':<25} ${results.get('gross_profit', 0):,.2f}")
    print(f"{'Gross Loss':<25} ${results.get('gross_loss', 0):,.2f}")
    print("="*60)


def parse_killzones(kz_string: str) -> List[str]:
    """Parse kill zones string into list of time ranges"""
    return [kz.strip() for kz in kz_string.split(',')]


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ICT Strategy Backtester')
    parser.add_argument('--csv', required=True, help='Path to OHLCV CSV file')
    parser.add_argument('--timezone', default='America/New_York', help='Target timezone')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (fraction)')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission in basis points')
    parser.add_argument('--slippage', type=float, default=0.5, help='Slippage in basis points')
    parser.add_argument('--htf-frame', default='4h', help='Higher timeframe for bias')
    parser.add_argument('--lookback', type=int, default=20, help='Liquidity sweep lookback window')
    parser.add_argument('--max-holding', type=int, default=240, help='Max holding time in minutes')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--killzones', default='02:00-05:00,07:00-10:00', help='Trading kill zones')
    parser.add_argument('--out-sample-ratio', type=float, default=0.3, help='Out-of-sample ratio')
    
    args = parser.parse_args()
    
    try:
        # Configuration
        config = {
            'timezone': args.timezone,
            'initial_capital': args.initial_capital,
            'risk_per_trade': args.risk_per_trade,
            'commission': args.commission,
            'slippage': args.slippage,
            'htf_frame': args.htf_frame,
            'lookback': args.lookback,
            'max_holding_minutes': args.max_holding,
            'killzones': parse_killzones(args.killzones),
            'max_trades_per_session': 2,
            'reward_risk_ratio': 2.0,
            'use_fibonacci_ote': False,
            'plot': args.plot
        }
        
        logger.info("Starting ICT Strategy Backtest")
        logger.info(f"Configuration: {config}")
        
        # Initialize strategy
        strategy = ICTStrategy(config)
        
        # Load data
        strategy.load_data(args.csv)
        
        # Generate signals
        strategy.generate_signals()
        
        # Run backtest
        results = strategy.backtest()
        
        if results:
            # Print results
            print_results_table(results, config)
            
            # Create visualizations
            create_visualizations(strategy, results, config)
            
            # Save trades
            save_trades_csv(strategy.trades)
        else:
            logger.warning("No results generated from backtest")
        
        logger.info("Backtest completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
    parser.add_argument('--csv', required=True, help='Path to OHLCV CSV file')
    parser.add_argument('--timezone', default='America/New_York', help='Target timezone')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (fraction)')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission in basis points')
    parser.add_argument('--slippage', type=float, default=0.5, help='Slippage in basis points')
    parser.add_argument('--htf-frame', default='4h', help='Higher timeframe for bias')
    parser.add_argument('--lookback', type=int, default=20, help='Liquidity sweep lookback window')
    parser.add_argument('--max-holding', type=int, default=240, help='Max holding time in minutes')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--killzones', default='02:00-05:00,07:00-10:00', help='Trading kill zones')
    parser.add_argument('--out-sample-ratio', type=float, default=0.3, help='Out-of-sample ratio')

python ICT\\run3.py --csv SPY.csv --plot
"""