#!/usr/bin/env python3
"""
Stock Data Updater - Incrementally fetch and append OHLCV data to CSV files
Usage: python stock_updater.py --ticker AAPL [--interval 1m]
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple
import yfinance as yf
import pandas as pd
import pytz


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update stock data CSV files')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--interval', default='1m', help='Data interval (default: 1m)')
    return parser.parse_args()


def get_csv_path(ticker: str) -> str:
    """Get the CSV file path for a given ticker."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f'{ticker}.csv')


def get_last_timestamp(csv_path: str) -> Optional[datetime]:
    """Get the last timestamp from existing CSV file."""
    if not os.path.exists(csv_path):
        return None
    
    try:
        # Read the last line of the CSV to get the most recent timestamp
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            
            last_row = None
            for row in reader:
                if row:  # Skip empty rows
                    last_row = row
            
            if last_row and len(last_row) > 0:
                # Parse the timestamp (assuming it's in the first column)
                timestamp_str = last_row[0]
                # Handle different timestamp formats
                try:
                    # Try parsing with timezone info first
                    if '+' in timestamp_str or timestamp_str.endswith('Z'):
                        dt = pd.to_datetime(timestamp_str, utc=True)
                    else:
                        # Assume UTC if no timezone info
                        dt = pd.to_datetime(timestamp_str)
                        if dt.tz is None:
                            dt = dt.tz_localize('UTC')
                    return dt.to_pydatetime()
                except:
                    return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def is_trading_hours(dt: datetime) -> bool:
    """Check if datetime falls within US trading hours (9:30-16:00 ET, weekdays only)."""
    # Convert to US/Eastern timezone
    et_tz = pytz.timezone('US/Eastern')
    if dt.tz is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    dt_et = dt.astimezone(et_tz)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if dt_et.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's within trading hours (9:30 AM to 4:00 PM ET)
    trading_start = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    trading_end = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return trading_start <= dt_et <= trading_end


def fetch_new_data(ticker: str, interval: str, start_time: Optional[datetime]) -> pd.DataFrame:
    """Fetch new data from yfinance starting from the given timestamp."""
    try:
        stock = yf.Ticker(ticker)
        
        # If no start time, fetch recent data (last 7 days for 1m interval)
        if start_time is None:
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(days=7)
        else:
            # Add a small buffer to ensure we don't miss any data
            start_time = start_time + timedelta(minutes=1)
            ##duration = timedelta(days=7)
            ##end_time = start_time + duration
            end_time = datetime.now(pytz.UTC)
        
        # Fetch data
        data = stock.history(
            start=start_time,
            end=end_time,
            interval=interval,
            prepost=False,  # Only regular trading hours
            auto_adjust=True,
            back_adjust=False
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Reset index to make Datetime a column
        data = data.reset_index()
        
        # Filter for trading hours only
        if not data.empty:
            data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
            trading_mask = data['Datetime'].apply(is_trading_hours)
            data = data[trading_mask]
        
        # Rename columns to match OHLCV format
        if not data.empty:
            data = data.rename(columns={
                'Datetime': 'Timestamp',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Select only the columns we need
            data = data[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def append_to_csv(csv_path: str, new_data: pd.DataFrame, last_timestamp: Optional[datetime]) -> int:
    """Append new data to CSV file, avoiding duplicates."""
    if new_data.empty:
        return 0
    
    # Filter out data that might already exist
    if last_timestamp:
        new_data = new_data[new_data['Timestamp'] > last_timestamp]
    
    if new_data.empty:
        return 0
    
    # Check if file exists and has header
    file_exists = os.path.exists(csv_path)
    write_header = not file_exists
    
    if file_exists:
        # Check if file is empty or has no header
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                write_header = True
    
    # Append to CSV
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if needed
            if write_header:
                writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Write data rows
            for _, row in new_data.iterrows():
                writer.writerow([
                    row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    f"{row['Open']:.4f}",
                    f"{row['High']:.4f}",
                    f"{row['Low']:.4f}",
                    f"{row['Close']:.4f}",
                    f"{row['Volume'] / 100000:.4f}"
                ])
        
        return len(new_data)
        
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return 0


def main():
    """Main function."""
    args = parse_arguments()
    ticker = args.ticker.upper()
    interval = args.interval
    
    print(f"Updating {ticker} data with {interval} interval...")
    
    # Get CSV path
    csv_path = get_csv_path(ticker)
    
    # Get last timestamp from existing file
    last_timestamp = get_last_timestamp(csv_path)
    
    if last_timestamp:
        print(f"Last data point: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        print("No existing data found, fetching recent data...")
    
    # Fetch new data
    new_data = fetch_new_data(ticker, interval, last_timestamp)
    
    if new_data.empty:
        print("No new data available.")
        return
    
    # Append new data to CSV
    rows_added = append_to_csv(csv_path, new_data, last_timestamp)
    def app_log(msg):
        print(f"APPLOG: {msg}")
    if rows_added > 0:
        latest_date = new_data['Timestamp'].max().strftime('%Y-%m-%d')
        print(f"Appended {rows_added} new rows for {ticker} on {latest_date}")
        print(f"Data saved to: {csv_path}")
        app_log(f"Stock data updated for ticker: {args.ticker} with with new lines: {rows_added} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        print("No new data to append (data might already be up to date).")
        app_log(f"No new data to append for ticker: {args.ticker} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import yfinance
        import pandas
        import pytz
    except ImportError as e:
        print(f"Required package not found: {e}")
        print("Please install required packages: pip install yfinance pandas pytz")
        sys.exit(1)
    
    main()
    
    """
    python Utils\\UTILstock_updater.py --ticker SPY --interval 1m
    python UTILstock_updater.py --ticker [] --interval [1m, 5m, 15m, 1h, 1d]
    """