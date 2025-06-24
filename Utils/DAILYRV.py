import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RealizedVolatilityCalculator:
    """
    A class to calculate various realized volatility estimators from OHLCV data.
    """
    
    def __init__(self, trading_start_hour=9, trading_start_minute=30, 
                 trading_end_hour=16, trading_end_minute=0):
        """
        Initialize the calculator with trading hours.
        
        Parameters:
        trading_start_hour (int): Start hour of trading (24-hour format)
        trading_start_minute (int): Start minute of trading
        trading_end_hour (int): End hour of trading (24-hour format)
        trading_end_minute (int): End minute of trading
        """
        self.trading_start = f"{trading_start_hour:02d}:{trading_start_minute:02d}"
        self.trading_end = f"{trading_end_hour:02d}:{trading_end_minute:02d}"
        self.annualization_factor = 252  # Trading days per year
        
    def load_and_filter_data(self, filepath):
        """
        Load CSV data and filter for trading hours.
        
        Parameters:
        filepath (str): Path to the CSV file
        
        Returns:
        pd.DataFrame: Filtered OHLCV data
        """
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_columns = ['datetime', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Loaded {len(df)} rows from {filepath}")
            
            # Parse datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Remove rows with missing or invalid OHLCV data
            initial_rows = len(df)
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Remove rows where high < low (invalid data)
            df = df[df['high'] >= df['low']]
            
            # Remove rows with non-positive prices
            price_cols = ['open', 'high', 'low', 'close']
            df = df[(df[price_cols] > 0).all(axis=1)]
            
            if len(df) < initial_rows:
                print(f"Removed {initial_rows - len(df)} rows with invalid/missing data")
            
            # Filter for trading hours
            df['time'] = df['datetime'].dt.strftime('%H:%M')
            df = df[(df['time'] >= self.trading_start) & (df['time'] <= self.trading_end)]
            
            print(f"After filtering for trading hours ({self.trading_start} - {self.trading_end}): {len(df)} rows")
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def parkinson_estimator(self, high, low):
        """
        Calculate Parkinson volatility estimator.
        RV = (1/(4*ln(2))) * ln(H/L)^2
        """
        return (1 / (4 * np.log(2))) * np.log(high / low) ** 2
    
    def garman_klass_estimator(self, open_price, high, low, close):
        """
        Calculate Garman-Klass volatility estimator.
        RV = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
        """
        return (np.log(high / close) * np.log(high / open_price) + 
                np.log(low / close) * np.log(low / open_price))
    
    def rogers_satchell_estimator(self, open_price, high, low, close):
        """
        Calculate Rogers-Satchell volatility estimator.
        RV = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
        """
        return (np.log(high / open_price) * np.log(high / close) + 
                np.log(low / open_price) * np.log(low / close))
    
    def close_to_close_estimator(self, close_prices):
        """
        Calculate Close-to-Close volatility estimator.
        RV = ln(C_t / C_{t-1})^2
        """
        log_returns = np.log(close_prices / close_prices.shift(1))
        return log_returns ** 2
    
    def calculate_daily_volatilities(self, df):
        """
        Calculate daily realized volatilities using all estimators.
        
        Parameters:
        df (pd.DataFrame): OHLCV data
        
        Returns:
        pd.DataFrame: Daily volatilities
        """
        # Add date column for grouping
        df['date'] = df['datetime'].dt.date
        
        # Group by date and calculate volatilities
        daily_results = []
        
        for date, group in df.groupby('date'):
            if len(group) < 2:  # Need at least 2 observations for close-to-close
                continue
                
            # Sort by datetime within the day
            group = group.sort_values('datetime')
            
            # Calculate intraday volatilities
            parkinson_vals = self.parkinson_estimator(group['high'], group['low'])
            gk_vals = self.garman_klass_estimator(group['open'], group['high'], 
                                                 group['low'], group['close'])
            rs_vals = self.rogers_satchell_estimator(group['open'], group['high'], 
                                                    group['low'], group['close'])
            ctc_vals = self.close_to_close_estimator(group['close'])
            
            # Sum intraday values to get daily realized volatility
            rv_parkinson = parkinson_vals.sum()
            rv_gk = gk_vals.sum()
            rv_rs = rs_vals.sum()
            rv_ctc = ctc_vals.sum()  # This will have NaN for first observation
            
            # Handle NaN in close-to-close
            rv_ctc = ctc_vals.dropna().sum() if not ctc_vals.dropna().empty else np.nan
            
            # Calculate average of all estimators (excluding NaN)
            estimators = [rv_parkinson, rv_gk, rv_rs, rv_ctc]
            valid_estimators = [x for x in estimators if not np.isnan(x)]
            rv_average = np.mean(valid_estimators) if valid_estimators else np.nan
            
            daily_results.append({
                'Date': date,
                'RV_Parkinson': rv_parkinson,
                'RV_GarmanKlass': rv_gk,
                'RV_RogersSatchell': rv_rs,
                'RV_CloseToClose': rv_ctc,
                'average': rv_average,
                'observations': len(group)  # Track number of observations per day
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(daily_results)
        
        # Annualize volatilities 
        # For high-frequency data, we already have the daily realized volatility
        # We just need to annualize by multiplying by sqrt(252) for variance scaling
        vol_columns = ['RV_Parkinson', 'RV_GarmanKlass', 'RV_RogersSatchell', 
                      'RV_CloseToClose', 'average']
        
        for col in vol_columns:
            # Convert to annualized volatility (standard deviation)
            results_df[col] = np.sqrt(results_df[col] * self.annualization_factor)
        
        return results_df
    
    def process_data(self, input_file='data.csv', output_file='realized_volatility.csv'):
        """
        Main processing function to calculate and save realized volatilities.
        
        Parameters:
        input_file (str): Input CSV file path
        output_file (str): Output CSV file path
        """
        try:
            print("=== Realized Volatility Calculator ===")
            print(f"Trading hours: {self.trading_start} - {self.trading_end}")
            print(f"Annualization factor: {self.annualization_factor} trading days")
            print()
            
            # Load and filter data
            df = self.load_and_filter_data(input_file)
            
            if df.empty:
                print("No valid data found after filtering!")
                return
            
            # Calculate daily volatilities
            print("Calculating realized volatilities...")
            results_df = self.calculate_daily_volatilities(df)
            
            # Display summary statistics
            print(f"\nCalculated volatilities for {len(results_df)} trading days")
            print(f"Average observations per day: {results_df['observations'].mean():.1f}")
            print("\nSummary Statistics (Annualized %):")
            print("=" * 50)
            
            vol_cols = ['RV_Parkinson', 'RV_GarmanKlass', 'RV_RogersSatchell', 
                       'RV_CloseToClose', 'average']
            
            for col in vol_cols:
                if col in results_df.columns:
                    mean_vol = results_df[col].mean() * 100  # Convert to percentage
                    std_vol = results_df[col].std() * 100
                    print(f"{col:<20}: Mean = {mean_vol:.2f}%, Std = {std_vol:.2f}%")
            
            # Save results (drop the observations column from output)
            output_df = results_df.drop('observations', axis=1)
            output_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Display first few rows
            print(f"\nFirst 5 rows of results:")
            display_df = output_df.head().copy()
            # Convert to percentage for display
            vol_cols = ['RV_Parkinson', 'RV_GarmanKlass', 'RV_RogersSatchell', 
                       'RV_CloseToClose', 'average']
            for col in vol_cols:
                if col in display_df.columns:
                    display_df[col] = (display_df[col] * 100).round(2)
            print(display_df.to_string(index=False))
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            raise

def plot_realized_volatility(csv_path: str):
    # Load the data
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # Ensure date is sorted
    df.sort_values('Date', inplace=True)
    
    # Plotting
    plt.figure(figsize=(14, 6))
    
    plt.plot(df['Date'], df['RV_Parkinson'], label='Parkinson', linewidth=1.2)
    plt.plot(df['Date'], df['RV_GarmanKlass'], label='Garman-Klass', linewidth=1.2)
    plt.plot(df['Date'], df['RV_RogersSatchell'], label='Rogers-Satchell', linewidth=1.2)
    plt.plot(df['Date'], df['RV_CloseToClose'], label='Close-to-Close', linewidth=1.2)
    plt.plot(df['Date'], df['average'], label='Average', color='black', linewidth=2.0, linestyle='--')
    
    plt.title('Daily Realized Volatility by Estimator')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Annualized)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show plot
    plt.show()

def main():
    """
    Main function to run the realized volatility calculation.
    You can modify the trading hours and file paths here.
    """
    
    # Initialize calculator with custom trading hours
    # Example: US market hours (9:30 AM - 4:00 PM EST)
    calculator = RealizedVolatilityCalculator(
        trading_start_hour=13, 
        trading_start_minute=30,
        trading_end_hour=20, 
        trading_end_minute=0
    )
    
    # Process the data
    calculator.process_data(
        input_file='data\SPY.csv',
        output_file='realized_volatility.csv'
    )
    
    Show_Plot = True ##Change
    if Show_Plot == True:
        plot_realized_volatility("realized_volatility.csv") 

if __name__ == "__main__":
    main()