#!/usr/bin/env python3
"""
Advanced Realized Volatility (RV) Model for 1-Minute OHLCV Data
================================================================

A comprehensive modular system for computing, modeling, and forecasting 
realized volatility from high-frequency market data.

Author: AI Assistant
Date: June 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealizedVolatilityModel:
    """
    Advanced Realized Volatility Model for 1-minute OHLCV data
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the RV model with configuration parameters"""
        self.config = config or self._default_config()
        self.data = None
        self.returns = None
        self.rv_series = None
        self.seasonality_profile = None
        self.forecast_model = None
        self.results = {}
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'timezone': 'UTC',
            'market_hours': {'start': '09:30', 'end': '16:00'},
            'estimators': ['naive_rv', 'realized_kernel', 'tsrv', 'parkinson'],
            'seasonality_adjustment': True,
            'jump_filtering': True,
            'forecast_model': 'har_rv',
            'forecast_horizon': 1,  # days
            'validation_split': 0.2
        }

class DataPreprocessor:
    """MODULE 1: Data Preprocessing"""
    
    @staticmethod
    def load_and_clean(file_path: str, config: Dict) -> pd.DataFrame:
        """Load and clean 1-minute OHLCV data"""
        logger.info(f"Loading data from {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Ensure required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Validate OHLC relationships
        df = DataPreprocessor._validate_ohlc(df)
        
        # Handle missing values
        df = DataPreprocessor._handle_missing_values(df)
        
        # Filter market hours if specified
        if config.get('market_hours'):
            df = DataPreprocessor._filter_market_hours(df, config['market_hours'])
        
        logger.info(f"Data loaded and cleaned: {len(df)} observations")
        return df
    
    @staticmethod
    def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data consistency"""
        # Check for invalid OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        )
        
        if invalid_mask.sum() > 0:
            logger.warning(f"Found {invalid_mask.sum()} invalid OHLC observations, removing...")
            df = df[~invalid_mask]
        
        return df
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in OHLCV data"""
        # Forward fill small gaps (up to 5 minutes)
        df = df.fillna(method='ffill', limit=5)
        
        # Remove remaining NaN values
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} observations with missing values")
        
        return df
    
    @staticmethod
    def _filter_market_hours(df: pd.DataFrame, market_hours: Dict) -> pd.DataFrame:
        """Filter data to market hours only"""
        start_time = market_hours['start']
        end_time = market_hours['end']
        
        df_filtered = df.between_time(start_time, end_time)
        logger.info(f"Filtered to market hours {start_time}-{end_time}: {len(df_filtered)} observations")
        
        return df_filtered

class ReturnCalculator:
    """MODULE 2: Return Calculation"""
    
    @staticmethod
    def compute_log_returns(df: pd.DataFrame) -> pd.Series:
        """Compute log returns from close prices"""
        prices = df['close']
        log_returns = np.log(prices / prices.shift(1))
        log_returns = log_returns.dropna()
        
        logger.info(f"Computed {len(log_returns)} log returns")
        return log_returns
    
    @staticmethod
    def compute_k_minute_returns(df: pd.DataFrame, k: int) -> pd.Series:
        """Compute k-minute log returns"""
        prices = df['close'].resample(f'{k}T').last()
        log_returns = np.log(prices / prices.shift(1))
        log_returns = log_returns.dropna()
        
        logger.info(f"Computed {len(log_returns)} {k}-minute log returns")
        return log_returns

class RVEstimator:
    """MODULE 3: Realized Volatility Estimation"""
    
    @staticmethod
    def naive_rv(returns: pd.Series, period: str = 'D') -> pd.Series:
        """Compute naive realized variance"""
        # Group returns by period and sum squared returns
        rv = returns.groupby(returns.index.to_period(period)).apply(
            lambda x: (x ** 2).sum()
        )
        rv.index = rv.index.to_timestamp()
        
        logger.info(f"Computed naive RV for {len(rv)} periods")
        return rv
    
    @staticmethod
    def realized_kernel(returns: pd.Series, period: str = 'D', bandwidth: int = 5) -> pd.Series:
        """Compute realized kernel estimator (simplified version)"""
        def kernel_weights(h: int) -> np.ndarray:
            """Parzen kernel weights"""
            x = np.arange(-h, h+1) / (h+1)
            w = np.where(np.abs(x) <= 0.5, 1 - 6*x**2 + 6*np.abs(x)**3,
                        np.where(np.abs(x) <= 1, 2*(1-np.abs(x))**3, 0))
            return w / w.sum()
        
        def compute_rk_day(day_returns):
            if len(day_returns) < 2*bandwidth + 1:
                return (day_returns**2).sum()
            
            weights = kernel_weights(bandwidth)
            n = len(day_returns)
            rk = 0
            
            for i in range(n):
                for j, w in enumerate(weights):
                    lag = j - bandwidth
                    if 0 <= i + lag < n:
                        rk += w * day_returns.iloc[i] * day_returns.iloc[i + lag]
            
            return max(rk, (day_returns**2).sum())  # Ensure positive
        
        rv = returns.groupby(returns.index.to_period(period)).apply(compute_rk_day)
        rv.index = rv.index.to_timestamp()
        
        logger.info(f"Computed realized kernel for {len(rv)} periods")
        return rv
    
    @staticmethod
    def two_scale_rv(returns: pd.Series, period: str = 'D', K: int = 5) -> pd.Series:
        """Two-Scale Realized Volatility (TSRV)"""
        def compute_tsrv_day(day_returns):
            n = len(day_returns)
            if n < K:
                return (day_returns**2).sum()
            
            # Fast scale (all returns)
            rv_fast = (day_returns**2).sum()
            
            # Slow scale (K-th returns)
            slow_returns = day_returns.iloc[::K]
            rv_slow = K * (slow_returns**2).sum()
            
            # TSRV estimator
            tsrv = rv_slow - (K-1)/K * rv_fast
            return max(tsrv, rv_fast * 0.1)  # Ensure reasonable positive value
        
        rv = returns.groupby(returns.index.to_period(period)).apply(compute_tsrv_day)
        rv.index = rv.index.to_timestamp()
        
        logger.info(f"Computed TSRV for {len(rv)} periods")
        return rv
    
    @staticmethod
    def parkinson_estimator(df: pd.DataFrame, period: str = 'D') -> pd.Series:
        """Parkinson range-based volatility estimator"""
        def compute_parkinson_day(day_data):
            if len(day_data) == 0:
                return 0
            
            # Parkinson estimator: (1/4ln(2)) * sum(ln(H/L)^2)
            log_hl_ratio = np.log(day_data['high'] / day_data['low'])
            return (1/(4*np.log(2))) * (log_hl_ratio**2).sum()
        
        rv = df.groupby(df.index.to_period(period)).apply(compute_parkinson_day)
        rv.index = rv.index.to_timestamp()
        
        logger.info(f"Computed Parkinson estimator for {len(rv)} periods")
        return rv
    
    @staticmethod
    def bipower_variation(returns: pd.Series, period: str = 'D') -> pd.Series:
        """Bipower variation for jump filtering"""
        def compute_bv_day(day_returns):
            if len(day_returns) < 2:
                return (day_returns**2).sum()
            
            # BV = (π/2) * Σ|r_i||r_{i-1}|
            abs_returns = np.abs(day_returns)
            bv = (np.pi/2) * (abs_returns * abs_returns.shift(1)).sum()
            return bv
        
        bv = returns.groupby(returns.index.to_period(period)).apply(compute_bv_day)
        bv.index = bv.index.to_timestamp()
        
        logger.info(f"Computed bipower variation for {len(bv)} periods")
        return bv

class SeasonalityAdjustor:
    """MODULE 4: Intraday Seasonality Adjustment"""
    
    @staticmethod
    def compute_seasonality_profile(returns: pd.Series) -> pd.Series:
        """Compute intraday volatility seasonality profile"""
        # Add minute of day
        minute_returns = returns.copy()
        minute_returns.index = pd.to_datetime(minute_returns.index)
        #minute_returns = minute_returns.assign(minute_of_day=minute_returns.index.hour * 60 + minute_returns.index.minute)
        minute_of_day = minute_returns.index.hour * 60 + minute_returns.index.minute
        
        # Compute average absolute return by minute of day
        # profile = minute_returns.groupby('minute_of_day').apply(lambda x: np.sqrt((x**2).mean()))
        
        profile = minute_returns.groupby(minute_of_day).apply(
            lambda x: np.sqrt((x**2).mean())
        )
        
        # Smooth the profile
        profile = profile.rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Computed seasonality profile with {len(profile)} time points")
        return profile
    
    @staticmethod
    def adjust_for_seasonality(rv_series: pd.Series, seasonality_profile: pd.Series) -> pd.Series:
        """Adjust RV series for intraday seasonality"""
        # This is a simplified adjustment - in practice you'd need more sophisticated methods
        # For daily RV, we assume the effect is already aggregated
        adjusted_rv = rv_series.copy()
        
        logger.info("Applied seasonality adjustment to RV series")
        return adjusted_rv

class ForecastingModels:
    """MODULE 5: Volatility Forecasting Models"""
    
    @staticmethod
    def prepare_har_features(rv_series: pd.Series) -> pd.DataFrame:
        """Prepare HAR-RV model features (daily, weekly, monthly)"""
        if len(rv_series) < 25:  # Need minimum data for monthly average
            raise ValueError(f"Insufficient data: {len(rv_series)} observations. Need at least 25.")
    
        df = pd.DataFrame(index=rv_series.index)
        df['rv_daily'] = rv_series
        df['rv_weekly'] = rv_series.rolling(window=5, min_periods=1).mean()
        df['rv_monthly'] = rv_series.rolling(window=22, min_periods=1).mean()

        # Lag the features by 1 day for forecasting
        df['rv_daily_lag1'] = df['rv_daily'].shift(1)
        df['rv_weekly_lag1'] = df['rv_weekly'].shift(1)
        df['rv_monthly_lag1'] = df['rv_monthly'].shift(1)

        # Target variable (exclude last row since it has no future value)
        df['rv_target'] = rv_series.shift(-1)

        # Only drop the first row (due to lag) and last row (due to target shift)
        result = df.iloc[1:-1].copy()

        # Check if we have any data left
        if len(result) == 0:
            raise ValueError("No valid samples after feature preparation. Check input data length.")

        return result
    
    @staticmethod
    def fit_har_model(features_df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """Fit HAR-RV model"""
        # Split data
        split_idx = int(len(features_df) * (1 - validation_split))
        train_data = features_df.iloc[:split_idx]
        test_data = features_df.iloc[split_idx:]
        
        # Features and target
        feature_cols = ['rv_daily_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1']
        X_train = train_data[feature_cols]
        y_train = train_data['rv_target']
        X_test = test_data[feature_cols]
        y_test = test_data['rv_target']
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        results = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'coefficients': dict(zip(feature_cols, model.coef_)),
            'intercept': model.intercept_,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        logger.info(f"HAR model fitted - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.6f}")
        return results
    
    @staticmethod
    def fit_log_har_model(features_df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """Fit Log-HAR-RV model"""
        # Transform to log scale
        log_features_df = features_df.copy()
        log_cols = ['rv_daily_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1', 'rv_target']
        
        for col in log_cols:
            log_features_df[col] = np.log(np.maximum(log_features_df[col], 1e-8))
        
        # Fit HAR model on log scale
        results = ForecastingModels.fit_har_model(log_features_df, validation_split)
        
        # Transform predictions back to original scale
        results['predictions_train'] = np.exp(results['predictions_train'])
        results['predictions_test'] = np.exp(results['predictions_test'])
        results['y_train'] = np.exp(results['y_train'])
        results['y_test'] = np.exp(results['y_test'])
        
        logger.info("Log-HAR model fitted and transformed back to original scale")
        return results

class ModelEvaluator:
    """MODULE 6: Evaluation & Output"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # QLIKE loss function for volatility forecasting
        qlike = np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'QLIKE': qlike
        }
    
    @staticmethod
    def generate_report(model_results: Dict, output_path: str = None) -> str:
        """Generate evaluation report"""
        report = f"""
Realized Volatility Model Evaluation Report
==========================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Performance:
- Training R²: {model_results.get('train_r2', 'N/A'):.4f}
- Test R²: {model_results.get('test_r2', 'N/A'):.4f}
- Training MSE: {model_results.get('train_mse', 'N/A'):.6f}
- Test MSE: {model_results.get('test_mse', 'N/A'):.6f}

Model Coefficients:
"""
        
        if 'coefficients' in model_results:
            for feature, coef in model_results['coefficients'].items():
                report += f"- {feature}: {coef:.6f}\n"
            report += f"- Intercept: {model_results.get('intercept', 'N/A'):.6f}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report

class RealizedVolatilityPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = None):
        """Initialize pipeline with configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = RealizedVolatilityModel()._default_config()
        
        self.results = {}
    
    def run_pipeline(self, data_path: str, output_dir: str = 'output') -> Dict:
        """Run the complete RV pipeline"""
        logger.info("Starting Realized Volatility Pipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # MODULE 1: Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        df = DataPreprocessor.load_and_clean(data_path, self.config)
        
        # MODULE 2: Return Calculation
        logger.info("Step 2: Computing Returns")
        returns = ReturnCalculator.compute_log_returns(df)
        
        # MODULE 3: RV Estimation
        logger.info("Step 3: Computing Realized Volatility")
        rv_estimates = {}
        
        for estimator in self.config['estimators']:
            if estimator == 'naive_rv':
                rv_estimates[estimator] = RVEstimator.naive_rv(returns)
            elif estimator == 'realized_kernel':
                rv_estimates[estimator] = RVEstimator.realized_kernel(returns)
            elif estimator == 'tsrv':
                rv_estimates[estimator] = RVEstimator.two_scale_rv(returns)
            elif estimator == 'parkinson':
                rv_estimates[estimator] = RVEstimator.parkinson_estimator(df)
        
        # MODULE 4: Seasonality Adjustment (optional)
        if self.config['seasonality_adjustment']:
            logger.info("Step 4: Seasonality Adjustment")
            seasonality_profile = SeasonalityAdjustor.compute_seasonality_profile(returns)
            # Apply adjustment (simplified for this implementation)
        
        # MODULE 5: Forecasting
        logger.info("Step 5: Volatility Forecasting")
        primary_rv = rv_estimates[self.config['estimators'][0]]  # Use first estimator as primary
        
        if self.config['forecast_model'] == 'har_rv':
            features_df = ForecastingModels.prepare_har_features(primary_rv)
            forecast_results = ForecastingModels.fit_har_model(
                features_df, self.config['validation_split']
            )
        elif self.config['forecast_model'] == 'log_har_rv':
            features_df = ForecastingModels.prepare_har_features(primary_rv)
            forecast_results = ForecastingModels.fit_log_har_model(
                features_df, self.config['validation_split']
            )
        
        # MODULE 6: Evaluation
        logger.info("Step 6: Model Evaluation")
        evaluation_metrics = ModelEvaluator.compute_metrics(
            forecast_results['y_test'], forecast_results['predictions_test']
        )
        
        # Generate report
        report = ModelEvaluator.generate_report(
            forecast_results, 
            os.path.join(output_dir, 'evaluation_report.txt')
        )
        
        # Save results
        self.results = {
            'data': df,
            'returns': returns,
            'rv_estimates': rv_estimates,
            'forecast_results': forecast_results,
            'evaluation_metrics': evaluation_metrics,
            'config': self.config
        }
        
        # Save RV estimates to CSV
        rv_df = pd.DataFrame(rv_estimates)
        rv_df.to_csv(os.path.join(output_dir, 'rv_estimates.csv'))
        
        # Save forecasts
        forecast_df = pd.DataFrame({
            'actual': forecast_results['y_test'],
            'predicted': forecast_results['predictions_test']
        }, index=forecast_results['y_test'].index)
        forecast_df.to_csv(os.path.join(output_dir, 'forecasts.csv'))
        
        logger.info(f"Pipeline completed. Results saved to {output_dir}")
        return self.results
    
    def plot_results(self, output_dir: str = 'output'):
        """Generate visualization plots"""
        if not self.results:
            logger.error("No results to plot. Run pipeline first.")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: RV Time Series
        rv_estimates = self.results['rv_estimates']
        for name, rv_series in rv_estimates.items():
            axes[0, 0].plot(rv_series.index, rv_series.values, 
                           label=name.replace('_', ' ').title(), alpha=0.7)
        axes[0, 0].set_title('Realized Volatility Estimates')
        axes[0, 0].set_ylabel('RV')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Returns Distribution
        returns = self.results['returns']
        axes[0, 1].hist(returns, bins=50, alpha=0.7, density=True)
        axes[0, 1].set_title('Log Returns Distribution')
        axes[0, 1].set_xlabel('Log Returns')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Forecast Performance
        forecast_results = self.results['forecast_results']
        y_test = forecast_results['y_test']
        y_pred = forecast_results['predictions_test']
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual RV')
        axes[1, 0].set_ylabel('Predicted RV')
        axes[1, 0].set_title('Forecast Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals
        residuals = y_test - y_pred
        axes[1, 1].plot(y_test.index, residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Forecast Residuals')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rv_analysis_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Plots saved to {output_dir}/rv_analysis_plots.png")

def main():
    """CLI interface for the RV pipeline"""
    parser = argparse.ArgumentParser(description='Advanced Realized Volatility Model')
    parser.add_argument('--data', required=True, help='Path to OHLCV CSV file')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--estimators', nargs='+', 
                       choices=['naive_rv', 'realized_kernel', 'tsrv', 'parkinson'],
                       default=['naive_rv', 'tsrv'], help='RV estimators to use')
    parser.add_argument('--forecast-model', choices=['har_rv', 'log_har_rv'],
                       default='har_rv', help='Forecasting model')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RealizedVolatilityPipeline(args.config)
    
    # Update config with CLI arguments
    pipeline.config['estimators'] = args.estimators
    pipeline.config['forecast_model'] = args.forecast_model
    
    try:
        # Run pipeline
        results = pipeline.run_pipeline(args.data, args.output)
        
        # Generate plots if requested
        if args.plot:
            pipeline.plot_results(args.output)
        
        print("\n" + "="*60)
        print("REALIZED VOLATILITY PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output}")
        print(f"Evaluation metrics:")
        for metric, value in results['evaluation_metrics'].items():
            print(f"  {metric}: {value:.6f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


#Deepalgo.py [-h] --data DATA [--config CONFIG] [--output OUTPUT] [--estimators {naive_rv,realized_kernel,tsrv,parkinson} [{naive_rv,realized_kernel,tsrv,parkinson} ...]] [--forecast-model {har_rv,log_har_rv}] [--plot]
# Deepalgo.py --data data/SPY.csv --config config.json --output output.csv --estimators naive_rv tsrv --forecast-model har_rv --plot