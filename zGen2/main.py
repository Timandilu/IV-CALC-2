#!/usr/bin/env python3
# venv\Scripts\python.exe main.py 
"""
Expert-Level Realized Volatility Prediction System
Implements HAR-RV, Tree-based, LSTM, and Ensemble models for 1-day ahead RV forecasting
"""

import os
import sys
import argparse
import logging
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class RVLogger:
    """Centralized logging system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create unique run ID
        self.run_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        
        # Setup logging
        log_file = self.log_dir / f"rv_prediction_{self.run_id}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)

class DataValidator:
    """Validates input data quality and structure"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV data structure and quality"""
        errors = []
        
        # Check required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        if errors:
            return False, errors
            
        # Check data types
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except:
            errors.append("Invalid timestamp format")
            
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    errors.append(f"Column {col} contains non-numeric data")
                    
        # Check for missing values
        if df[numeric_cols].isnull().any().any():
            errors.append("Missing values detected in numeric columns")
            
        # Check OHLC consistency
        ohlc_errors = (df['high'] < df[['open', 'close']].max(axis=1)) | \
                     (df['low'] > df[['open', 'close']].min(axis=1))
        if ohlc_errors.any():
            errors.append(f"OHLC inconsistencies found in {ohlc_errors.sum()} rows")
            
        # Check time ordering
        #if not df['timestamp'].is_monotonic_increasing:
         #   errors.append("Timestamps are not in ascending order")
            
        # Check sampling frequency
        time_diffs = df['datetime'].diff().dropna()
        mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else pd.Timedelta('1min')
        if mode_diff != pd.Timedelta('1min'):
            errors.append(f"Sampling frequency is {mode_diff}, expected 1 minute")
            
        return len(errors) == 0, errors

class RVCalculator:
    """Calculates realized volatility and related measures"""
    
    @staticmethod
    def calculate_rv(df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """Calculate daily realized volatility from intraday returns"""
        # Calculate log returns
        df = df.copy()
        df['log_return'] = np.log(df[price_col]) - np.log(df[price_col].shift(1))
        df['log_return'] = df['log_return'].fillna(0)
        
        # Square returns
        df['squared_return'] = df['log_return'] ** 2
        
        # Group by date and sum
        df['date'] = df['datetime'].dt.date
        daily_rv = df.groupby('date')['squared_return'].sum()
        
        return daily_rv
    
    @staticmethod
    def calculate_realized_measures(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive realized measures"""
        df = df.copy()
        df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
        df['log_return'] = df['log_return'].fillna(0)
        df['date'] = df['datetime'].dt.date
        
        # Group by date and calculate measures
        daily_measures = df.groupby('date').agg({
            'log_return': [
                lambda x: np.sum(x**2),  # RV
                lambda x: np.sum(x**3) / (np.sum(x**2)**(3/2)) if np.sum(x**2) > 0 else 0,  # Skewness
                lambda x: np.sum(x**4) / (np.sum(x**2)**2) if np.sum(x**2) > 0 else 0,  # Kurtosis
                'count'  # Number of observations
            ],
            'volume': 'sum',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).round(8)
        
        # Flatten column names
        daily_measures.columns = ['RV', 'RSkew', 'RKurt', 'n_obs', 'volume', 'high', 'low', 'close']
        
        # Calculate jump component (simplified)
        daily_measures['RJ'] = np.maximum(0, daily_measures['RV'] - 
                                         daily_measures['RV'].rolling(5).median())
        
        return daily_measures

class FeatureEngineer:
    """Creates features for volatility prediction"""
    
    @staticmethod
    def create_har_features(rv_series: pd.Series, lags: List[int] = [1, 5, 22]) -> pd.DataFrame:
        """Create HAR-style lagged features"""
        features = pd.DataFrame(index=rv_series.index)
        
        for lag in lags:
            if lag == 1:
                features[f'RV_lag_{lag}'] = rv_series.shift(lag)
            else:
                # Weekly/monthly averages
                features[f'RV_lag_{lag}'] = rv_series.rolling(lag).mean().shift(1)
                
        return features
    
    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from OHLCV data"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['ATR'] = (df['high'] - df['low']).rolling(14).mean()
        features['RSI'] = FeatureEngineer._calculate_rsi(df['close'])
        features['MACD'] = FeatureEngineer._calculate_macd(df['close'])
        
        # Volume features
        features['Volume_MA'] = df['volume'].rolling(20).mean()
        features['Volume_Ratio'] = df['volume'] / features['Volume_MA']
        
        # Volatility features
        features['Close_Volatility'] = df['close'].pct_change().rolling(20).std()
        features['High_Low_Ratio'] = df['high'] / df['low']
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2

class HARModel:
    """HAR-RV baseline model implementation"""
    
    def __init__(self):
        self.coefficients = None
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit HAR model using OLS"""
        from sklearn.linear_model import LinearRegression
        
        # Remove NaN values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data points for HAR model fitting")
            
        # Fit linear regression
        self.model = LinearRegression()
        self.model.fit(X_clean, y_clean)
        self.fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using fitted HAR model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        return self.model.predict(X)

class LSTMModel:
    """LSTM model for volatility prediction"""
    
    def __init__(self, sequence_length: int = 30, features: int = 1):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Assume first column is target
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM architecture"""
        self.model = Sequential([
            LSTM(50, return_sequences=True, 
                 input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae'])
    
    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int = 100, validation_split: float = 0.2):
        """Train LSTM model"""
        # Prepare data
        data = pd.concat([y, X], axis=1)
        data = data.dropna()
        
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Insufficient data for sequence length {self.sequence_length}")
            
        X_seq, y_seq = self._prepare_sequences(data)
        
        if len(X_seq) == 0:
            raise ValueError("No sequences could be created from the data")
            
        # Build model if not exists
        if self.model is None:
            self.features = X_seq.shape[2]
            self.build_model()
            
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.fitted = True
        return history
    
    def predict(self, X: pd.DataFrame, y_last: pd.Series) -> np.ndarray:
        """Make predictions using fitted LSTM model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Prepare last sequence
        data = pd.concat([y_last, X], axis=1)
        data = data.dropna()
        
        if len(data) < self.sequence_length:
            # Pad with mean values if insufficient data
            mean_vals = data.mean()
            padding_rows = self.sequence_length - len(data)
            padding_df = pd.DataFrame([mean_vals] * padding_rows, 
                                    columns=data.columns,
                                    index=pd.date_range(start=data.index[0] - pd.Timedelta(days=padding_rows),
                                                       periods=padding_rows, freq='D'))
            data = pd.concat([padding_df, data])
            
        # Scale and predict
        scaled_data = self.scaler.transform(data)
        X_seq = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        prediction_scaled = self.model.predict(X_seq, verbose=0)
        
        # Inverse transform
        dummy_data = np.zeros((1, scaled_data.shape[1]))
        dummy_data[0, 0] = prediction_scaled[0, 0]
        prediction = self.scaler.inverse_transform(dummy_data)[0, 0]
        
        return np.array([prediction])

class EnsembleModel:
    """Ensemble model combining multiple approaches"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.fitted = False
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all models in the ensemble"""
        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    model.fit(X, y)
                else:
                    model.fit(X, y)
            except Exception as e:
                print(f"Warning: Failed to fit {name} model: {e}")
                
        self.fitted = True
        
    def predict(self, X: pd.DataFrame, y_last: Optional[pd.Series] = None) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                if name == 'lstm' and y_last is not None:
                    pred = model.predict(X, y_last)
                else:
                    pred = model.predict(X)
                    
                if len(pred) > 0:
                    predictions.append(pred * self.weights[name])
                    total_weight += self.weights[name]
            except Exception as e:
                print(f"Warning: Failed to predict with {name} model: {e}")
                
        if not predictions:
            raise ValueError("No models produced valid predictions")
            
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        return ensemble_pred

class StatisticalTests:
    """Statistical tests for forecast evaluation"""
    
    @staticmethod
    def diebold_mariano_test(forecast1: np.array, forecast2: np.array, 
                            actual: np.array) -> Tuple[float, float]:
        """Diebold-Mariano test for forecast accuracy comparison"""
        e1 = forecast1 - actual
        e2 = forecast2 - actual
        
        d = e1**2 - e2**2
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        if d_var == 0:
            return 0, 1
            
        dm_stat = d_mean / np.sqrt(d_var / len(d))
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return dm_stat, p_value
    
    @staticmethod
    def clark_west_test(restricted_forecast: np.array, unrestricted_forecast: np.array,
                       actual: np.array) -> Tuple[float, float]:
        """Clark-West test for nested model comparison"""
        e_r = restricted_forecast - actual
        e_u = unrestricted_forecast - actual
        
        f_diff = restricted_forecast - unrestricted_forecast
        d = e_r**2 - e_u**2 + f_diff**2
        
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        if d_var == 0:
            return 0, 1
            
        cw_stat = d_mean / np.sqrt(d_var / len(d))
        p_value = 1 - stats.norm.cdf(cw_stat)
        
        return cw_stat, p_value

class RVPredictor:
    """Main class orchestrating the prediction system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = RVLogger()
        self.data_validator = DataValidator()
        self.rv_calculator = RVCalculator()
        self.feature_engineer = FeatureEngineer()
        
        # Models
        self.har_model = HARModel()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.lstm_model = LSTMModel()
        self.ensemble_model = EnsembleModel()
        
        # Data storage
        self.data = None
        self.features = None
        self.target = None
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'har_lags': [1, 5, 22],
            'train_ratio': 0.7,
            'sequence_length': 30,
            'ensemble_weights': {
                'har': 0.3,
                'rf': 0.25,
                'xgb': 0.25,
                'lstm': 0.2
            }
        }
    
    def load_data(self, ohlcv_path: str, exog_path: Optional[str] = None) -> bool:
        """Load and validate input data"""
        try:
            # Load OHLCV data
            self.logger.info(f"Loading OHLCV data from {ohlcv_path}")
            ohlcv_df = pd.read_csv(ohlcv_path)
            
            # Validate data
            is_valid, errors = self.data_validator.validate_ohlcv(ohlcv_df)
            
            if not is_valid:
                for error in errors:
                    self.logger.error(error)
                return False
            
            # Process timestamps
            ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['datetime'])
            ohlcv_df = ohlcv_df.sort_values('datetime').reset_index(drop=True)
            
            self.data = ohlcv_df
            self.logger.info(f"Loaded {len(ohlcv_df)} rows of OHLCV data")
            
            # Load exogenous data if provided
            if exog_path and os.path.exists(exog_path):
                self.logger.info(f"Loading exogenous data from {exog_path}")
                exog_df = pd.read_csv(exog_path)
                # TODO: Implement exogenous data alignment
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare features for modeling"""
        try:
            self.logger.info("Calculating realized volatility measures")
            
            # Calculate daily RV and other measures
            daily_measures = self.rv_calculator.calculate_realized_measures(self.data)
            
            # Create HAR features
            har_features = self.feature_engineer.create_har_features(
                daily_measures['RV'], self.config['har_lags']
            )
            
            # Combine features
            self.features = har_features.copy()
            self.target = daily_measures['RV']
            
            # Add realized measures as features
            for col in ['RSkew', 'RKurt', 'RJ']:
                if col in daily_measures.columns:
                    self.features[col] = daily_measures[col]
            
            # Remove rows with NaN values
            valid_idx = ~(self.features.isnull().any(axis=1) | self.target.isnull())
            self.features = self.features[valid_idx]
            self.target = self.target[valid_idx]
            
            self.logger.info(f"Prepared {len(self.features)} samples with {len(self.features.columns)} features")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            raise
    
    def train_models(self):
        """Train all models"""
        try:
            # Split data
            split_idx = int(len(self.features) * self.config['train_ratio'])
            
            X_train = self.features.iloc[:split_idx]
            y_train = self.target.iloc[:split_idx]
            X_test = self.features.iloc[split_idx:]
            y_test = self.target.iloc[split_idx:]
            
            self.logger.info(f"Training models on {len(X_train)} samples")
            
            # Train HAR model
            self.logger.info("Training HAR-RV model")
            har_features = X_train[['RV_lag_1', 'RV_lag_5', 'RV_lag_22']].dropna()
            har_target = y_train[har_features.index]
            self.har_model.fit(har_features, har_target)
            
            # Train tree models
            self.logger.info("Training Random Forest model")
            X_train_clean = X_train.fillna(X_train.mean())
            self.rf_model.fit(X_train_clean, y_train)
            
            self.logger.info("Training XGBoost model")
            self.xgb_model.fit(X_train_clean, y_train)
            
            # Train LSTM model
            self.logger.info("Training LSTM model")
            try:
                self.lstm_model.fit(X_train, y_train, epochs=50)
            except Exception as e:
                self.logger.warning(f"LSTM training failed: {e}")
            
            # Setup ensemble
            self.ensemble_model.add_model('har', self.har_model, 
                                        self.config['ensemble_weights']['har'])
            self.ensemble_model.add_model('rf', self.rf_model,
                                        self.config['ensemble_weights']['rf'])
            self.ensemble_model.add_model('xgb', self.xgb_model,
                                        self.config['ensemble_weights']['xgb'])
            if self.lstm_model.fitted:
                self.ensemble_model.add_model('lstm', self.lstm_model,
                                            self.config['ensemble_weights']['lstm'])
            
            self.ensemble_model.fitted = True
            
            self.logger.info("Model training completed")
            
            # Store test data for evaluation
            self.X_test = X_test
            self.y_test = y_test
            
        except Exception as e:
            self.logger.error(f"Failed to train models: {e}")
            raise
    
    def evaluate_models(self) -> Dict:
        """Evaluate model performance"""
        try:
            results = {}
            
            # Get predictions from all models
            X_test_clean = self.X_test.fillna(self.X_test.mean())
            
            # HAR predictions
            har_features = self.X_test[['RV_lag_1', 'RV_lag_5', 'RV_lag_22']].fillna(method='ffill')
            har_pred = self.har_model.predict(har_features)
            
            # Tree model predictions
            rf_pred = self.rf_model.predict(X_test_clean)
            xgb_pred = self.xgb_model.predict(X_test_clean)
            
            # LSTM predictions (if available)
            lstm_pred = None
            if self.lstm_model.fitted:
                try:
                    lstm_pred = self.lstm_model.predict(self.X_test, self.target.iloc[:len(self.X_test)])
                except:
                    self.logger.warning("LSTM prediction failed")
            
            # Evaluate each model
            models_pred = {
                'HAR': har_pred,
                'RF': rf_pred,
                'XGB': xgb_pred
            }
            
            if lstm_pred is not None:
                models_pred['LSTM'] = lstm_pred
            
            for name, pred in models_pred.items():
                # Align predictions with test target
                min_len = min(len(pred), len(self.y_test))
                pred_aligned = pred[:min_len]
                y_test_aligned = self.y_test.iloc[:min_len]
                
                # Calculate metrics
                mae = mean_absolute_error(y_test_aligned, pred_aligned)
                rmse = np.sqrt(mean_squared_error(y_test_aligned, pred_aligned))
                r2 = r2_score(y_test_aligned, pred_aligned)
                
                results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'predictions': pred_aligned.tolist()
                }
                
                # Statistical tests against HAR
                if name != 'HAR':
                    dm_stat, dm_pval = StatisticalTests.diebold_mariano_test(
                        pred_aligned, har_pred[:min_len], y_test_aligned
                    )
                    cw_stat, cw_pval = StatisticalTests.clark_west_test(
                        har_pred[:min_len], pred_aligned, y_test_aligned
                    )
                    
                    results[name]['DM_stat'] = dm_stat
                    results[name]['DM_pval'] = dm_pval
                    results[name]['CW_stat'] = cw_stat
                    results[name]['CW_pval'] = cw_pval
                
                self.logger.info(f"{name} - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate models: {e}")
            raise
    
    def predict_next_day(self) -> float:
        """Predict next day realized volatility"""
        try:
            # Get latest features
            latest_features = self.features.iloc[-1:].fillna(method='ffill')
            
            # Ensemble prediction
            if hasattr(self, 'ensemble_model') and self.ensemble_model.fitted:
                y_last = self.target.iloc[-self.config['sequence_length']:]
                prediction = self.ensemble_model.predict(latest_features, y_last)[0]
            else:
                # Fallback to HAR
                har_features = latest_features[['RV_lag_1', 'RV_lag_5', 'RV_lag_22']]
                prediction = self.har_model.predict(har_features)[0]
            
            self.logger.info(f"Next day RV prediction: {prediction:.6f}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Failed to make prediction: {e}")
            raise
    
    def save_models(self, output_dir: str):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save models
        joblib.dump(self.har_model, output_path / 'har_model.pkl')
        joblib.dump(self.rf_model, output_path / 'rf_model.pkl')
        joblib.dump(self.xgb_model, output_path / 'xgb_model.pkl')
        
        if self.lstm_model.fitted:
            self.lstm_model.model.save(output_path / 'lstm_model.h5')
            joblib.dump(self.lstm_model.scaler, output_path / 'lstm_scaler.pkl')
        
        # Save configuration
        with open(output_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
            
        self.logger.info(f"Models saved to {output_path}")

# Additional utility functions

def plot_predictions(true_values: np.array, predictions: Dict[str, np.array], 
                    save_path: str = None):
    """Plot predictions vs true values"""
    plt.figure(figsize=(15, 10))
    
    # Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(true_values, label='True RV', color='black', linewidth=2)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(pred, label=f'{name} Prediction', color=colors[i % len(colors)], alpha=0.7)
    
    plt.title('Realized Volatility Predictions')
    plt.xlabel('Time')
    plt.ylabel('Realized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plots for each model
    for i, (name, pred) in enumerate(predictions.items()):
        plt.subplot(2, 2, i + 2)
        plt.scatter(true_values, pred, alpha=0.6, color=colors[i % len(colors)])
        
        # Add diagonal line
        min_val, max_val = min(true_values.min(), pred.min()), max(true_values.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Calculate R²
        r2 = r2_score(true_values, pred)
        plt.title(f'{name} (R² = {r2:.3f})')
        plt.xlabel('True RV')
        plt.ylabel('Predicted RV')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def generate_config_template(output_path: str = "config_template.json"):
    """Generate configuration template"""
    config_template = {
        "har_lags": [1, 5, 22],
        "train_ratio": 0.7,
        "sequence_length": 30,
        "ensemble_weights": {
            "har": 0.3,
            "rf": 0.25,
            "xgb": 0.25,
            "lstm": 0.2
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "lstm": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping_patience": 10
        },
        "evaluation": {
            "test_size": 0.3,
            "cv_folds": 5,
            "statistical_tests": ["DM", "CW", "MCS"]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"Configuration template saved to {output_path}")

# Error handling and robustness improvements
class RobustRVPredictor(RVPredictor):
    """Enhanced RV Predictor with better error handling"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.validation_errors = []
        self.model_performance = {}
    
    def validate_predictions(self, predictions: np.array) -> bool:
        """Validate prediction quality"""
        issues = []
        
        # Check for extreme values
        if np.any(predictions < 0):
            issues.append(f"Negative volatility predictions detected: {np.sum(predictions < 0)} values")
        
        # Check for outliers (>3 sigma from recent history)
        if hasattr(self, 'target') and len(self.target) > 0:
            recent_mean = self.target.iloc[-22:].mean()  # Last month
            recent_std = self.target.iloc[-22:].std()
            
            outliers = np.abs(predictions - recent_mean) > 3 * recent_std
            if np.any(outliers):
                issues.append(f"Extreme predictions detected: {np.sum(outliers)} values >3σ from recent history")
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(predictions)):
            issues.append(f"Non-finite predictions detected: {np.sum(~np.isfinite(predictions))} values")
        
        if issues:
            for issue in issues:
                self.logger.warning(issue)
            return False
        
        return True
    
    def robust_train_models(self):
        """Train models with enhanced error handling"""
        successful_models = []
        
        try:
            super().train_models()
            
            # Test each model individually
            X_test_sample = self.X_test.iloc[:5].fillna(self.X_test.mean())
            
            # Test HAR model
            try:
                har_features = X_test_sample[['RV_lag_1', 'RV_lag_5', 'RV_lag_22']].fillna(method='ffill')
                har_pred = self.har_model.predict(har_features)
                if self.validate_predictions(har_pred):
                    successful_models.append('HAR')
            except Exception as e:
                self.logger.error(f"HAR model validation failed: {e}")
            
            # Test RF model
            try:
                rf_pred = self.rf_model.predict(X_test_sample)
                if self.validate_predictions(rf_pred):
                    successful_models.append('RF')
            except Exception as e:
                self.logger.error(f"RF model validation failed: {e}")
            
            # Test XGB model
            try:
                xgb_pred = self.xgb_model.predict(X_test_sample)
                if self.validate_predictions(xgb_pred):
                    successful_models.append('XGB')
            except Exception as e:
                self.logger.error(f"XGB model validation failed: {e}")
            
            # Test LSTM model
            if self.lstm_model.fitted:
                try:
                    y_sample = self.target.iloc[-self.config['sequence_length']:]
                    lstm_pred = self.lstm_model.predict(X_test_sample, y_sample)
                    if self.validate_predictions(lstm_pred):
                        successful_models.append('LSTM')
                except Exception as e:
                    self.logger.error(f"LSTM model validation failed: {e}")
            
            self.logger.info(f"Successfully validated models: {successful_models}")
            
            if not successful_models:
                raise ValueError("No models passed validation")
                
        except Exception as e:
            self.logger.error(f"Robust model training failed: {e}")
            raise

def predict_command(data_path: str, model_path: str, output_path: str) -> Dict:
    """
    Complete prediction method that loads trained models and makes next-day RV predictions
    
    Args:
        data_path: Path to OHLCV CSV file
        model_path: Path to directory containing trained models
        output_path: Output directory for predictions
        
    Returns:
        Dictionary containing predictions and metadata
    """

    from tensorflow.keras.models import load_model
    
    # Initialize logger
    logger = RVLogger()
    logger.info("Starting prediction process")
    
    try:
        # Validate input paths
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
            
        model_dir = Path(model_path)
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        config_path = model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.warning("Config file not found, using defaults")
            config = {
                'har_lags': [1, 5, 22],
                'sequence_length': 30,
                'ensemble_weights': {
                    'har': 0.3,
                    'rf': 0.25,
                    'xgb': 0.25,
                    'lstm': 0.2
                }
            }
        
        # Load and prepare data
        logger.info(f"Loading data from {data_path}")
        ohlcv_df = pd.read_csv(data_path)
        
        # Validate data
        data_validator = DataValidator()
        is_valid, errors = data_validator.validate_ohlcv(ohlcv_df)
        if not is_valid:
            for error in errors:
                logger.error(error)
            raise ValueError("Data validation failed")
        
        # Process data
        ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['datetime'])
        ohlcv_df = ohlcv_df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate RV and features
        logger.info("Calculating realized volatility and features")
        rv_calculator = RVCalculator()
        feature_engineer = FeatureEngineer()
        
        daily_measures = rv_calculator.calculate_realized_measures(ohlcv_df)
        har_features = feature_engineer.create_har_features(
            daily_measures['RV'], config['har_lags']
        )
        
        # Combine features
        features = har_features.copy()
        target = daily_measures['RV']
        
        # Add realized measures as features
        for col in ['RSkew', 'RKurt', 'RJ']:
            if col in daily_measures.columns:
                features[col] = daily_measures[col]
        
        # Remove rows with NaN values
        valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_idx]
        target = target[valid_idx]
        
        if len(features) == 0:
            raise ValueError("No valid data points after feature engineering")
            
        logger.info(f"Prepared {len(features)} samples for prediction")
        
        # Load trained models
        logger.info("Loading trained models")
        models = {}
        predictions = {}
        
        # Load HAR model
        har_model_path = model_dir / 'har_model.pkl'
        if har_model_path.exists():
            models['HAR'] = joblib.load(har_model_path)
            logger.info("HAR model loaded")
        else:
            logger.warning("HAR model not found")
        
        # Load Random Forest model
        rf_model_path = model_dir / 'rf_model.pkl'
        if rf_model_path.exists():
            models['RF'] = joblib.load(rf_model_path)
            logger.info("Random Forest model loaded")
        else:
            logger.warning("Random Forest model not found")
        
        # Load XGBoost model
        xgb_model_path = model_dir / 'xgb_model.pkl'
        if xgb_model_path.exists():
            models['XGB'] = joblib.load(xgb_model_path)
            logger.info("XGBoost model loaded")
        else:
            logger.warning("XGBoost model not found")
        
        # Load LSTM model
        lstm_model_path = model_dir / 'lstm_model.h5'
        lstm_scaler_path = model_dir / 'lstm_scaler.pkl'
        if lstm_model_path.exists() and lstm_scaler_path.exists():
            try:
                from tensorflow.keras.models import load_model
                lstm_model = load_model(lstm_model_path)
                lstm_scaler = joblib.load(lstm_scaler_path)
                models['LSTM'] = {'model': lstm_model, 'scaler': lstm_scaler}
                logger.info("LSTM model loaded")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
        else:
            logger.warning("LSTM model files not found")
        
        if not models:
            raise ValueError("No trained models found")
        
        # Make predictions with each model
        logger.info("Making predictions")
        
        # Get latest features for prediction
        latest_features = features.iloc[-1:].copy()
        latest_features_clean = latest_features.fillna(features.mean())
        
        # HAR model prediction
        if 'HAR' in models:
            try:
                har_features_cols = ['RV_lag_1', 'RV_lag_5', 'RV_lag_22']
                har_features_latest = latest_features[har_features_cols].fillna(method='ffill')
                har_pred = models['HAR'].predict(har_features_latest)[0]
                predictions['HAR'] = float(har_pred)
                logger.info(f"HAR prediction: {har_pred:.6f}")
            except Exception as e:
                logger.error(f"HAR prediction failed: {e}")
        
        # Random Forest prediction
        if 'RF' in models:
            try:
                rf_pred = models['RF'].predict(latest_features_clean)[0]
                predictions['RF'] = float(rf_pred)
                logger.info(f"Random Forest prediction: {rf_pred:.6f}")
            except Exception as e:
                logger.error(f"Random Forest prediction failed: {e}")
        
        # XGBoost prediction
        if 'XGB' in models:
            try:
                xgb_pred = models['XGB'].predict(latest_features_clean)[0]
                predictions['XGB'] = float(xgb_pred)
                logger.info(f"XGBoost prediction: {xgb_pred:.6f}")
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
        
        # LSTM prediction
        if 'LSTM' in models:
            try:
                lstm_model = models['LSTM']['model']
                lstm_scaler = models['LSTM']['scaler']
                
                # Prepare sequence data
                sequence_length = config['sequence_length']
                
                # Get last sequence of target and features
                if len(target) >= sequence_length:
                    sequence_data = pd.concat([target.iloc[-sequence_length:], 
                                             features.iloc[-sequence_length:]], axis=1)
                else:
                    # Pad with mean values if insufficient data
                    mean_vals = pd.concat([target, features], axis=1).mean()
                    padding_rows = sequence_length - len(target)
                    padding_df = pd.DataFrame([mean_vals] * padding_rows, 
                                            columns=pd.concat([target, features], axis=1).columns)
                    sequence_data = pd.concat([padding_df, pd.concat([target, features], axis=1)])
                    sequence_data = sequence_data.iloc[-sequence_length:]
                
                # Scale and predict
                scaled_data = lstm_scaler.transform(sequence_data)
                X_seq = scaled_data.reshape(1, sequence_length, -1)
                
                lstm_pred_scaled = lstm_model.predict(X_seq, verbose=0)[0, 0]
                
                # Inverse transform
                dummy_data = np.zeros((1, scaled_data.shape[1]))
                dummy_data[0, 0] = lstm_pred_scaled
                lstm_pred = lstm_scaler.inverse_transform(dummy_data)[0, 0]
                
                predictions['LSTM'] = float(lstm_pred)
                logger.info(f"LSTM prediction: {lstm_pred:.6f}")
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")
        
        # Calculate ensemble prediction
        if len(predictions) > 1:
            ensemble_weights = config['ensemble_weights']
            weighted_sum = 0
            total_weight = 0
            
            for model_name, pred_value in predictions.items():
                weight = ensemble_weights.get(model_name.lower(), 1.0)
                weighted_sum += pred_value * weight
                total_weight += weight
            
            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
            predictions['Ensemble'] = float(ensemble_pred)
            logger.info(f"Ensemble prediction: {ensemble_pred:.6f}")
        
        # Prepare results
        results = {
            'predictions': predictions,
            'metadata': {
                'prediction_date': datetime.now().isoformat(),
                'data_end_date': ohlcv_df['datetime'].max().isoformat(),
                'models_used': list(predictions.keys()),
                'data_points': len(ohlcv_df),
                'latest_rv': float(target.iloc[-1]) if len(target) > 0 else None,
                'config': config
            }
        }
        
        # Save predictions
        predictions_file = output_dir / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV format
        pred_df = pd.DataFrame([predictions])
        pred_df.insert(0, 'prediction_date', datetime.now().strftime('%Y-%m-%d'))
        pred_df.insert(1, 'data_end_date', ohlcv_df['datetime'].max().strftime('%Y-%m-%d'))
        
        csv_file = output_dir / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        pred_df.to_csv(csv_file, index=False)
        
        logger.info(f"Predictions saved to {predictions_file} and {csv_file}")
        
        # Display results
        print("\n" + "="*60)
        print("REALIZED VOLATILITY PREDICTIONS")
        print("="*60)
        print(f"Data Period: {ohlcv_df['datetime'].min().strftime('%Y-%m-%d')} to {ohlcv_df['datetime'].max().strftime('%Y-%m-%d')}")
        print(f"Latest RV: {target.iloc[-1]:.6f}" if len(target) > 0 else "Latest RV: N/A")
        print("Latest RV ")
        print(f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*60)
        
        for model_name, pred_value in predictions.items():
            print(f"{model_name:12}: {pred_value:.6f}")
        
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction process failed: {e}")
        raise
def app_log(msg):
    print(f"APPLOG: {msg}")

# Update the main function to use the new predict_command
def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Realized Volatility Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train volatility prediction models')
    train_parser.add_argument('--data', required=True, help='Path to OHLCV CSV file')
    train_parser.add_argument('--output', required=True, help='Output directory for models')
    train_parser.add_argument('--exog', help='Path to exogenous variables CSV file')
    train_parser.add_argument('--config', help='Path to configuration JSON file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make volatility predictions')
    predict_parser.add_argument('--data', required=True, help='Path to OHLCV CSV file')
    predict_parser.add_argument('--model', required=True, help='Path to trained models directory')
    predict_parser.add_argument('--output', required=True, help='Output directory for predictions')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--true', required=True, help='Path to true RV values CSV')
    evaluate_parser.add_argument('--pred', required=True, help='Path to predictions CSV')
    evaluate_parser.add_argument('--metrics', required=True, help='Output path for metrics JSON')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Load configuration
        config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Initialize predictor
        predictor = RVPredictor(config)
        
        # Load data
        if not predictor.load_data(args.data, args.exog):
            print("Failed to load data. Check logs for details.")
            return 1
        
        # Prepare features and train models
        predictor.prepare_features()
        predictor.train_models()
        
        # Evaluate models
        results = predictor.evaluate_models()
        
        # Save models and results
        predictor.save_models(args.output)
        
        # Save evaluation results
        results_path = Path(args.output) / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Training completed. Models saved to {args.output}")
        
        # Display best model
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        print(f"Best model: {best_model} (RMSE: {results[best_model]['RMSE']:.6f})")
        
    elif args.command == 'predict':
        # Use the new predict_command function
        try:
            results = predict_command(args.data, args.model, args.output)
            
            # Display summary
            predictions = results['predictions']
            if 'Ensemble' in predictions:
                print(f"\nRecommended prediction (Ensemble): {predictions['Ensemble']:.6f}")
                annualized_rv = np.sqrt(predictions['Ensemble'] * 252) * 100
                print(f"Annualized RV: {annualized_rv:.6f}")
                app_log(f"Annualized Volatility (%): {annualized_rv:.4f}")
            else:
                best_pred = list(predictions.values())[0]
                best_model = list(predictions.keys())[0]
                print(f"\nPrediction ({best_model}): {best_pred:.6f}")
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            return 1
        
    elif args.command == 'evaluate':
        # Standalone evaluation
        print("Evaluation mode - implementation would compare predictions to true values")
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    
    # Check if generating config template
    if len(sys.argv) > 1 and sys.argv[1] == '--config-template':
        generate_config_template()
        sys.exit(0)
    
    # Run main CLI
    sys.exit(main())



    ### Installation
"""

pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow joblib scipy

FIRST: 

zGen2\\.venv\\Scripts\\activate
### Basic Usage

1. **Validate System**:

python zGen2\\main.py --validate


2. **Generate Configuration Template**:

python zGen2\\main.py --config-template


3. **Train Models**:

python zGen2\\main.py train --data data\SPY.csv --output models/


4. **Make Predictions**:

python zGen2\\main.py predict --data data\SPY.csv --model models/ --output forecasts/
python zGen2\\main.py predict --data sliced_data\\2022-05-31.csv  --model models/ --output forecasts/
"""
