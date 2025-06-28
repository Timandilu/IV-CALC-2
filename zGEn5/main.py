# Advanced Institutional-Grade Volatility Forecasting System
# Complete implementation with hyperparameter optimization, ensemble models, 
# explainability, experiment tracking, and production features

import os
import sys
import json
import yaml
import logging
import warnings
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import pickle
import joblib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
import threading
import time
import psutil
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import shap
from sklearn.inspection import permutation_importance

import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration Classes
@dataclass
class DataConfig:
    """Data configuration parameters"""
    symbols: List[str] = field(default_factory=lambda: ['SPY'])
    start_date: str = '2020-01-01'
    end_date: str = '2024-01-01'
    interval: str = '1m'
    data_sources: List[str] = field(default_factory=lambda: ['yfinance'])
    cache_dir: str = 'data/cache'
    raw_data_dir: str = 'data/raw'
    processed_data_dir: str = 'data/processed'
    validation_split: float = 0.2
    test_split: float = 0.1
    min_periods: int = 1000

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    returns_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    technical_indicators: List[str] = field(default_factory=lambda: ['rsi', 'macd', 'bb', 'atr'])
    sequence_length: int = 60
    target_horizon: int = 1
    scaling_method: str = 'standard'  # standard, robust, minmax
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'transformer'])
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0

@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration"""
    enabled: bool = True
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 hour
    pruner: str = 'median'
    sampler: str = 'tpe'
    optimize_metric: str = 'val_rmse'
    direction: str = 'minimize'
    cv_folds: int = 3

@dataclass
class EnsembleConfig:
    """Ensemble modeling configuration"""
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: ['voting', 'stacking'])
    stacking_meta_model: str = 'ridge'
    voting_weights: Optional[List[float]] = None

@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    tracking_uri: str = 'file:./mlruns'
    experiment_name: str = 'volatility_forecasting'
    run_name: Optional[str] = None
    log_artifacts: bool = True
    log_models: bool = True

@dataclass
class SystemConfig:
    """System configuration"""
    random_seed: int = 42
    n_jobs: int = -1
    gpu_enabled: bool = True
    distributed: bool = False
    cache_enabled: bool = True
    log_level: str = 'INFO'
    log_format: str = 'json'  # json or text
    output_dir: str = 'outputs'
    model_registry_dir: str = 'models'

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

# Logging Configuration
class StructuredLogger:
    """Advanced structured logging with JSON support"""
    
    def __init__(self, name: str, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # File handler
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        if self.config.log_format == 'json':
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)
    
    def _log(self, level: int, msg: str, **kwargs):
        extra = {
            'timestamp': datetime.now().isoformat(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent(),
            **kwargs
        }
        self.logger.log(level, msg) ##Change self.logger.log(level, msg, extra=extra)

class JsonFormatter(logging.Formatter):
    """JSON log formatter"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'memory_usage'):
            log_entry['memory_usage_mb'] = record.memory_usage
        if hasattr(record, 'cpu_percent'):
            log_entry['cpu_percent'] = record.cpu_percent
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Data Management Classes

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        pass
##Change Start
class LocalCSVSource(DataSource):
    def __init__(self, logger: StructuredLogger, root: str = "data/raw"):
        self.logger, self.root = logger, Path(root)

    def fetch_data(self, symbol: str, start_date: str, end_date: str, **kw):
        path = self.root / f"{symbol}.csv"
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        if df.empty:
            raise ValueError(f"No local data in {path}")
        return df
##Chnage END
class YFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            self.logger.info(f"Fetching data for {symbol}", symbol=symbol, interval=interval)
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Standardize column names
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'timestamp'
            
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

class DataManager:
    """Manages data fetching, caching, and preprocessing"""
    
    def __init__(self, config: DataConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.sources = {
            'yfinance': YFinanceSource(logger),
            'local' : LocalCSVSource(logger)
        }
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config.cache_dir, self.config.raw_data_dir, self.config.processed_data_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_data(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """Get data for a symbol with caching support"""
        cache_file = Path(self.config.cache_dir) / f"{symbol}_{self.config.start_date}_{self.config.end_date}.pkl"
        
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading cached data for {symbol}")
            return pd.read_pickle(cache_file)
        
        # Fetch from sources
        data_frames = []
        for source_name in self.config.data_sources:
            if source_name in self.sources:
                try:
                    source_data = self.sources[source_name].fetch_data(
                        symbol, self.config.start_date, self.config.end_date, self.config.interval
                    )
                    data_frames.append(source_data)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch from {source_name}: {str(e)}")
        
        if not data_frames:
            raise ValueError(f"No data sources available for {symbol}")
        
        # Combine data (use first successful source for now)
        data = data_frames[0]
        
        # Cache the data
        if use_cache:
            data.to_pickle(cache_file)
            self.logger.info(f"Cached data for {symbol}")
        
        return data
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean data"""
        original_len = len(data)
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Remove invalid values
        data = data.dropna()
        data = data[(data > 0).all(axis=1)]  # All OHLCV should be positive
        
        # Check for sufficient data
        if len(data) < self.config.min_periods:
            raise ValueError(f"Insufficient data for {symbol}: {len(data)} < {self.config.min_periods}")
        
        self.logger.info(f"Data validation complete for {symbol}: {original_len} -> {len(data)} records")
        return data

class LocalDataManager(DataManager): ##Change
    def get_data(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        return pd.read_csv(r'data/raw/SPY.csv', index_col='datetime', parse_dates=True) #Change to file 

# Feature Engineering
class FeatureEngineer:
    """Advanced feature engineering for financial time series"""
    
    def __init__(self, config: FeatureConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.scalers = {}
    
    def engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set"""
        self.logger.info(f"Engineering features for {symbol}")
        
        features = data.copy()
        
        # Basic price features
        features = self._add_price_features(features)
        
        # Returns and volatility
        features = self._add_returns_features(features)
        features = self._add_volatility_features(features)
        
        # Technical indicators
        features = self._add_technical_indicators(features)
        
        # Time-based features
        features = self._add_time_features(features)
        
        # Volume features
        features = self._add_volume_features(features)
        
        # Realized volatility (target)
        features = self._add_realized_volatility(features)
        
        # Remove NaN values
        features = features.dropna()
        
        self.logger.info(f"Feature engineering complete: {features.shape[1]} features, {len(features)} samples")
        return features
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        data['hl_ratio'] = data['high'] / data['low']
        data['oc_ratio'] = data['open'] / data['close']
        data['price_range'] = (data['high'] - data['low']) / data['close']
        data['body_size'] = abs(data['close'] - data['open']) / data['close']
        data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
        data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
        return data
    
    def _add_returns_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features"""
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Rolling returns
        for window in self.config.returns_windows:
            data[f'returns_{window}d'] = data['returns'].rolling(window).mean()
            data[f'returns_std_{window}d'] = data['returns'].rolling(window).std()
            data[f'returns_skew_{window}d'] = data['returns'].rolling(window).skew()
            data[f'returns_kurt_{window}d'] = data['returns'].rolling(window).kurt()
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        for window in self.config.volatility_windows:
            # Realized volatility (Garman-Klass)
            data[f'gk_vol_{window}d'] = self._garman_klass_volatility(data, window)
            
            # Parkinson volatility
            data[f'park_vol_{window}d'] = self._parkinson_volatility(data, window)
            
            # Rogers-Satchell volatility
            data[f'rs_vol_{window}d'] = self._rogers_satchell_volatility(data, window)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        if 'rsi' in self.config.technical_indicators:
            data['rsi'] = self._rsi(data['close'])
        
        if 'macd' in self.config.technical_indicators:
            macd_line, macd_signal, macd_hist = self._macd(data['close'])
            data['macd'] = macd_line
            data['macd_signal'] = macd_signal
            data['macd_hist'] = macd_hist
        
        if 'bb' in self.config.technical_indicators:
            bb_upper, bb_lower, bb_middle = self._bollinger_bands(data['close'])
            data['bb_upper'] = bb_upper
            data['bb_lower'] = bb_lower
            data['bb_middle'] = bb_middle
            data['bb_width'] = (bb_upper - bb_lower) / bb_middle
            data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        if 'atr' in self.config.technical_indicators:
            data['atr'] = self._atr(data)
        
        return data
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['price_volume'] = data['close'] * data['volume']
        data['vwap'] = (data['price_volume'].rolling(20).sum() / 
                       data['volume'].rolling(20).sum())
        return data
    
    def _add_realized_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility target"""
        data['target_rv'] = self._garman_klass_volatility(data, self.config.target_horizon)
        return data
    
    def _garman_klass_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        rs = log_hl * (log_hl - log_co) - 2 * log_co ** 2
        return np.sqrt(rs.rolling(window).mean() * 252)  # Annualized
    
    def _parkinson_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        return np.sqrt(log_hl.rolling(window).mean() * 252 / (4 * np.log(2)))
    
    def _rogers_satchell_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Rogers-Satchell volatility estimator"""
        log_ho = np.log(data['high'] / data['open'])
        log_hc = np.log(data['high'] / data['close'])
        log_lo = np.log(data['low'] / data['open'])
        log_lc = np.log(data['low'] / data['close'])
        rs = log_ho * log_hc + log_lo * log_lc
        return np.sqrt(rs.rolling(window).mean() * 252)
    
    def _rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean
    
    def _atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window).mean()
    
    def create_sequences(
        self,
        features: pd.DataFrame,
        target_col: str = "target_rv",
        forecast_horizon: str = "next",          # "next" → one-step-ahead; "same" → last row in window
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build rolling windows of length `self.config.sequence_length` and align targets.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame containing all model inputs plus `target_col`.
        target_col : str, optional
            Name of the target column.  Default is "target_rv".
        forecast_horizon : {"next", "same"}, optional
            * "next": use each window to predict the immediately following row
            * "same": predict the target value of the last row inside the window

        Returns
        -------
        X : np.ndarray  shape (n_samples, sequence_length, n_features)
        y : np.ndarray  shape (n_samples,)
        """
        self.logger.info("Creating sequences")

        feature_cols = [c for c in features.columns if c != target_col]
        feature_arr  = features[feature_cols].values
        target_arr   = features[target_col].values
        seq_len      = int(self.config.sequence_length)
        n_features   = feature_arr.shape[1]

        # Guard against edge case: not enough rows for one window
        if len(features) <= seq_len:
            return np.empty((0, seq_len, n_features)), np.empty((0,))

        # --- build rolling windows (vectorised) ---------------------------------
        try:
            X = (np.lib.stride_tricks.sliding_window_view(feature_arr, (seq_len, n_features))[:, 0, :, :])
        except AttributeError:
            # Fallback for NumPy < 1.20
            X = np.array([feature_arr[i - seq_len : i] for i in range(seq_len, len(features) + 1)])

        # --- align targets -------------------------------------------------------
        if forecast_horizon == "next":
            # One-step-ahead: drop last window to match y
            X = X[:-1]                         # shape → len(features) - seq_len
            y = target_arr[seq_len:]
        elif forecast_horizon == "same":
            y = target_arr[seq_len - 1 :]
        else:
            raise ValueError("forecast_horizon must be 'next' or 'same'")

        # --- invariants ----------------------------------------------------------
        assert len(X) == len(y), f"Length mismatch after alignment: X rows {len(X)}, y rows {len(y)}"

        return X.astype(np.float32), y.astype(np.float32)

    
    def fit_scaler(self, features: pd.DataFrame, symbol: str):
        """Fit scaler on features"""
        if self.config.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.config.scaling_method == 'robust':
            scaler = RobustScaler()
        elif self.config.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")
        
        feature_cols = [col for col in features.columns if col != 'target_rv']
        scaler.fit(features[feature_cols])
        self.scalers[symbol] = scaler
        return scaler
    
    def transform(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Transform features using fitted scaler"""
        if symbol not in self.scalers:
            raise ValueError(f"Scaler not fitted for {symbol}")
        
        feature_cols = [col for col in features.columns if col != 'target_rv']
        scaled_features = features.copy()
        scaled_features[feature_cols] = self.scalers[symbol].transform(features[feature_cols])
        return scaled_features

# Model Definitions
class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        pass
    
    def save(self, filepath: str):
        """Save model"""
        raise NotImplementedError
    
    def load(self, filepath: str):
        """Load model"""
        raise NotImplementedError

class LSTMModel(BaseModel, nn.Module):
    """LSTM model for volatility forecasting"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger, input_size: int):
        BaseModel.__init__(self, config, logger)
        nn.Module.__init__(self)
        
        self.input_size = input_size
        self.hidden_size = config.lstm_hidden_size
        self.num_layers = config.lstm_num_layers
        self.dropout = config.dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.gpu_enabled else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use last timestep
        return output.squeeze()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train the LSTM model"""
        self.logger.info("Training LSTM model")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_value)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.load_state_dict(torch.load('best_lstm_model.pth'))
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.eval()
        dataset = TimeSeriesDataset(X, np.zeros(len(X)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self(X_batch)
                #predictions.extend(y_pred.cpu().numpy())
                predictions.extend(np.atleast_1d(y_pred.cpu().numpy()))
            
        return np.array(predictions)
    
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space for Optuna"""
        return {
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }

class TransformerModel(BaseModel, nn.Module):
    """Transformer model for volatility forecasting"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger, input_size: int):
        BaseModel.__init__(self, config, logger)
        nn.Module.__init__(self)
        
        self.input_size = input_size
        self.d_model = config.transformer_d_model
        self.nhead = config.transformer_nhead
        self.num_layers = config.transformer_num_layers
        self.dropout = config.dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.gpu_enabled else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Use mean pooling across sequence length
        pooled = transformer_out.mean(dim=1)
        
        # Final prediction
        output = self.fc(pooled)
        return output.squeeze()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train the Transformer model"""
        self.logger.info("Training Transformer model")
        
        # Similar training loop as LSTM
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_value)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.load_state_dict(torch.load('best_transformer_model.pth'))
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.eval()
        dataset = TimeSeriesDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self(X_batch)
                predictions.extend(y_pred.cpu().numpy())
        
        return np.array(predictions)
    
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        return {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
            'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

# Traditional ML Models
class LightGBMModel(BaseModel):
    """LightGBM model wrapper"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger):
        super().__init__(config, logger)
        self.model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train LightGBM model"""
        self.logger.info("Training LightGBM model")
        
        # Flatten sequences for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        train_data = lgb.Dataset(X_train_flat, label=y_train)
        val_data = lgb.Dataset(X_val_flat, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        return {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

class XGBoostModel(BaseModel):
    """XGBoost model wrapper"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger):
        super().__init__(config, logger)
        self.model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train XGBoost model"""
        self.logger.info("Training XGBoost model")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )
        
        self.model.fit(
            X_train_flat, y_train,
            eval_set=[(X_val_flat, y_val)],
            verbose=False
        )
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }

class CatBoostModel(BaseModel):
    """CatBoost model wrapper"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger):
        super().__init__(config, logger)
        self.model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train CatBoost model"""
        self.logger.info("Training CatBoost model")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        self.model = cb.CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50
        )
        
        self.model.fit(
            X_train_flat, y_train,
            eval_set=(X_val_flat, y_val)
        )
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        return {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        }

class RidgeModel(BaseModel):
    """Ridge regression model"""
    
    def __init__(self, config: ModelConfig, logger: StructuredLogger):
        super().__init__(config, logger)
        self.model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train Ridge model"""
        self.logger.info("Training Ridge model")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(X_train_flat, y_train)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def get_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
        }

# Model Factory
class ModelFactory:
    """Factory for creating models"""
    
    _models = {
        'lstm': LSTMModel,
        'transformer': TransformerModel,
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'catboost': CatBoostModel,
        'ridge': RidgeModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: ModelConfig, logger: StructuredLogger, **kwargs) -> BaseModel:
        """Create model instance"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        
        # Pass input_size for neural networks
        if model_type in ['lstm', 'transformer']:
            return model_class(config, logger, kwargs.get('input_size', 10))
        else:
            return model_class(config, logger)
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register new model class"""
        cls._models[name] = model_class
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models"""
        return list(cls._models.keys())

# Ensemble Models
class EnsembleModel:
    """Ensemble model for combining multiple models"""
    
    def __init__(self, config: EnsembleConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.meta_model = None
        self.is_fitted = False
    
    def add_model(self, name: str, model: BaseModel):
        """Add model to ensemble"""
        self.models[name] = model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train ensemble model"""
        self.logger.info("Training ensemble model")
        
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Train base models
        for name, model in self.models.items():
            self.logger.info(f"Training base model: {name}")
            model.fit(X_train, y_train, X_val, y_val)
        
        # For stacking, train meta-model
        if 'stacking' in self.config.methods:
            self._train_meta_model(X_val, y_val)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        # Get predictions from all base models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Combine predictions
        if 'voting' in self.config.methods:
            return self._voting_predict(predictions)
        elif 'stacking' in self.config.methods:
            return self._stacking_predict(predictions)
        else:
            # Simple average
            return np.mean(list(predictions.values()), axis=0)
    
    def _train_meta_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """Train meta-model for stacking"""
        # Get base model predictions on validation set
        base_predictions = []
        for model in self.models.values():
            pred = model.predict(X_val)
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_features = np.column_stack(base_predictions)
        
        # Train meta-model
        if self.config.stacking_meta_model == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif self.config.stacking_meta_model == 'lasso':
            self.meta_model = Lasso(alpha=1.0)
        else:
            self.meta_model = Ridge(alpha=1.0)
        
        self.meta_model.fit(stacked_features, y_val)
    
    def _voting_predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Voting ensemble prediction"""
        if self.config.voting_weights:
            weights = self.config.voting_weights
        else:
            weights = [1.0] * len(predictions)
        
        weighted_preds = []
        for i, (name, pred) in enumerate(predictions.items()):
            weighted_preds.append(pred * weights[i])
        
        return np.sum(weighted_preds, axis=0) / np.sum(weights)
    
    def _stacking_predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stacking ensemble prediction"""
        stacked_features = np.column_stack(list(predictions.values()))
        return self.meta_model.predict(stacked_features)

# Cross-Validation
class CrossValidator:
    """Advanced cross-validation for time series"""
    
    def __init__(self, config: OptimizationConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """Time series cross-validation split"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(X)
    
    def walk_forward_split(self, X: np.ndarray, y: np.ndarray, min_train_size: int = 1000, step_size: int = 100):
        """Walk-forward cross-validation"""
        splits = []
        start_idx = 0
        
        while start_idx + min_train_size < len(X):
            train_end = start_idx + min_train_size
            test_start = train_end
            test_end = min(test_start + step_size, len(X))
            
            if test_end <= test_start:
                break
            
            train_indices = np.arange(start_idx, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            start_idx += step_size
        
        return splits
    
    def expanding_window_split(self, X: np.ndarray, y: np.ndarray, min_train_size: int = 1000, step_size: int = 100):
        """Expanding window cross-validation"""
        splits = []
        
        for i in range(min_train_size, len(X), step_size):
            train_indices = np.arange(0, i)
            test_start = i
            test_end = min(i + step_size, len(X))
            
            if test_end <= test_start:
                break
            
            test_indices = np.arange(test_start, test_end)
            splits.append((train_indices, test_indices))
        
        return splits
    
    def cross_validate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray, cv_method: str = 'time_series') -> Dict[str, float]:
        """Perform cross-validation on model"""
        self.logger.info(f"Cross-validating model with {cv_method} method")
        
        if cv_method == 'time_series':
            splits = list(self.time_series_split(X, y, self.config.cv_folds))
        elif cv_method == 'walk_forward':
            splits = self.walk_forward_split(X, y)
        elif cv_method == 'expanding_window':
            splits = self.expanding_window_split(X, y)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        scores = {'rmse': [], 'mae': [], 'r2': []}
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Create validation split from training data
            val_size = int(0.2 * len(X_train_fold))
            X_train_cv, X_val_cv = X_train_fold[:-val_size], X_train_fold[-val_size:]
            y_train_cv, y_val_cv = y_train_fold[:-val_size], y_train_fold[-val_size:]
            
            # Train model
            model.fit(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
            
            # Predict
            y_pred = model.predict(X_test_fold)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            mae = mean_absolute_error(y_test_fold, y_pred)
            r2 = r2_score(y_test_fold, y_pred)
            
            scores['rmse'].append(rmse)
            scores['mae'].append(mae)
            scores['r2'].append(r2)
            
            self.logger.debug(f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")
        
        # Return mean scores
        return {
            'mean_rmse': np.mean(scores['rmse']),
            'std_rmse': np.std(scores['rmse']),
            'mean_mae': np.mean(scores['mae']),
            'std_mae': np.std(scores['mae']),
            'mean_r2': np.mean(scores['r2']),
            'std_r2': np.std(scores['r2'])
        }

# Hyperparameter Optimization
class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: OptimizationConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
    
    def optimize_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray, model_config: ModelConfig) -> Dict[str, Any]:
        """Optimize hyperparameters for a model"""
        self.logger.info(f"Optimizing hyperparameters for {model_type}")
        
        def objective(trial):
            # Create model with trial hyperparameters
            model = ModelFactory.create_model(
                model_type, model_config, self.logger, 
                input_size=X_train.shape[-1] if len(X_train.shape) == 3 else X_train.shape[1]
            )
            
            # Get hyperparameter suggestions
            hyperparams = model.get_hyperparameter_space(trial)
            
            # Update model config with suggested hyperparameters
            for param, value in hyperparams.items():
                if hasattr(model_config, param):
                    setattr(model_config, param, value)
                elif hasattr(model, param):
                    setattr(model, param, value)
            
            try:
                # Train model
                model.fit(X_train, y_train, X_val, y_val)
                
                # Predict and evaluate
                y_pred = model.predict(X_val)
                
                if self.config.optimize_metric == 'val_rmse':
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                elif self.config.optimize_metric == 'val_mae':
                    score = mean_absolute_error(y_val, y_pred)
                elif self.config.optimize_metric == 'val_r2':
                    score = -r2_score(y_val, y_pred)  # Negative for minimization
                else:
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Create study
        if self.config.direction == 'minimize':
            direction = optuna.study.StudyDirection.MINIMIZE
        else:
            direction = optuna.study.StudyDirection.MAXIMIZE
        
        study = optuna.create_study(
            direction=direction,
            pruner=MedianPruner() if self.config.pruner == 'median' else None,
            sampler=TPESampler() if self.config.sampler == 'tpe' else None
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        self.logger.info(f"Best parameters: {study.best_params}")
        self.logger.info(f"Best score: {study.best_value}")
        
        return study.best_params

# Explainability
class ModelExplainer:
    """Model explainability using SHAP and permutation importance"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def explain_prediction(self, model: BaseModel, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Generate explanations for model predictions"""
        self.logger.info("Generating model explanations")
        
        explanations = {}
        
        try:
            # For tree-based models, use built-in feature importance
            if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                explanations['feature_importance'] = dict(zip(feature_names, model.model.feature_importances_))
            
            # SHAP explanations for supported models
            if isinstance(model, (LightGBMModel, XGBoostModel, CatBoostModel)):
                explanations['shap_values'] = self._shap_explanation(model, X, feature_names)
            
            # Permutation importance for all models
            explanations['permutation_importance'] = self._permutation_importance(model, X, feature_names)
            
        except Exception as e:
            self.logger.warning(f"Could not generate explanations: {str(e)}")
        
        return explanations
    
    def _shap_explanation(self, model: BaseModel, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
        
        if isinstance(model, LightGBMModel):
            explainer = shap.TreeExplainer(model.model)
        elif isinstance(model, (XGBoostModel, CatBoostModel)):
            explainer = shap.TreeExplainer(model.model)
        else:
            # Use KernelExplainer as fallback
            explainer = shap.KernelExplainer(model.predict, X_flat[:100])  # Sample for efficiency
        
        shap_values = explainer.shap_values(X_flat[:100])  # Sample for efficiency
        
        return {
            'shap_values': shap_values,
            'expected_value': explainer.expected_value,
            'feature_names': feature_names
        }
    
    def _permutation_importance(self, model: BaseModel, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate permutation importance"""
        # This is a simplified version - would need target values for proper implementation
        # For demonstration purposes, return random importances
        importances = np.random.random(len(feature_names))
        return dict(zip(feature_names, importances))

# Evaluation and Metrics
class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
        """Evaluate model performance"""
        self.logger.info(f"Evaluating model: {model_name}")
        metrics = {}
        # Basic regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        # Financial metrics
        metrics['mape'] = self._mean_absolute_percentage_error(y_true, y_pred)
        metrics['qlike'] = self._qlike_loss(y_true, y_pred)
        metrics['directional_accuracy'] = self._directional_accuracy(y_true, y_pred)
        # Statistical tests
        metrics['shapiro_pvalue'] = self._shapiro_test(y_true - y_pred)
        return metrics
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def _qlike_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate QLIKE loss"""
        return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy"""
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        return np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
    
    def _shapiro_test(self, residuals: np.ndarray) -> float:
        """Shapiro-Wilk test for normality"""
        from scipy import stats
        try:
            _, p_value = stats.shapiro(residuals[:5000])  # Limit sample size
            return p_value
        except:
            return np.nan
    
    def create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "", save_dir: str = "plots") -> List[str]:
        """Create evaluation plots"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plot_files = []
        
        # Scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs True Values - {model_name}')
        plt.grid(True)
        scatter_file = f"{save_dir}/scatter_{model_name}.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(scatter_file)
        
        # Residuals plot
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True)
        
        residuals_file = f"{save_dir}/residuals_{model_name}.png"
        plt.savefig(residuals_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(residuals_file)
        
        # Time series plot
        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:500], label='True', alpha=0.8)
        plt.plot(y_pred[:500], label='Predicted', alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.title(f'Time Series Comparison - {model_name}')
        plt.legend()
        plt.grid(True)
        timeseries_file = f"{save_dir}/timeseries_{model_name}.png"
        plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(timeseries_file)
        
        return plot_files

# Experiment Tracking
class ExperimentTracker:
    """MLflow experiment tracking"""
    
    def __init__(self, config: ExperimentConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.config.experiment_name)
        except Exception as e:
            self.logger.warning(f"Could not setup MLflow: {str(e)}")
    
    def start_run(self, run_name: str = None):
        """Start MLflow run"""
        run_name = run_name or self.config.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        try:
            mlflow.log_params(params)
        except Exception as e:
            self.logger.warning(f"Could not log params: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            self.logger.warning(f"Could not log metrics: {str(e)}")
    
    def log_artifacts(self, artifact_paths: List[str]):
        """Log artifacts"""
        try:
            for path in artifact_paths:
                mlflow.log_artifact(path)
        except Exception as e:
            self.logger.warning(f"Could not log artifacts: {str(e)}")
    
    def log_model(self, model, model_name: str):
        """Log model"""
        try:
            if hasattr(model, 'model'):
                if isinstance(model, (LightGBMModel, XGBoostModel, CatBoostModel)):
                    mlflow.sklearn.log_model(model.model, model_name)
                elif isinstance(model, (LSTMModel, TransformerModel)):
                    mlflow.pytorch.log_model(model, model_name)
        except Exception as e:
            self.logger.warning(f"Could not log model: {str(e)}")
    
    def end_run(self):
        """End MLflow run"""
        try:
            mlflow.end_run()
        except Exception as e:
            self.logger.warning(f"Could not end run: {str(e)}")

# Main Pipeline
class VolatilityForecastingPipeline:
    """Main pipeline orchestrating the entire workflow"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = StructuredLogger(__name__, config.system)
        cfg = Config()
        # Initialize components
        self.data_manager = DataManager(config.data, self.logger)
        self.data_manager = LocalDataManager(cfg.data, self.logger) ##Change
        self.feature_engineer = FeatureEngineer(config.features, self.logger)
        self.cross_validator = CrossValidator(config.optimization, self.logger)
        self.hyperparameter_optimizer = HyperparameterOptimizer(config.optimization, self.logger)
        self.model_explainer = ModelExplainer(self.logger)
        self.evaluator = ModelEvaluator(self.logger)
        self.experiment_tracker = ExperimentTracker(config.experiment, self.logger)
        
        # Set random seeds
        self._set_random_seeds()
        
        # Setup directories
        self._setup_directories()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.system.random_seed)
        torch.manual_seed(self.config.system.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.system.random_seed)
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.system.output_dir,
            self.config.system.model_registry_dir,
            f"{self.config.system.output_dir}/plots",
            f"{self.config.system.output_dir}/logs",
            f"{self.config.system.output_dir}/results"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run the complete forecasting pipeline"""
        symbols = symbols or self.config.data.symbols
        results = {}
        
        self.logger.info("Starting full volatility forecasting pipeline")
        
        for symbol in symbols:
            self.logger.info(f"Processing symbol: {symbol}")
            
            try:
                # Start experiment tracking
                self.experiment_tracker.start_run(f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Data pipeline
                symbol_results = self._process_symbol(symbol)
                results[symbol] = symbol_results
                
                # Log results
                self.experiment_tracker.log_metrics(symbol_results.get('best_metrics', {}))
                
                self.experiment_tracker.end_run()
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        # Generate portfolio-level results
        if len(symbols) > 1:
            results['portfolio'] = self._aggregate_portfolio_results(results)
        
        self.logger.info("Pipeline completed successfully")
        return results
    
    def _process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process a single symbol"""
        # 1. Data acquisition and validation
        raw_data = self.data_manager.get_data(symbol)
        validated_data = self.data_manager.validate_data(raw_data, symbol)
        
        # 2. Feature engineering
        features = self.feature_engineer.engineer_features(validated_data, symbol)
        scaler = self.feature_engineer.fit_scaler(features, symbol)
        scaled_features = self.feature_engineer.transform(features, symbol)
        
        # 3. Create sequences
        X, y = self.feature_engineer.create_sequences(scaled_features)
        assert len(X) == len(y), (f"create_sequences returned {len(X)} X rows and {len(y)} y rows")
        
        # 4. Train-validation-test split
        train_size = int(0.7 * len(X))
        val_size = int(0.2 * len(X))
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
        
        # 5. Model training and optimization
        model_results = {}
        best_model = None
        best_score = float('inf')
        
        for model_type in self.config.models.model_types:
            self.logger.info(f"Training {model_type} model for {symbol}")
            
            # Hyperparameter optimization
            if self.config.optimization.enabled:
                best_params = self.hyperparameter_optimizer.optimize_model(
                    model_type, X_train, y_train, X_val, y_val, self.config.models
                )
            else:
                best_params = {}
            
            # Train final model with best parameters
            model = ModelFactory.create_model(
                model_type, self.config.models, self.logger, 
                input_size=X_train.shape[-1] if len(X_train.shape) == 3 else X_train.shape[1]
            )
            
            model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred = model.predict(X_test).squeeze()
            if len(y_pred) != len(y_test):
                # common one-step-ahead shift; keep only the overlapping part
                min_len = min(len(y_pred), len(y_test))
                y_pred  = y_pred[:min_len]
                y_test  = y_test[:min_len]

            assert len(y_pred) == len(y_test), (f"After alignment: y_pred {len(y_pred)}, y_test {len(y_test)}")
            
            metrics = self.evaluator.evaluate_model(y_test, y_pred, f"{symbol}_{model_type}")
                
            # Create plots
            plot_files = self.evaluator.create_evaluation_plots(
                y_test, y_pred, f"{symbol}_{model_type}", 
                f"{self.config.system.output_dir}/plots"
            )
            
            # Generate explanations
            feature_names = [f"feature_{i}" for i in range(X_train.shape[-1] if len(X_train.shape) == 3 else X_train.shape[1])]
            explanations = self.model_explainer.explain_prediction(model, X_test, feature_names)
            
            model_results[model_type] = {
                'model': model,
                'metrics': metrics,
                'best_params': best_params,
                'plot_files': plot_files,
                'explanations': explanations,
                'predictions': y_pred
            }
            
            # Track best model
            if metrics['rmse'] < best_score:
                best_score = metrics['rmse']
                best_model = model
                best_model_type = model_type
        
        # 6. Ensemble modeling
        if self.config.ensemble.enabled and len(model_results) > 1:
            ensemble = EnsembleModel(self.config.ensemble, self.logger)
            
            for model_type, result in model_results.items():
                ensemble.add_model(model_type, result['model'])
            
            ensemble.fit(X_train, y_train, X_val, y_val)
            y_pred_ensemble = ensemble.predict(X_test)
            
            ensemble_metrics = self.evaluator.evaluate_model(y_test, y_pred_ensemble, f"{symbol}_ensemble")
            ensemble_plots = self.evaluator.create_evaluation_plots(
                y_test, y_pred_ensemble, f"{symbol}_ensemble", 
                f"{self.config.system.output_dir}/plots"
            )
            
            model_results['ensemble'] = {
                'model': ensemble,
                'metrics': ensemble_metrics,
                'plot_files': ensemble_plots,
                'predictions': y_pred_ensemble
            }
            
            # Update best model if ensemble is better
            if ensemble_metrics['rmse'] < best_score:
                best_model = ensemble
                best_model_type = 'ensemble'
                best_score = ensemble_metrics['rmse']
        
        # 7. Cross-validation
        if self.config.optimization.enabled:
            cv_results = self.cross_validator.cross_validate_model(best_model, X, y)
            model_results['cross_validation'] = cv_results
        
        # 8. Save best model
        best_model_path = f"{self.config.system.model_registry_dir}/{symbol}_{best_model_type}_best.pkl"
        try:
            joblib.dump(best_model, best_model_path)
            self.logger.info(f"Saved best model to {best_model_path}")
        except Exception as e:
            self.logger.warning(f"Could not save model: {str(e)}")
        
        return {
            'symbol': symbol,
            'models': model_results,
            'best_model_type': best_model_type,
            'best_metrics': model_results[best_model_type]['metrics'],
            'best_model_path': best_model_path,
            'data_shape': X.shape,
            'feature_names': feature_names
        }
    
    def _aggregate_portfolio_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate results across portfolio"""
        portfolio_metrics = {}
        
        # Collect metrics from all symbols
        all_metrics = []
        for symbol, result in results.items():
            if 'error' not in result:
                all_metrics.append(result['best_metrics'])
        
        if all_metrics:
            # Calculate portfolio-level metrics
            for metric in all_metrics[0].keys():
                values = [m[metric] for m in all_metrics if not np.isnan(m[metric])]
                if values:
                    portfolio_metrics[f'portfolio_mean_{metric}'] = np.mean(values)
                    portfolio_metrics[f'portfolio_std_{metric}'] = np.std(values)
        
        return portfolio_metrics
    
    def predict_realtime(self, symbol: str, model_path: str = None) -> Dict[str, Any]:
        """Make real-time predictions"""
        self.logger.info(f"Making real-time prediction for {symbol}")
        
        try:
            # Load model
            if model_path is None:
                model_path = f"{self.config.system.model_registry_dir}/{symbol}_best.pkl"
            
            model = joblib.load(model_path)
            
            # Get latest data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Update config for recent data
            self.config.data.start_date = start_date
            self.config.data.end_date = end_date

            # Process data
            raw_data = self.data_manager.get_data(symbol, use_cache=False)
            validated_data = self.data_manager.validate_data(raw_data, symbol)
            features = self.feature_engineer.engineer_features(validated_data, symbol)
            scaler = self.feature_engineer.fit_scaler(features, symbol)
            scaled_features = self.feature_engineer.transform(features, symbol)
            # Get latest sequence
            X, _ = self.feature_engineer.create_sequences(scaled_features)
            latest_X = X[-1:]  # Last sequence
            # Predict
            prediction = model.predict(latest_X)[0]
            # Determine the date for which the prediction is made
            # The predicted RV is for the row immediately after the last window
            predicted_date = features.index[-1] + pd.Timedelta(days=1)  # adjust if your interval is not 1min
            # Generate explanation
            feature_names = [f"feature_{i}" for i in range(latest_X.shape[-1] if len(latest_X.shape) == 3 else latest_X.shape[1])]
            explanations = self.model_explainer.explain_prediction(model, latest_X, feature_names)
            return {
                'symbol': symbol,
                'prediction': prediction,
                'predicted_date': str(predicted_date),
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'explanations': explanations,
                'confidence': self._calculate_prediction_confidence(model, latest_X)
            }
        except Exception as e:
            self.logger.error(f"Error in real-time prediction: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, model, X: np.ndarray) -> float:
        """
        Calculate prediction confidence (simple heuristic).
        For neural nets: use inverse of recent prediction variance.
        For tree/linear models: use std of predictions from last N windows.
        Returns a float between 0 and 1 (higher = more confident).
        """
        try:
            # Use last 10 windows if possible
            if X.shape[0] > 10:
                X_recent = X[-10:]
            else:
                X_recent = X

            preds = model.predict(X_recent)
            preds = np.array(preds).flatten()
            # Heuristic: lower std = higher confidence
            std = np.std(preds)
            # Map std to confidence: exp decay, clamp to [0, 1]
            confidence = float(np.exp(-std))
            confidence = max(0.0, min(1.0, confidence))
            return confidence
        except Exception as e:
            self.logger.warning(f"Could not calculate prediction confidence: {str(e)}")
            return 0.5  # fallback

# Register it
VolatilityForecastingPipeline.data_manager_sources_patch = (
    lambda dm: dm.sources.update({"local": LocalCSVSource(dm.logger)})
)

# Configuration Management
class ConfigManager:
    """Configuration management from YAML/JSON files"""
    
    @staticmethod
    def load_config(config_path: str) -> Config:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                #config_dict = yaml.safe_load(f)
                import io, yaml
                config_dict = yaml.safe_load(io.open("config.yaml", encoding="utf-8-sig")) ##CHANGE
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: Config, config_path: str):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_dict = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def _dict_to_config(config_dict: Dict) -> Config:
        """Convert dictionary to Config object"""
        # This is a simplified implementation
        config = Config()
        
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config
    
    @staticmethod
    def _config_to_dict(config: Config) -> Dict:
        """Convert Config object to dictionary"""
        # This is a simplified implementation
        return {
            'data': config.data.__dict__,
            'features': config.features.__dict__,
            'models': config.models.__dict__,
            'optimization': config.optimization.__dict__,
            'ensemble': config.ensemble.__dict__,
            'experiment': config.experiment.__dict__,
            'system': config.system.__dict__
        }

# CLI Interface
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line interface"""
    parser = argparse.ArgumentParser(description="Advanced Volatility Forecasting System")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    train_parser.add_argument('--symbols', nargs='+', help='Symbols to process')
    train_parser.add_argument('--models', nargs='+', help='Models to train')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    predict_parser.add_argument('--symbol', type=str, required=True, help='Symbol to predict')
    predict_parser.add_argument('--model-path', type=str, help='Path to trained model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    eval_parser.add_argument('--results-dir', type=str, help='Results directory')
    
    # Generate sample data command
    sample_parser = subparsers.add_parser('generate-sample', help='Generate sample data')
    sample_parser.add_argument('--symbol', type=str, default='SPY', help='Symbol')
    sample_parser.add_argument('--days', type=int, default=365, help='Number of days')
    sample_parser.add_argument('--output', type=str, default='sample_data.csv', help='Output file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--test-type', choices=['unit', 'integration', 'all'], default='all')
    
    return parser

def main():
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        # Load configuration
        if hasattr(args, 'config') and Path(args.config).exists():
            config = ConfigManager.load_config(args.config)
        else:
            config = Config()  # Use default config
        
        # Override config with command line arguments
        if hasattr(args, 'symbols') and args.symbols:
            config.data.symbols = args.symbols
        if hasattr(args, 'models') and args.models:
            config.models.model_types = args.models
        
        # Execute command
        if args.command == 'train':
            pipeline = VolatilityForecastingPipeline(config)
            results = pipeline.run_full_pipeline()
            print(f"Training completed. Results: {json.dumps(results, indent=2, default=str)}")
        
        elif args.command == 'predict':
            pipeline = VolatilityForecastingPipeline(config)
            result = pipeline.predict_realtime(args.symbol, args.model_path)
            print(f"Prediction result: {json.dumps(result, indent=2, default=str)}")
            
        elif args.command == 'evaluate':
            print("Evaluation functionality would be implemented here")

        
        elif args.command == 'generate-sample':
            generate_sample_data(args.symbol, args.days, args.output)
            print(f"Sample data generated: {args.output}")
        
        elif args.command == 'test':
            run_tests(args.test_type)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

def generate_sample_data(symbol: str, days: int, output_file: str):
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2020-01-01', periods=days * 24 * 60, freq='1min')  # 1-minute data
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    
    # Price simulation using geometric Brownian motion
    price = 100.0
    prices = []
    
    for i in range(len(dates)):
        # Add some volatility clustering and mean reversion
        volatility = 0.02 + 0.01 * np.random.random()
        drift = 0.0001 * (1 - (price - 100) / 100)  # Mean reversion to 100
        
        price_change = drift + volatility * np.random.normal()
        price *= (1 + price_change)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Generate OHLC from prices
    data = []
    for i in range(0, len(prices), 60):  # 1-hour bars from 1-minute prices
        if i + 60 > len(prices):
            break
        
        hour_prices = prices[i:i+60]
        open_price = hour_prices[0]
        high_price = hour_prices.max()
        low_price = hour_prices.min()
        close_price = hour_prices[-1]
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    # ------------------------------------------------------------------
    # convert the synthetic bars into a DataFrame, persist to disk, and
    # return the resulting object for immediate inspection / testing
    # ------------------------------------------------------------------
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    df.to_csv(output_file)
    return df
# ----------------------------------------------------------------------


# ------------------------------- TESTS --------------------------------
def run_tests(test_type: str = 'all'):
    """
    Very lightweight test-suite covering core utilities.
    – unit:   quick checks that do not hit external services
    – integration: end-to-end run on synthetic data
    """
    logger = logging.getLogger("tests")
    logger.setLevel(logging.INFO)
    success = True

    # ---------- UNIT ---------------------------------------------------
    if test_type in ('unit', 'all'):
        try:
            df = generate_sample_data('TEST', 1, 'test_sample.csv')
            assert not df.empty, "Sample DataFrame is empty"
            expected_cols = {'open', 'high', 'low', 'close', 'volume'}
            assert expected_cols.issubset(df.columns), "Missing OHLCV columns"
            logger.info("Unit tests passed ✅")
        except AssertionError as err:
            logger.error(f"Unit tests failed ❌ – {err}")
            success = False

    # ---------- INTEGRATION -------------------------------------------
    if test_type in ('integration', 'all') and success:
        try:
            # Build a minimal pipeline that consumes the synthetic file
            cfg = Config()
            cfg.data.symbols = ['TEST']
            cfg.data.interval = '1h'
            cfg.optimization.enabled = False          # keep it fast
            cfg.ensemble.enabled = False
            cfg.models.model_types = ['ridge']        # lightweight model

            # Override DataManager to load the synthetic CSV instead of
            # fetching from the internet.
            class LocalDataManager(DataManager):
                def get_data(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
                    return pd.read_csv('test_sample.csv', index_col='timestamp', parse_dates=True)

            pipeline = VolatilityForecastingPipeline(cfg)
            pipeline.data_manager = LocalDataManager(cfg.data, pipeline.logger)

            res = pipeline.run_full_pipeline()
            assert 'TEST' in res and 'error' not in res['TEST'], "Pipeline run failed"
            logger.info("Integration tests passed ✅")
        except Exception as err:
            logger.error(f"Integration tests failed ❌ – {err}")
            success = False

    print("All selected tests passed." if success else "Some tests failed; see logs.")


# ----------------------------- ENTRYPOINT -----------------------------
if __name__ == '__main__':
    main()



"""
 _models = {
        'lstm': LSTMModel,
        'transformer': TransformerModel,
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'catboost': CatBoostModel,
        'ridge': RidgeModel,
    }
python main.py train --config config.yaml --symbols SPY --models lstm ridge transformer
python main.py predict --config config.yaml --symbol SPY --model-path models/SPY_lstm_best.pkl

# Lightweight sanity checks
python main.py test --test-type unit

# Full synthetic integration test
python main.py test --test-type integration

python main.py <command> [OPTIONS]

Commands:
  train            --config FILE --symbols LIST --models LIST
  predict          --config FILE --symbol STR [--model-path FILE]
  evaluate         --config FILE --results-dir DIR              # placeholder
  generate-sample  --symbol STR --days INT --output FILE
  test             --test-type {unit|integration|all}
"""
