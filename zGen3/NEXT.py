#!/usr/bin/env python3
"""
Professional Realized Volatility Forecasting System
===================================================

A comprehensive deep learning system for predicting next-day realized volatility
from high-frequency OHLCV data. Designed for institutional-grade quantitative analysis.

Author: Quantitative Research Team
Version: 1.0.0
"""

import os
import sys
import json
import warnings
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# ==================== CONFIGURATION ====================
INTERVAL = "5min"

class Config:
    """Configuration parameters for the RV forecasting system."""
    
    # Data parameters
    TRADING_MINUTES_PER_DAY = 390  # NYSE trading session
    WINDOW_SIZE = 390              # 1-day history for RV computation
    LOOKBACK_MINUTES = 390     # Input sequence length
    FORECAST_HORIZON = 390         # Predict next full trading day RV
    
    # Model parameters
    HIDDEN_DIM = 128 #LSTM 128 TF 32
    LSTM_LAYERS = 2 
    DROPOUT_RATE = 0.2 #LSTM 0.2 TF 0.1
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 10
    
    # Evaluation parameters
    TEST_SPLIT = 0.1
    VALIDATION_SPLIT = 0.2
    
    # Feature engineering
    VWAP_WINDOW = 10
    MOMENTUM_WINDOW = 5
    RV_WINDOWS = [5, 15, 30, 60]
    
    # Paths
    MODEL_DIR = Path("models")
    LOG_DIR = Path("logs")
    PLOT_DIR = Path("plots")
    
    def __init__(self):
        # Create directories
        for dir_path in [self.MODEL_DIR, self.LOG_DIR, self.PLOT_DIR]:
            dir_path.mkdir(exist_ok=True)

# ==================== LOGGING SETUP ====================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup professional logging configuration."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Config.LOG_DIR / f"rv_forecasting_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("RVForecasting")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# ==================== DATA PREPROCESSING ====================

class DataPreprocessor:
    """Professional data preprocessing pipeline for OHLCV data."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = []
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate OHLCV data."""
        
        self.logger.info(f"Loading data from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(df)} rows")
            # Validate required columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df = downsample_prices(df, INTERVAL)
            self.logger.info("Downsampled data to interval: " , INTERVAL, "with shape: ", df.shape)
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Add trading day
            df['trading_day'] = df['datetime'].dt.date
            
            self.logger.info(f"Data spans from {df['datetime'].min()} to {df['datetime'].max()}")
            self.logger.info(f"Number of trading days: {df['trading_day'].nunique()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def compute_returns_and_rv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns and realized volatility targets."""
        
        self.logger.info("Computing log returns and RV targets...")
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Next-day RV target (forward-looking)
        df['RV_target'] = (
            df['log_return']
            .rolling(window=self.config.FORECAST_HORIZON, min_periods=self.config.FORECAST_HORIZON)
            .apply(lambda x: np.sqrt((x**2).sum()), raw=True)
            .shift(-self.config.FORECAST_HORIZON)
        )
        
        # Remove NaN values
        initial_len = len(df)
        #removed_rows = df[df['log_return'].isna()]
        df = df.dropna(subset=['log_return']).copy()
        self.logger.info(f"Removed {initial_len - len(df)} rows with NaN log returns")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive feature engineering for financial time series."""
        
        self.logger.info("Engineering features...")
        
        # Basic features
        df['log_volume'] = np.log(df['volume'] + 1)
        df['hl_range'] = df['high'] - df['low']
        df['oc_range'] = abs(df['close'] - df['open'])
        
        # VWAP and deviation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (
            (df['typical_price'] * df['volume'])
            .rolling(self.config.VWAP_WINDOW)
            .sum() / df['volume'].rolling(self.config.VWAP_WINDOW).sum()
        )
        df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)
        
        # Momentum features
        df['momentum'] = df['log_return'].rolling(self.config.MOMENTUM_WINDOW).sum()
        df['price_momentum'] = (df['close'] / df['close'].shift(self.config.MOMENTUM_WINDOW) - 1)
        
        # Rolling realized volatilities
        for window in self.config.RV_WINDOWS:
            col_name = f'rv_{window}min'
            df[col_name] = (
                df['log_return']
                .rolling(window=window, min_periods=window//2)
                .apply(lambda x: np.sqrt((x**2).sum()), raw=True)
            )
        
        # Time-based features
        df['minute_of_day'] = (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute - 570) % 390
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 390)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 390)
        
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        
        # Volatility regime indicators
        df['rv_regime'] = pd.qcut(df['rv_60min'].fillna(df['rv_60min'].median()), 
                                  q=3, labels=['low', 'medium', 'high'])
        df['rv_regime_low'] = (df['rv_regime'] == 'low').astype(int)
        df['rv_regime_high'] = (df['rv_regime'] == 'high').astype(int)
        
        # Market microstructure features
        df['bid_ask_proxy'] = df['hl_range'] / (df['close'] + 1e-8)
        df['volume_price_trend'] = df['volume'] * np.sign(df['log_return'])
        
        self.logger.info(f"Generated {len([col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'trading_day']])} features")
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare sequences for deep learning models."""
        
        self.logger.info("Preparing sequences for training...")
        
        # Select feature columns
        feature_cols = [
            'log_return', 'log_volume', 'hl_range', 'oc_range',
            'vwap_deviation', 'momentum', 'price_momentum',
            'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos',
            'rv_regime_low', 'rv_regime_high', 'bid_ask_proxy', 'volume_price_trend'
        ]
        
        # Add rolling RV features
        feature_cols.extend([f'rv_{w}min' for w in self.config.RV_WINDOWS])
        
        # Filter available columns

        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_names = feature_cols
        
        # Remove rows with NaN targets

        #df.to_csv("debug_output.csv", index=False)
        if 'RV_target' in df.columns and df['RV_target'].notna().any():
            df_clean = df.dropna(subset=['RV_target']).copy()
        else:
            # Fallback: keep entire df or handle as needed
            self.logger.info("No valid RV_target found, fallback deployed") ##
            df_clean = df.copy()
        df_clean = df_clean.reset_index(drop=True)
        self.logger.info(f"Using {len(df_clean)} samples with valid targets")
        
        # Prepare feature matrix
        feature_data = df_clean[feature_cols].fillna(method='ffill').fillna(0)
        
        # Normalize features
        feature_data_normalized = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        sequences = []
        targets = []
        if 'RV_target' in df.columns and df['RV_target'].notna().any():
            for i in range(self.config.LOOKBACK_MINUTES, len(feature_data_normalized)):
                if not pd.isna(df_clean.iloc[i]['RV_target']):
                    seq = feature_data_normalized[i-self.config.LOOKBACK_MINUTES:i]
                    target = df_clean.iloc[i]['RV_target']

                    sequences.append(seq)
                    targets.append(target)
        else:
            self.logger.warning("No valid RV_target found, using entire feature set without targets")
            for i in range(self.config.LOOKBACK_MINUTES, len(feature_data_normalized)):
                seq = feature_data_normalized[i-self.config.LOOKBACK_MINUTES:i]
                sequences.append(seq)
                targets.append(0.0)
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        self.logger.info(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        
        return sequences, targets, feature_cols

# ==================== NEURAL NETWORK MODELS ====================

class RVDataset(Dataset):
    """PyTorch dataset for realized volatility prediction."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """LSTM-based model for RV forecasting."""
    
    def __init__(self, input_size: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the last time step
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        return self.fc(output).squeeze()

class TransformerModel(nn.Module):
    """Transformer-based model for RV forecasting."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        return self.fc(pooled).squeeze()

# ==================== MODEL TRAINER ====================

class RVModelTrainer:
    """Professional model training and evaluation."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> Dict:
        """Train model with early stopping and learning rate scheduling."""
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        
        self.logger.info(f"Starting training for {model_name}...")
        
        for epoch in range(self.config.EPOCHS):
            # Training
            self.logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)

            if (epoch + 1) % 1 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS} - "
                               f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          self.config.MODEL_DIR / f"best_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

# ==================== EVALUATION METRICS ====================

class RVEvaluator:
    """Comprehensive evaluation metrics for RV forecasting."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def qlike_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """QLIKE loss function commonly used in volatility forecasting."""
        return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
    
    def directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prev: np.ndarray) -> float:
        """Directional accuracy: predict if volatility increases or decreases."""
        true_direction = (y_true > y_prev).astype(int)
        pred_direction = (y_pred > y_prev).astype(int)
        return np.mean(true_direction == pred_direction)
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prev: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive evaluation metrics."""
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'qlike': self.qlike_loss(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        if y_prev is not None:
            metrics['directional_accuracy'] = self.directional_accuracy(y_true, y_pred, y_prev)
        
        return metrics
    
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    model_name: str, save_path: Optional[Path] = None):
        """Create comprehensive evaluation plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual RV')
        axes[0, 0].set_ylabel('Predicted RV')
        axes[0, 0].set_title(f'{model_name}: Predicted vs Actual')
        
        # Residuals
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_true, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual RV')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Time series comparison
        indices = range(min(len(y_true), 100))  # Show last 100 predictions
        axes[1, 0].plot(indices, y_true[:len(indices)], label='Actual', linewidth=2)
        axes[1, 0].plot(indices, y_pred[:len(indices)], label='Predicted', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('RV')
        axes[1, 0].set_title('Time Series Comparison (Last 100 points)')
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved evaluation plot to {save_path}")
        
        plt.show()

# ==================== MAIN SYSTEM CLASS ====================

class RVForecastingSystem:
    """Main system orchestrating the entire RV forecasting pipeline."""
    
    def __init__(self, csv_path: str, log_level: str = "INFO"):
        self.config = Config()
        self.logger = setup_logging(log_level)
        self.csv_path = csv_path
        
        self.preprocessor = DataPreprocessor(self.config, self.logger)
        self.trainer = RVModelTrainer(self.config, self.logger)
        self.evaluator = RVEvaluator(self.logger)
        self.models = {}
        self.results = {}
    
    def run_full_pipeline(self, model_types: List[str] = ["lstm", "transformer"]):
        """Execute the complete forecasting pipeline."""
        
        try:
            self.logger.info("=== STARTING RV FORECASTING PIPELINE ===")
            
            # Load and preprocess data

            df = self.preprocessor.load_data(self.csv_path)
            df = self.preprocessor.compute_returns_and_rv(df)
            df = self.preprocessor.engineer_features(df)
            
            # Prepare sequences
            X, y, feature_names = self.preprocessor.prepare_sequences(df)
            
            # Train-test split (time series aware)
            split_idx = int(len(X) * (1 - self.config.TEST_SPLIT))
            val_split_idx = int(split_idx * (1 - self.config.VALIDATION_SPLIT))
            
            X_train, X_val, X_test = X[:val_split_idx], X[val_split_idx:split_idx], X[split_idx:]
            y_train, y_val, y_test = y[:val_split_idx], y[val_split_idx:split_idx], y[split_idx:]
            
            self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Create data loaders
            train_dataset = RVDataset(X_train, y_train)
            val_dataset = RVDataset(X_val, y_val)
            test_dataset = RVDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            
            # Train models
            input_size = X.shape[2]
            
            for model_type in model_types:
                self.logger.info(f"Training {model_type.upper()} model...")
                
                if model_type == "lstm":
                    model = LSTMModel(
                        input_size=input_size,
                        hidden_dim=self.config.HIDDEN_DIM,
                        num_layers=self.config.LSTM_LAYERS,
                        dropout=self.config.DROPOUT_RATE
                    )
                elif model_type == "transformer":
                    model = TransformerModel(
                        input_size=input_size,
                        d_model=self.config.HIDDEN_DIM,
                        dropout=self.config.DROPOUT_RATE
                    )
                else:
                    self.logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Train model
                training_history = self.trainer.train_model(model, train_loader, val_loader, model_type)
                
                # Evaluate model
                model.eval()
                predictions = []
                actuals = []
                
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(self.trainer.device)
                        pred = model(batch_x).cpu().numpy()
                        predictions.extend(pred)
                        actuals.extend(batch_y.numpy())
                
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                # Calculate metrics
                metrics = self.evaluator.evaluate_model(actuals, predictions)
                
                # Store results
                self.models[model_type] = model
                self.results[model_type] = {
                    'metrics': metrics,
                    'training_history': training_history,
                    'predictions': predictions,
                    'actuals': actuals
                }
                
                # Log results
                self.logger.info(f"{model_type.upper()} Results:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.6f}")
                
                # Create plots
                plot_path = self.config.PLOT_DIR / f"{model_type}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.evaluator.plot_results(actuals, predictions, model_type.upper(), plot_path)
            
            # Save results
            self.save_results()
            
            self.logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def save_results(self):
        """Save comprehensive results to JSON."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.LOG_DIR / "results_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'metrics': result['metrics'],
                'training_history': {
                    'train_losses': result['training_history']['train_losses'],
                    'val_losses': result['training_history']['val_losses'],
                    'best_val_loss': result['training_history']['best_val_loss']
                }
            }
        def to_serializable(val):
            if isinstance(val, dict):
                return {k: to_serializable(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [to_serializable(v) for v in val]
            elif isinstance(val, np.generic):
                return val.item()
            else:
                return val

        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'metrics': to_serializable(result['metrics']),
                'training_history': {
                    'train_losses': to_serializable(result['training_history']['train_losses']),
                    'val_losses': to_serializable(result['training_history']['val_losses']),
                    'best_val_loss': to_serializable(result['training_history']['best_val_loss']),
                },
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def predict_next_day_rv(self, model_type: str = "lstm") -> float:
        """Real-time inference for next-day RV prediction."""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        model = self.models[model_type]
        model.eval()

        # Load and preprocess the latest data
        df = self.preprocessor.load_data(self.csv_path)
        df = self.preprocessor.compute_returns_and_rv(df)
        df = self.preprocessor.engineer_features(df)
        df = df.drop(columns=['RV_target'])
        X, y, feature_names = self.preprocessor.prepare_sequences(df)

        if len(X) == 0:
            self.logger.error("No valid sequences available for prediction.")
            raise ValueError("No valid sequences available for prediction.")

        # Use the most recent sequence for prediction
        latest_sequence = torch.FloatTensor(X[-1]).unsqueeze(0).to(self.trainer.device)  # shape: (1, seq_len, features)

        with torch.no_grad():
            pred = model(latest_sequence)
            prediction = float(pred.cpu().numpy().item())
            annualized_prediction = prediction * (252 ** 0.5)
        self.logger.info(f"Next-day RV prediction: {prediction:.6f}")
        self.logger.info(f"Annualized prediction: {annualized_prediction:.6f}")
        return prediction

# ==================== CLI INTERFACE ====================
def app_log(msg):
    print(f"APPLOG: {msg}")

def main():
    """Command-line interface for the RV Forecasting System."""
    
    parser = argparse.ArgumentParser(
        description="Professional Realized Volatility Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train models and evaluate
    python NEXT.py train --csv-path data/minute_data.csv --models lstm transformer
    
    # Make prediction with trained model
    python NEXT.py predict --csv-path data/minute_data.csv --model lstm
    
    # Full pipeline with custom configuration
    python NEXT.py full --csv-path data/minute_data.csv --log-level DEBUG
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train RV forecasting models')
    train_parser.add_argument('--csv-path', required=True, help='Path to OHLCV CSV file')
    train_parser.add_argument('--models', nargs='+', default=['lstm'], 
                             choices=['lstm', 'transformer'], 
                             help='Models to train')
    train_parser.add_argument('--log-level', default='INFO', 
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate RV predictions')
    predict_parser.add_argument('--csv-path', required=True, help='Path to OHLCV CSV file')
    predict_parser.add_argument('--model', default='lstm', choices=['lstm', 'transformer'],
                               help='Model to use for prediction')
    predict_parser.add_argument('--log-level', default='INFO',
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--csv-path', required=True, help='Path to OHLCV CSV file')
    full_parser.add_argument('--models', nargs='+', default=['lstm', 'transformer'],
                            choices=['lstm', 'transformer'],
                            help='Models to train and evaluate')
    full_parser.add_argument('--log-level', default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--csv-path', required=True, help='Path to OHLCV CSV file')
    eval_parser.add_argument('--model-dir', help='Directory containing trained models')
    eval_parser.add_argument('--log-level', default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Validate CSV path
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
        
        # Initialize system
        system = RVForecastingSystem(str(csv_path), args.log_level)
        
        if args.command == 'train':
            system.logger.info("Starting training pipeline...")
            system.run_full_pipeline(args.models)
            
        elif args.command == 'predict':
            system.logger.info("Starting prediction pipeline...")

            # Check for existing model file in models/ directory
            model_filename = f"best_{args.model}_model.pt"
            model_path = system.config.MODEL_DIR / model_filename
            if model_path.exists():
                system.logger.info(f"Found pre-trained model: {model_path}")
                input_size = 15 + len(system.config.RV_WINDOWS)  # match feature count in prepare_sequences
                if args.model == "lstm":
                    model = LSTMModel(
                        input_size=input_size,
                        hidden_dim=system.config.HIDDEN_DIM,
                        num_layers=system.config.LSTM_LAYERS,
                        dropout=system.config.DROPOUT_RATE
                    )
                elif args.model == "transformer":
                    model = TransformerModel(
                        input_size=input_size,
                        d_model=system.config.HIDDEN_DIM,
                        dropout=system.config.DROPOUT_RATE
                    )
                else:
                    raise ValueError(f"Unknown model type: {args.model}")

                model.load_state_dict(torch.load(model_path, map_location=system.trainer.device))
                model = model.to(system.trainer.device)
                model.eval()
                system.models[args.model] = model
            else:
                system.logger.info(f"No pre-trained model found at {model_path}, training a new one...")
                system.run_full_pipeline([args.model])

            # Then make prediction
            prediction = system.predict_next_day_rv(args.model)
            annualized_prediction = prediction * (2520000 ** 0.5) ##RR
            print(f"\n{'='*50}")
            print(f"NEXT-DAY REALIZED VOLATILITY PREDICTION")
            print(f"{'='*50}")
            print(f"Model: {args.model.upper()}")
            print(f"Prediction: {prediction * 100:.6f} %")
            app_log(f"Annualized Volatility (%): {annualized_prediction:.6f}")
            print(f"Annualized Prediction: {annualized_prediction} %")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
        elif args.command == 'full':
            system.logger.info("Starting full pipeline...")
            system.run_full_pipeline(args.models)
            
            # Display summary results
            print(f"\n{'='*60}")
            print(f"REALIZED VOLATILITY FORECASTING RESULTS")
            print(f"{'='*60}")
            
            for model_name, result in system.results.items():
                print(f"\n{model_name.upper()} Model Performance:")
                print(f"{'-'*30}")
                metrics = result['metrics']
                print(f"RMSE:                 {metrics['rmse']:.6f}")
                print(f"MAE:                  {metrics['mae']:.6f}")
                print(f"R²:                   {metrics['r2']:.6f}")
                print(f"QLIKE:                {metrics['qlike']:.6f}")
                print(f"MAPE:                 {metrics['mape']:.2f}%")
                if 'directional_accuracy' in metrics:
                    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
            
            print(f"\n{'='*60}")
            
        elif args.command == 'evaluate':
            system.logger.info("Starting evaluation pipeline...")
            # Implementation for loading and evaluating pre-trained models
            system.logger.warning("Evaluation of pre-trained models not yet implemented")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


# ==================== ADDITIONAL UTILITIES ====================

class RVBacktester:
    """Backtesting framework for RV forecasting strategies."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def simple_volatility_strategy(self, predictions: np.ndarray, 
                                  actuals: np.ndarray, 
                                  returns: np.ndarray) -> Dict:
        """Simple volatility-based trading strategy."""
        
        # Strategy: Go long when predicted volatility is below median
        # (volatility mean reversion)
        median_vol = np.median(predictions)
        
        positions = np.where(predictions < median_vol, 1, -1)  # 1 for long, -1 for short
        strategy_returns = positions[:-1] * returns[1:]  # Offset for next-day returns
        
        # Calculate strategy metrics
        total_return = np.sum(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(np.cumsum(strategy_returns))
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(strategy_returns > 0),
            'volatility': np.std(strategy_returns) * np.sqrt(252)
        }
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)
        return np.min(drawdown)


class RVMonitor:
    """Real-time monitoring and alerting system."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.thresholds = {
            'high_volatility': 0.05,    # 5% daily volatility
            'low_volatility': 0.005,    # 0.5% daily volatility
            'prediction_error': 0.02    # 2% prediction error
        }
    
    def check_volatility_regime(self, rv_prediction: float) -> Dict:
        """Monitor volatility regime and generate alerts."""
        
        alerts = []
        
        if rv_prediction > self.thresholds['high_volatility']:
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'message': f"High volatility predicted: {rv_prediction:.4f}",
                'severity': 'WARNING'
            })
        
        if rv_prediction < self.thresholds['low_volatility']:
            alerts.append({
                'type': 'LOW_VOLATILITY',
                'message': f"Low volatility predicted: {rv_prediction:.4f}",
                'severity': 'INFO'
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'rv_prediction': rv_prediction,
            'alerts': alerts
        }
    
    def generate_daily_report(self, system: 'RVForecastingSystem') -> str:
        """Generate daily performance report."""
        
        report = f"""
DAILY REALIZED VOLATILITY REPORT
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

MODEL PERFORMANCE SUMMARY:
"""
        
        for model_name, result in system.results.items():
            metrics = result['metrics']
            report += f"""
{model_name.upper()} Model:
  RMSE:     {metrics['rmse']:.6f}
  R²:       {metrics['r2']:.4f}
  QLIKE:    {metrics['qlike']:.6f}
"""
        
        report += f"""
{'='*50}
Report generated by RV Forecasting System v1.0.0
"""
        
        return report


# ==================== CONFIGURATION VALIDATION ====================

def validate_data_format(csv_path: str) -> bool:
    """Validate that the CSV file has the required format."""
    
    try:
        df = pd.read_csv(csv_path, nrows=10)  # Read first 10 rows for validation
        
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Required columns: {required_columns}")
            print(f"Found columns: {list(df.columns)}")
            return False
        
        # Validate datetime format
        pd.to_datetime(df['datetime'].iloc[0])
        
        # Validate numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Error: Column '{col}' is not numeric")
                return False
        
        print("✓ Data format validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating data format: {str(e)}")
        return False

def downsample_prices(df_prices: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Downsample 1-minute OHLCV data to a specified interval.

    Args:
        df_prices (pd.DataFrame): DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']
        interval (str): Pandas offset alias for resampling (e.g., '5min', '15min', '1H')

    Returns:
        pd.DataFrame: Resampled OHLCV DataFrame
    """
    df = df_prices.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled = df.resample(interval).agg(ohlcv_dict).dropna()

    resampled.reset_index(inplace=True)
    return resampled

def create_sample_data(output_path: str, days: int = 30) -> None:
    """Create sample OHLCV data for testing."""
    
    print(f"Creating sample data with {days} trading days...")
    
    # Generate sample data
    np.random.seed(42)
    
    # Trading hours: 9:30 AM to 4:00 PM EST (390 minutes)
    start_time = pd.Timestamp('2024-01-01 09:30:00')
    
    data = []
    current_price = 100.0
    
    for day in range(days):
        day_start = start_time + pd.Timedelta(days=day)
        
        # Skip weekends
        if day_start.weekday() >= 5:
            continue
        
        for minute in range(390):  # 390 minutes in trading day
            timestamp = day_start + pd.Timedelta(minutes=minute)
            
            # Realistic price movement
            return_vol = 0.02 + 0.01 * np.random.randn()  # Base volatility with noise
            price_change = np.random.normal(0, return_vol) * current_price
            
            new_price = current_price + price_change
            
            # Generate OHLCV
            high = new_price + abs(np.random.normal(0, 0.001)) * current_price
            low = new_price - abs(np.random.normal(0, 0.001)) * current_price
            open_price = current_price
            close_price = new_price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'datetime': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            current_price = new_price
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample data created: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")


# ==================== PERFORMANCE PROFILER ====================

class PerformanceProfiler:
    """Profile system performance and resource usage."""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = datetime.now()
        
    def stop_profiling(self) -> Dict:
        """Stop profiling and return metrics."""
        if self.start_time is None:
            return {}
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            'duration_seconds': duration,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat()
        }


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Add development utilities
    if len(sys.argv) > 1 and sys.argv[1] == "create-sample":
        if len(sys.argv) < 3:
            print("Usage: python NEXT.py create-sample <output_path> [days]")
            sys.exit(1)
        
        output_path = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        
        create_sample_data(output_path, days)
        
        print(f"\nSample data created! You can now run:")
        print(f"python NEXT.py full --csv-path {output_path}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        if len(sys.argv) < 3:
            print("Usage: python NEXT.py validate <csv_path>")
            sys.exit(1)
        
        csv_path = sys.argv[2]
        if validate_data_format(csv_path):
            print("✓ Data validation successful!")
        else:
            print("✗ Data validation failed!")
            sys.exit(1)
    
    else:
        main()


# ==================== VERSION INFO ====================

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"
__description__ = "Professional Realized Volatility Forecasting System"

"""
SYSTEM REQUIREMENTS:
- Python 3.8+
- PyTorch 1.9+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- At least 4GB RAM for training
- GPU recommended for large datasets

USAGE EXAMPLES:

1. Create sample data:
   python zGen3\\NEXT.py create-sample sample_data.csv 60

2. Validate data format:
   python zGen3\\NEXT.py validate data\\SPY.csv

3. Run full pipeline:
   python zGen3\\NEXT.py full --csv-path data\\SPY.csv

4. Train specific models:
   python zGen3\\NEXT.py train --csv-path data\\SPY.csv --models transformer

5. Generate predictions:
   python zGen3\\NEXT.py predict --csv-path data\\SPY.csv --model lstm

FEATURES:
- Professional-grade data preprocessing
- Multiple neural network architectures (LSTM, Transformer)
- Comprehensive evaluation metrics
- Real-time prediction capabilities
- Backtesting framework
- Monitoring and alerting
- Extensive logging and error handling
- Time series cross-validation
- Feature engineering for financial data
- Robust handling of missing data
- Performance profiling
- CLI interface with multiple commands
"""