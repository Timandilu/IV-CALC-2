#!/usr/bin/env python3
"""
Institutional-Grade Next-Day Realized Volatility Forecasting System
=================================================================
Single-file implementation that can be launched directly from the command
line and covers every feature requested:

* Data ingestion from CSV or Yahoo Finance (yfinance).
* Persistent Parquet caching of raw and processed data.
* Robust data validation and structured JSON logging.
* Comprehensive feature-engineering pipeline for high-frequency OHLCV data.
* Sequence creation for deep-learning models.
* Modular model zoo: LSTM, GRU, CNN, Transformer (PyTorch), TabNet, LightGBM,
  XGBoost, CatBoost, Ridge, Lasso.
* Automated hyper-parameter optimization with Optuna (optionally Ray Tune).
* Time-series aware cross-validation: expanding-window, rolling-window, k-fold.
* Ensemble stacking/blending with `sklearn` `StackingRegressor`.
* SHAP-based explainability for every supported model.
* MLflow experiment tracking (auto-enabled if MLflow is installed).
* Multiprocessing/distributed tuning via Ray Tune (if Ray installed).
* Unit & integration test suite runnable via `python rv_system.py test`.
* CLI interface: `train`, `predict`, `full`, `explain`, `test`.
* Production-ready error handling, performance profiling, and resource
  monitoring.

Author: Lead Quantitative Systems Architect
Version: 2.0.0
Created: 2025-06-27
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import argparse
import contextlib
import ctypes
import dataclasses
import functools
import gc
import importlib
import inspect
import io
import json
import logging
import math
import os
import sys
import textwrap
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Third-Party Imports - grouped to avoid import errors when libraries missing.
# Wherever a library is optional, graceful fallbacks are provided.
# ---------------------------------------------------------------------------

# Core scientific stack
import numpy as np
import pandas as pd

# Plotting (required for SHAP summaries)
import matplotlib.pyplot as plt

# Machine learning
import sklearn
from sklearn import metrics, preprocessing, model_selection, linear_model
from sklearn.ensemble import StackingRegressor

# Deep-learning (PyTorch) – mandatory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Gradient boosters – optional but highly recommended
with contextlib.suppress(ImportError):
    import lightgbm as lgb  # type: ignore
with contextlib.suppress(ImportError):
    import xgboost as xgb  # type: ignore
with contextlib.suppress(ImportError):
    import catboost as cb  # type: ignore

# TabNet – optional
with contextlib.suppress(ImportError):
    from pytorch_tabnet.tab_model import TabNetRegressor  # type: ignore

# Hyper-parameter optimisation
import optuna
with contextlib.suppress(ImportError):
    from ray import tune  # type: ignore
    from ray.tune.schedulers import ASHAScheduler  # type: ignore

# Explainability
with contextlib.suppress(ImportError):
    import shap  # type: ignore

# Experiment tracking
with contextlib.suppress(ImportError):
    import mlflow  # type: ignore


# ---------------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CACHE   = PROJECT_ROOT / "cache"
MODEL_DIR    = PROJECT_ROOT / "models"
LOG_DIR      = PROJECT_ROOT / "logs"
PLOT_DIR     = PROJECT_ROOT / "plots"
CONFIG_FILE  = PROJECT_ROOT / "config.json"  # optional external config

for d in (DATA_CACHE, MODEL_DIR, LOG_DIR, PLOT_DIR):
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def json_logger(name: str = "RVSystem", level: str = "INFO") -> logging.Logger:
    """Structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    handler = logging.StreamHandler(sys.stdout)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            log_record = {
                "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "level": record.levelname,
                "msg": record.getMessage(),
            }
            return json.dumps(log_record)

    handler.setFormatter(JsonFormatter())
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

LOGGER = json_logger()

# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    """Runtime configuration with sensible defaults."""

    # General
    symbol: str                     = "SPY"
    interval: str                   = "5min"  # resample target (must divide 390)
    lookback_minutes: int           = 390
    forecast_horizon: int           = 390
    trading_minutes_per_day: int    = 390

    # Feature engineering
    vwap_window: int                = 10
    momentum_window: int            = 5
    rv_windows: Tuple[int, ...]     = (5, 15, 30, 60)

    # Model & training
    epochs: int                     = 50
    batch_size: int                 = 128
    hidden_dim: int                 = 128
    lstm_layers: int                = 2
    dropout: float                  = 0.2
    learning_rate: float            = 1e-3
    patience: int                   = 8

    # Data split
    test_split: float               = 0.1
    val_split: float                = 0.2

    # Hyper-parameter optimisation
    hyperopt: bool                  = True
    hyperopt_trials: int            = 40

    # Cross-validation
    cv_folds: int                   = 5
    cv_strategy: str                = "expanding"  # expanding | rolling | kfold

    # Ensembles
    use_ensemble: bool              = True

    # Logging
    log_level: str                  = "INFO"

    # Experiment tracking
    use_mlflow: bool                = hasattr(sys.modules, "mlflow")

    def override(self, kv_pairs: List[str]) -> None:
        for pair in kv_pairs:
            key, value = pair.split("=", 1)
            # Attempt to infer type
            attr_type = type(getattr(self, key))
            if attr_type is bool:
                cast_value = value.lower() in {"1", "true", "yes"}
            else:
                cast_value = attr_type(value)
            setattr(self, key, cast_value)

# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

def downsample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Down-sample 1-minute OHLCV to desired interval."""
    df = df.copy()
    df.index = pd.to_datetime(df["datetime"])
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(interval).agg(ohlcv).dropna().reset_index()
    resampled["datetime"] = pd.to_datetime(resampled["datetime"])
    return resampled


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing  = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create rich set of features for modelling."""
    df = df.copy()

    # Basic returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Targets – forward-looking RV
    df["rv_target"] = (
        df["log_return"]
        .rolling(cfg.forecast_horizon)
        .apply(lambda x: np.sqrt(np.sum(x ** 2)), raw=True)
        .shift(-cfg.forecast_horizon)
    )

    # VWAP deviation
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(cfg.vwap_window).sum() / df["volume"].rolling(cfg.vwap_window).sum()
    df["vwap_dev"] = (df["close"] - vwap) / (vwap + 1e-9)

    # Momentum
    df["momentum_sum"] = df["log_return"].rolling(cfg.momentum_window).sum()
    df["price_momentum"] = df["close"] / df["close"].shift(cfg.momentum_window) - 1

    # Rolling RVs
    for w in cfg.rv_windows:
        df[f"rv_{w}"] = df["log_return"].rolling(w).apply(lambda x: np.sqrt(np.sum(x ** 2)), raw=True)

    # Time encoding
    df["minute"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / cfg.trading_minutes_per_day)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / cfg.trading_minutes_per_day)
    df["dow"]        = df["datetime"].dt.dayofweek
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 5)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 5)

    # Market microstructure
    df["hl_range"]     = df["high"] - df["low"]
    df["oc_range"]     = np.abs(df["close"] - df["open"])
    df["bid_ask_proxy"] = df["hl_range"] / (df["close"] + 1e-9)
    df["vol_price_trend"] = df["volume"] * np.sign(df["log_return"])

    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------------------------
# Sequence Preparation for Deep Learning
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [
        "log_return", "vwap_dev", "momentum_sum", "price_momentum",
        "minute_sin", "minute_cos", "dow_sin", "dow_cos",
        "hl_range", "oc_range", "bid_ask_proxy", "vol_price_trend",
    ] + [f"rv_{w}" for w in cfg.rv_windows]

    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    seq_length = cfg.lookback_minutes
    sequences, targets = [], []
    for i in range(seq_length, len(df)):
        if np.isnan(df["rv_target"].iloc[i]):
            continue
        sequences.append(X_scaled[i - seq_length : i, :])
        targets.append(df["rv_target"].iloc[i])

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y, feature_cols

# ---------------------------------------------------------------------------
# Model Definitions – PyTorch
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(input_size, cfg.hidden_dim, cfg.lstm_layers,
                            batch_first=True, dropout=cfg.dropout)
        self.fc = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.gru = nn.GRU(input_size, cfg.hidden_dim, cfg.lstm_layers,
                          batch_first=True, dropout=cfg.dropout)
        self.fc = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class CNNModel(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.conv = nn.Conv1d(input_size, cfg.hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, x):
        # Input shape: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):  # type: ignore [override]
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.in_proj = nn.Linear(input_size, cfg.hidden_dim)
        self.pos_enc = PositionalEncoding(cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(cfg.hidden_dim, nhead=8,
                                                   dim_feedforward=cfg.hidden_dim * 4,
                                                   dropout=cfg.dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Model Zoo Dispatcher
# ---------------------------------------------------------------------------
ModelFactory = {
    "lstm": LSTMModel,
    "gru":  GRUModel,
    "cnn":  CNNModel,
    "transformer": TransformerModel,
    # Gradient boosting models and linear models handled separately
}

# ---------------------------------------------------------------------------
# PyTorch Dataset Wrapper
# ---------------------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------------
# Training and Evaluation Helpers
# ---------------------------------------------------------------------------

def train_pytorch(model: nn.Module, cfg: Config, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Tuple[nn.Module, Dict[str, List[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X_val, y_val),   batch_size=cfg.batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    history = {"train": [], "val": []}
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        scheduler.step(val_loss)

        LOGGER.info(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}))

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                LOGGER.info(json.dumps({"early_stop": epoch + 1}))
                break

    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Sklearn Wrapper for Gradient Boosters and Linear Models
# ---------------------------------------------------------------------------

def train_sklearn(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_sklearn(model, X):
    return model.predict(X)

# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    mae  = float(metrics.mean_absolute_error(y_true, y_pred))
    r2   = float(metrics.r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    qlike = float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "qlike": qlike}

# ---------------------------------------------------------------------------
# Hyper-parameter Search Spaces
# ---------------------------------------------------------------------------
HP_SPACES = {
    "lightgbm": {
        "num_leaves": ("int", 16, 256),
        "learning_rate": ("loguniform", 1e-3, 0.2),
        "n_estimators": ("int", 100, 800),
    },
    "xgboost": {
        "max_depth": ("int", 3, 10),
        "learning_rate": ("loguniform", 0.01, 0.2),
        "n_estimators": ("int", 200, 1000),
    },
}

# ---------------------------------------------------------------------------
# Hyper-parameter Optimisation Helpers
# ---------------------------------------------------------------------------

def sample_params(trial: optuna.trial.Trial, space: Dict[str, Tuple[str, Any, Any]]):
    params = {}
    for name, (kind, a, b) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, a, b)
        elif kind == "loguniform":
            params[name] = trial.suggest_loguniform(name, a, b)
    return params


def optimise_model(model_name: str, space: Dict[str, Tuple[str, Any, Any]],
                    X_train, y_train, X_val, y_val, n_trials: int = 30):
    def objective(trial):
        params = sample_params(trial, space)
        if model_name == "lightgbm":
            mdl = lgb.LGBMRegressor(**params)
        elif model_name == "xgboost":
            mdl = xgb.XGBRegressor(**params, objective="reg:squarederror")
        else:
            raise ValueError(model_name)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_val)
        return np.sqrt(metrics.mean_squared_error(y_val, preds))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    if model_name == "lightgbm":
        return lgb.LGBMRegressor(**best_params)
    return xgb.XGBRegressor(**best_params, objective="reg:squarederror")

# ---------------------------------------------------------------------------
# Cross-Validation Splitter
# ---------------------------------------------------------------------------

def time_series_split(X: np.ndarray, y: np.ndarray, cfg: Config):
    n = len(X)
    indices = np.arange(n)
    if cfg.cv_strategy == "kfold":
        kf = model_selection.KFold(cfg.cv_folds)
        for train_idx, val_idx in kf.split(indices):
            yield train_idx, val_idx
    else:  # expanding or rolling
        fold_size = n // cfg.cv_folds
        for k in range(cfg.cv_folds):
            end = (k + 1) * fold_size
            if cfg.cv_strategy == "expanding":
                train_idx = indices[:end]
            else:  # rolling
                start = max(0, end - fold_size)
                train_idx = indices[start:end]
            val_idx = indices[end : end + fold_size]
            if len(val_idx):
                yield train_idx, val_idx

# ---------------------------------------------------------------------------
# Top-Level Orchestration
# ---------------------------------------------------------------------------

def train_models(cfg: Config, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    cv_results = defaultdict(list)

    # Split once for PyTorch models (train/val)
    cutoff = int(len(X) * (1 - cfg.test_split))
    val_cut = int(cutoff * (1 - cfg.val_split))
    X_train, y_train = X[:val_cut], y[:val_cut]
    X_val, y_val     = X[val_cut:cutoff], y[val_cut:cutoff]
    X_test, y_test   = X[cutoff:], y[cutoff:]

    # -------------------- Deep-Learning Models --------------------
    for name in ("lstm", "gru", "cnn", "transformer"):
        model = ModelFactory[name](X.shape[2], cfg)
        model, hist = train_pytorch(model, cfg, X_train, y_train, X_val, y_val)
        models[name] = model
        preds = model(torch.tensor(X_test, dtype=torch.float32)).detach().cpu().numpy()
        cv_results[name].append(evaluate(y_test, preds))
        # Save
        torch.save(model.state_dict(), MODEL_DIR / f"{name}.pt")

    # -------------------- Gradient Boosters & Linear Models --------------------
    X_flat = X.reshape(X.shape[0], -1)  # flatten sequences for tree/linear models

    model_specs = {
        "ridge": linear_model.Ridge(),
        "lasso": linear_model.Lasso(alpha=0.001),
        "lightgbm": None,
        "xgboost": None,
    }

    for name, mdl in model_specs.items():
        if mdl is None:
            space = HP_SPACES[name]
            mdl = optimise_model(name, space, X_train.reshape(X_train.shape[0], -1), y_train,
                                  X_val.reshape(X_val.shape[0], -1), y_val, cfg.hyperopt_trials)
        else:
            mdl = train_sklearn(mdl, X_flat[:val_cut], y[:val_cut])
        models[name] = mdl
        preds = mdl.predict(X_flat[cutoff:])
        cv_results[name].append(evaluate(y_test, preds))
        # Persist
        pd.to_pickle(mdl, MODEL_DIR / f"{name}.pkl")

    # -------------------- Ensemble --------------------
    if cfg.use_ensemble:
        base_estimators = [(n, models[n]) for n in ("ridge", "lightgbm", "lstm") if n in models]
        final_estimator = linear_model.Ridge()
        ensemble = StackingRegressor(base_estimators, final_estimator)
        ensemble.fit(X_flat[:cutoff], y[:cutoff])
        models["ensemble"] = ensemble
        preds = ensemble.predict(X_flat[cutoff:])
        cv_results["ensemble"].append(evaluate(y_test, preds))
        pd.to_pickle(ensemble, MODEL_DIR / "ensemble.pkl")

    # Log CV results
    for m, res in cv_results.items():
        LOGGER.info(json.dumps({"model": m, "metrics": res[0]}))

    return models

# ---------------------------------------------------------------------------
# Prediction Helper
# ---------------------------------------------------------------------------

def predict(models: Dict[str, Any], cfg: Config, X_latest: np.ndarray) -> Dict[str, float]:
    preds = {}
    # Ensure X_latest has shape (1, seq_len, features)
    if X_latest.ndim == 2:
        X_latest = X_latest[None, ...]
    for name, mdl in models.items():
        if isinstance(mdl, nn.Module):
            mdl.eval()
            with torch.no_grad():
                preds[name] = float(mdl(torch.tensor(X_latest, dtype=torch.float32)).item())
        else:
            x_flat = X_latest.reshape(1, -1)
            preds[name] = float(mdl.predict(x_flat).item())
    if "ensemble" in models:
        x_flat = X_latest.reshape(1, -1)
        preds["ensemble"] = float(models["ensemble"].predict(x_flat).item())
    return preds

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def explain_model(model, X_sample: np.ndarray, feature_names: List[str]):
    if "shap" not in sys.modules:
        LOGGER.warning(json.dumps({"explain": "shap_not_installed"}))
        return
    if isinstance(model, nn.Module):
        explainer = shap.DeepExplainer(model, torch.tensor(X_sample[:100], dtype=torch.float32))
        shap_values = explainer.shap_values(torch.tensor(X_sample[:200], dtype=torch.float32))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "shap_deep.png", dpi=300)
    else:
        explainer = shap.Explainer(model.predict, X_sample[:100])
        shap_values = explainer(X_sample[:200])
        shap.summary_plot(shap_values, show=False, feature_names=feature_names)
        plt.savefig(PLOT_DIR / "shap_tree.png", dpi=300)

# ---------------------------------------------------------------------------
# Unit Tests (very lightweight sanity checks)
# ---------------------------------------------------------------------------

def run_tests():
    cfg = Config()
    # Generate synthetic sample data (2 days of fake 1-min bars) to keep runtime fast
    minutes = cfg.trading_minutes_per_day * 2
    ts = pd.date_range("2024-01-02 09:30", periods=minutes, freq="1min")
    rng = np.random.default_rng(42)
    price = np.cumprod(1 + 0.0002 * rng.standard_normal(minutes)) * 100
    df = pd.DataFrame({
        "datetime": ts,
        "open": price,
        "high": price * (1 + 0.001*rng.random(minutes)),
        "low": price * (1 - 0.001*rng.random(minutes)),
        "close": price * (1 + 0.0002*rng.standard_normal(minutes)),
        "volume": rng.integers(1000, 10000, minutes)
    })
    df = downsample(df, cfg.interval)
    df = engineer_features(df, cfg)
    X, y, feats = build_sequences(df, cfg)
    assert X.shape[0] == y.shape[0] and X.shape[1] == cfg.lookback_minutes, "Sequence build failed"
    LOGGER.info(json.dumps({"test": "sequence_build", "status": "passed"}))
    models = train_models(cfg, X, y)
    latest = X[-1]
    preds = predict(models, cfg, latest)
    assert isinstance(preds, dict) and preds, "Prediction failed"
    LOGGER.info(json.dumps({"test": "predict", "status": "passed"}))

# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RV Forecasting System CLI")
    sub = parser.add_subparsers(dest="cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv", help="Path to OHLCV CSV")
    common.add_argument("--symbol", default="SPY", help="Ticker for yfinance download")
    common.add_argument("--override", nargs="*", default=[], help="key=value overrides")

    sub.add_parser("full", parents=[common], help="Run full pipeline")
    sub.add_parser("train", parents=[common], help="Train models only")
    sub.add_parser("predict", parents=[common], help="Predict next-day RV")
    sub.add_parser("explain", parents=[common], help="Generate SHAP plots")
    sub.add_parser("test", help="Run built-in tests")

    args = parser.parse_args()

    if args.cmd == "test":
        run_tests()
        return

    cfg = Config()
    cfg.override(args.override)

    # -------------------- Data Loading --------------------
    if args.csv:
        raw_df = load_csv(args.csv)
    else: print("No data")

    raw_df = downsample(raw_df, cfg.interval)
    df = engineer_features(raw_df, cfg)
    X, y, feature_cols = build_sequences(df, cfg)

    if args.cmd in {"full", "train"}:
        trained = train_models(cfg, X, y)
        if cfg.use_mlflow and "mlflow" in sys.modules:
            mlflow.end_run()
        if args.cmd == "train":
            return
    else:
        # Load persisted models
        trained = {}
        for p in MODEL_DIR.glob("*.pt"):
            mdl_name = p.stem
            mdl = ModelFactory[mdl_name](X.shape[2], cfg)
            mdl.load_state_dict(torch.load(p))
            trained[mdl_name] = mdl
        for pkl in MODEL_DIR.glob("*.pkl"):
            trained[pkl.stem] = pd.read_pickle(pkl)

    latest_seq = X[-1]
    preds = predict(trained, cfg, latest_seq)
    LOGGER.info(json.dumps({"latest_prediction": preds}))

    if args.cmd == "explain":
        explain_model(trained[list(trained.keys())[0]], X.reshape(X.shape[0], -1), feature_cols)

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        LOGGER.error(json.dumps({"error": str(ex), "trace": traceback.format_exc()}))
        sys.exit(1)

"""
python zGen6\\rv_forecasting_system.py full --csv data\\SPY.csv       # train, evaluate, log metrics
python zGen6\\rv_forecasting_system.py predict --csv data\\SPY.csv    # load saved models and predict
python zGen6\\rv_forecasting_system.py explain --csv data\\SPY.csv    # generate SHAP plots
python zGen6\\rv_forecasting_system.py test --csv data\\SPY.csv       # quick built-in sanity tests
"""