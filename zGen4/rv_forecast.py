#!/usr/bin/env python3
### SETTING  
interval = '30min'  # Default interval for realized variance features []min []H []D

###
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time


# ML imports
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

class RealizedVolatilityForecaster:
    """
    Main class for realized volatility forecasting system
    Following Li & Tang (2021) methodology
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.pca = None
        self.feature_names = []
        
    def _default_config(self) -> Dict:
        return {
            'rv_lags': [1, 5, 22],  # daily, weekly, monthly
            'iv_maturities': [30, 60, 90],  # days
            'iv_deltas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'rolling_window': 252,  # 1 year training window
            'test_window': 22,  # 1 month test window
            'random_state': 42,
            'log_transform': True
        }

class RVFeatureEngine:
    """Feature engineering for realized variance features"""
    
    @staticmethod
    def compute_rv_features(df_prices: pd.DataFrame) -> pd.DataFrame:
        offset = pd.tseries.frequencies.to_offset(interval)
        delta = pd.Timedelta(offset.nanos)
        intervaltominute = int(delta.total_seconds() / 60)

        print(f"Computing realized variance features for {intervaltominute}-minute intervals")

        hour = int(60 / intervaltominute)
        day = int(390 / intervaltominute)  # 6.5 hours of trading
        week = int(1444 / intervaltominute)  # 5 days of trading
        print(f"Using hour={hour}, day={day}, week={week} for {intervaltominute}-minute intervals")
        """
        Compute realized variance features from high-frequency price data per 5-minute intervals

        Args:
            df_prices: DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with RV features per 5-minute interval
        """
        df = df_prices.copy()
        df['date'] = pd.to_datetime(df['datetime'])
        df.set_index('date', inplace=True)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['returns_sq'] = df['returns'] ** 2

        # Initialize features DataFrame with same index as input
        rv_features = pd.DataFrame(index=df.index)

        # Current 5-minute RV (squared return)
        rv_features['RV'] = df['returns_sq']

        # Lagged RV features (previous minute intervals)
        rv_features['RV_lag1'] = df['returns_sq'].shift(1)
        rv_features['RV_lag2'] = df['returns_sq'].shift(2)
        rv_features['RV_lag3'] = df['returns_sq'].shift(3)
        rv_features['RV_lag5'] = df['returns_sq'].shift(5)

        # HAR components - rolling sums over different horizons
        # Note: Using 12 periods = 1 hour, 78 periods ≈ 1 day (6.5 trading hours), 390 ≈ 5 days
        rv_features['RV_hourly'] = df['returns_sq'].rolling(hour).sum()  # 1 hour RV
        rv_features['RV_daily'] = df['returns_sq'].rolling(day).sum()   # 1 day RV  
        rv_features['RV_weekly'] = df['returns_sq'].rolling(week).sum() # 5 days RV

        # SHAR (Square root of realized volatility)
        rv_features['RV_sqrt'] = np.sqrt(df['returns_sq'])
        rv_features['RV_sqrt_lag1'] = rv_features['RV_sqrt'].shift(1)
        rv_features['RV_sqrt_hourly'] = rv_features['RV_sqrt'].rolling(hour).mean()
        rv_features['RV_sqrt_daily'] = rv_features['RV_sqrt'].rolling(day).mean()
        
        # HARQ-F (Quadratic HAR)
        rv_features['RV_quad'] = df['returns_sq'] ** 2
        rv_features['RV_quad_lag1'] = rv_features['RV_quad'].shift(1)
        rv_features['RV_quad_hourly'] = rv_features['RV_quad'].rolling(hour).mean()
        rv_features['RV_quad_daily'] = rv_features['RV_quad'].rolling(day).mean()

        # MIDAS features (exponential weights) - adjusted spans for 5-min frequency
        if intervaltominute < 60:  # If time under 1 Hour
            for span in [hour, day, week]:  # 1 hour, 1 day, 5 days
                rv_features[f'RV_ema_{span}'] = df['returns_sq'].ewm(span=span).mean()

        # HExpGl (heterogeneous exponential gains/losses)
        positive_rv = df['returns_sq'].where(df['returns_sq'] > df['returns_sq'].shift(1), 0)
        negative_rv = df['returns_sq'].where(df['returns_sq'] <= df['returns_sq'].shift(1), 0)

        rv_features['RV_pos_hourly'] = positive_rv.rolling(hour).sum()
        rv_features['RV_neg_hourly'] = negative_rv.rolling(hour).sum()
        rv_features['RV_pos_daily'] = positive_rv.rolling(day).sum()
        rv_features['RV_neg_daily'] = negative_rv.rolling(day).sum()

        # Additional statistical features
        rv_features['RV_std_hourly'] = df['returns_sq'].rolling(hour).std()
        rv_features['RV_std_daily'] = df['returns_sq'].rolling(day).std()
        #rv_features['RV_skew_hourly'] = df['returns_sq'].rolling(day).skew() ##CHANGE
        #rv_features['RV_kurt_hourly'] = df['returns_sq'].rolling(day).kurt() ##CHANGE

        # Range-based volatility per 5-minute interval
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # Garman-Klass volatility for each 5-minute bar
            gk_vol = np.log(df['high']/df['low'])**2 - (2*np.log(2)-1) * np.log(df['close']/df['open'])**2
            rv_features['GK_vol'] = gk_vol
            rv_features['GK_vol_lag1'] = gk_vol.shift(1)
            rv_features['GK_vol_hourly'] = gk_vol.rolling(hour).mean()
            rv_features['GK_vol_daily'] = gk_vol.rolling(day).mean()

        # Additional 5-minute specific features
        # Intraday volatility patterns
        rv_features['RV_ma_short'] = df['returns_sq'].rolling(int(hour / 2)).mean()   # 30-min MA
        rv_features['RV_ma_medium'] = df['returns_sq'].rolling(int(hour * 2)).mean() # 2-hour MA

        # Volatility momentum
        rv_features['RV_momentum'] = df['returns_sq'] - df['returns_sq'].rolling(hour).mean()

        # Volatility ratio features
        rv_features['RV_ratio_short'] = df['returns_sq'] / df['returns_sq'].rolling(hour).mean()
        rv_features['RV_ratio_medium'] = df['returns_sq'] / df['returns_sq'].rolling(day).mean()

        print("Features shape:", rv_features.shape)
        if intervaltominute > 60:
            rv_features = rv_features.dropna(axis=1, how='all')  # Drop columns with all NaN values
        print("Final RV Features shape:", rv_features.shape)
        return rv_features
    
class IVFeatureEngine:
    """Feature engineering for implied variance features"""
    
    @staticmethod
    def fetch_options_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = 'SPY'
        """
        Fetch options data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with implied volatility features
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical stock price for risk-free rate calculation
            stock_data = ticker.history(start=start_date, end=end_date)
            
            iv_features = []
            
            # Iterate through dates to get options data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for date in date_range:
                if date.weekday() < 5:  # Skip weekends
                    try:
                        # Get options expirations
                        expirations = ticker.options
                        
                        if len(expirations) > 0:
                            date_features = {'date': date}
                            
                            # Process each maturity bucket
                            for maturity_days in [30, 60, 90]:
                                target_exp = date + timedelta(days=maturity_days)
                                
                                # Find closest expiration
                                exp_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
                                closest_exp = min(exp_dates, key=lambda x: abs((x - target_exp.date()).days))
                                closest_exp_str = closest_exp.strftime('%Y-%m-%d')
                                
                                if closest_exp_str in expirations:
                                    # Get options chain
                                    opt_chain = ticker.option_chain(closest_exp_str)
                                    
                                    # Process calls and puts
                                    for opt_type, chain in [('call', opt_chain.calls), ('put', opt_chain.puts)]:
                                        if not chain.empty:
                                            # Calculate moneyness and extract IV for different deltas
                                            current_price = stock_data.loc[stock_data.index.date == date.date(), 'Close']
                                            if not current_price.empty:
                                                S = current_price.iloc[0]
                                                
                                                for delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                                                    # Approximate strike for given delta
                                                    if opt_type == 'call':
                                                        target_strike = S * (1 + delta * 0.1)  # Simplified approximation
                                                    else:
                                                        target_strike = S * (1 - delta * 0.1)
                                                    
                                                    # Find closest strike
                                                    if 'strike' in chain.columns and 'impliedVolatility' in chain.columns:
                                                        closest_idx = (chain['strike'] - target_strike).abs().idxmin()
                                                        iv_val = chain.loc[closest_idx, 'impliedVolatility']
                                                        
                                                        feature_name = f'IV_{maturity_days}d_{opt_type}_delta{int(delta*10)}'
                                                        date_features[feature_name] = iv_val if pd.notna(iv_val) else 0
                            
                            iv_features.append(date_features)
                            
                    except Exception as e:
                        print(f"Error processing options for {date}: {e}")
                        continue
            
            return pd.DataFrame(iv_features).set_index('date')
            
        except Exception as e:
            print(f"Error fetching options data for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def compute_iv_features(df_iv: pd.DataFrame) -> pd.DataFrame:
        """
        Process implied volatility data into features
        
        Args:
            df_iv: DataFrame with implied volatility data
            
        Returns:
            DataFrame with processed IV features (targeting ~102 columns)
        """
        if df_iv.empty:
            return pd.DataFrame()
        
        iv_features = df_iv.copy()
        
        # Add cross-sectional features
        maturities = [30, 60, 90]
        deltas = list(range(1, 10))  # 0.1 to 0.9
        
        for maturity in maturities:
            # Average IV across deltas for each maturity
            call_cols = [col for col in df_iv.columns if f'IV_{maturity}d_call' in col]
            put_cols = [col for col in df_iv.columns if f'IV_{maturity}d_put' in col]
            
            if call_cols:
                iv_features[f'IV_{maturity}d_call_avg'] = df_iv[call_cols].mean(axis=1)
                iv_features[f'IV_{maturity}d_call_std'] = df_iv[call_cols].std(axis=1)
            
            if put_cols:
                iv_features[f'IV_{maturity}d_put_avg'] = df_iv[put_cols].mean(axis=1)
                iv_features[f'IV_{maturity}d_put_std'] = df_iv[put_cols].std(axis=1)
            
            # IV slope (term structure)
            if call_cols and len(call_cols) > 1:
                iv_features[f'IV_{maturity}d_call_slope'] = df_iv[call_cols].diff(axis=1).mean(axis=1)
            
            if put_cols and len(put_cols) > 1:
                iv_features[f'IV_{maturity}d_put_slope'] = df_iv[put_cols].diff(axis=1).mean(axis=1)
        
        # Cross-maturity features
        for delta in deltas:
            call_30 = f'IV_30d_call_delta{delta}'
            call_60 = f'IV_60d_call_delta{delta}'
            call_90 = f'IV_90d_call_delta{delta}'
            
            if all(col in df_iv.columns for col in [call_30, call_60, call_90]):
                iv_features[f'IV_call_delta{delta}_term_slope'] = (df_iv[call_90] - df_iv[call_30]) / 60
                iv_features[f'IV_call_delta{delta}_term_curve'] = df_iv[call_60] - (df_iv[call_30] + df_iv[call_90]) / 2
        
        # Put-call IV differences
        for maturity in maturities:
            for delta in deltas:
                call_col = f'IV_{maturity}d_call_delta{delta}'
                put_col = f'IV_{maturity}d_put_delta{delta}'
                
                if call_col in df_iv.columns and put_col in df_iv.columns:
                    iv_features[f'IV_{maturity}d_pc_diff_delta{delta}'] = df_iv[call_col] - df_iv[put_col]
        
        # Lagged IV features
        for lag in [1, 2, 5]:
            lag_cols = [col for col in df_iv.columns if col.startswith('IV_')]
            for col in lag_cols:
                iv_features[f'{col}_lag{lag}'] = df_iv[col].shift(lag)
        print("IV Features shape:", iv_features.shape)
        print("IV is NAN:", iv_features.isna().sum().sum())
        return iv_features.dropna()

class ModelEnsemble:
    """Ensemble of ML models for volatility forecasting"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.pca = None
        
    def fit_lasso(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Lasso regression with cross-validation"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        lasso = LassoCV(cv=5, random_state=self.config['random_state'], max_iter=2000)
        lasso.fit(X_scaled, y_train)

        self.models['lasso'] = lasso
        self.scalers['lasso'] = scaler
    
    def fit_pcr(self, X_train: np.ndarray, y_train: np.ndarray, n_components: int = 10) -> None:
        """Fit Principal Component Regression"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        pca = PCA(n_components=n_components, random_state=self.config['random_state'])
        X_pca = pca.fit_transform(X_scaled)
        
        pcr = LinearRegression()
        pcr.fit(X_pca, y_train)
        
        self.models['pcr'] = pcr
        self.scalers['pcr'] = scaler
        self.pca = pca
    
    def fit_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Random Forest Regressor"""
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
    
    def fit_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit XGBoost Regressor"""
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config['random_state']
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
    
    def fit_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Neural Network with 2 hidden layers"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        nn = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            random_state=self.config['random_state'],
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1
        )
        nn.fit(X_scaled, y_train)
        
        self.models['neural_network'] = nn
        self.scalers['neural_network'] = scaler
    
    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit all models"""
        self.fit_lasso(X_train, y_train)
        
        #print("Fitting PCR...")
        self.fit_pcr(X_train, y_train)
        
        #print("Fitting Random Forest...")
        self.fit_random_forest(X_train, y_train)
        
        #print("Fitting XGBoost...")
        self.fit_xgboost(X_train, y_train)
        
        #print("Fitting Neural Network...")
        self.fit_neural_network(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions from all models"""
        predictions = {}
        # Lasso
        if 'lasso' in self.models:
            X_scaled = self.scalers['lasso'].transform(X_test)
            predictions['lasso'] = self.models['lasso'].predict(X_scaled)
        # PCR
        if 'pcr' in self.models and self.pca is not None:
            X_scaled = self.scalers['pcr'].transform(X_test)
            X_pca = self.pca.transform(X_scaled)
            predictions['pcr'] = self.models['pcr'].predict(X_pca)
        # Random Forest
        if 'random_forest' in self.models:
            predictions['random_forest'] = self.models['random_forest'].predict(X_test)
        # XGBoost
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict(X_test)

        # Neural Network
        if 'neural_network' in self.models:
            X_scaled = self.scalers['neural_network'].transform(X_test)
            predictions['neural_network'] = self.models['neural_network'].predict(X_scaled)

        return predictions
    
    def ensemble_predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate ensemble prediction (average of all models)"""
        predictions = self.predict(X_test)
        flipped_predictions = {k: -v for k, v in predictions.items()}
        print("Ensemble Predictions:", flipped_predictions)
        if predictions:
            filtered = {k: v for k, v in flipped_predictions.items() if k != 'neural_network'}
            pred_array = np.column_stack(list(filtered.values()))
            return np.mean(pred_array, axis=1)
        return np.array([])

class RVForecaster:
    """Main forecasting system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.ensemble = ModelEnsemble(self.config)
        self.feature_names = []   
    
    ##CHANGE

    def _store_single_result(self, results, predictions, y_actual, date):
        """Store results for a single forecast"""
        results['dates'].append(date)
        results['actual'].append(y_actual)

        # HAR baseline - calculate from recent actual values
        if len(results['actual']) >= 5:
            har_pred = np.mean(results['actual'][-5:])  # Use last 5 actual values
        else:
            har_pred = y_actual  # Fallback for early predictions
        results['har_baseline'].append(har_pred)

        # Store model predictions
        for model_name, pred in predictions.items():
            if model_name not in results['predictions']:
                results['predictions'][model_name] = []

            # Handle different prediction formats
            if hasattr(pred, '__len__') and len(pred) > 0:
                results['predictions'][model_name].append(pred[0])
            elif hasattr(pred, '__len__') and len(pred) == 0:
                results['predictions'][model_name].append(har_pred)  # Fallback
            else:
                results['predictions'][model_name].append(pred)
        #
        # predictions = np.column_stack(list(predictions.values()))
        # Add ensemble prediction with proper error handling
        if 'ensemble' not in results['predictions']:
            results['predictions']['ensemble'] = []
        
        try:
            if hasattr(self.ensemble, 'enemble_predict'): ##CHange
                print("Using ensemble_predict method")
                ensemble_pred = self.ensemble.ensemble_predict(predictions)
            else:
                # Fallback: average of all model predictions
                valid_preds = []
                for model_name, pred in predictions.items():
                    if hasattr(pred, '__len__') and len(pred) > 0:
                        valid_preds.append(pred[0])
                    elif not hasattr(pred, '__len__'):
                        valid_preds.append(pred)
            
                ensemble_pred = np.mean(valid_preds) if valid_preds else har_pred
            
            # Handle ensemble prediction result
            if hasattr(ensemble_pred, '__len__') and len(ensemble_pred) > 0:
                results['predictions']['ensemble'].append(ensemble_pred[0])
            elif hasattr(ensemble_pred, '__len__') and len(ensemble_pred) == 0:
                results['predictions']['ensemble'].append(har_pred)  # Fallback
            else:
                results['predictions']['ensemble'].append(ensemble_pred)

        except Exception as e:
            print(f"Warning: Ensemble prediction failed: {e}. Using HAR baseline.")
            results['predictions']['ensemble'].append(har_pred)

    def _store_batch_results(self, results, batch_predictions, batch_actuals, batch_dates):
        """Store results for a batch of forecasts"""
        for i, (predictions, actual, date) in enumerate(zip(batch_predictions, batch_actuals, batch_dates)):
            self._store_single_result(results, predictions, actual, date)

    def _merge_results(self, main_results, chunk_results):
        if chunk_results is None or len(chunk_results) == 0:
            return main_results  
        else:    
            """Merge chunk results into main results dictionary"""
            main_results['dates'].extend(chunk_results['dates'])
            main_results['actual'].extend(chunk_results['actual'])
            main_results['har_baseline'].extend(chunk_results['har_baseline'])

            for model_name in chunk_results['predictions']:
                if model_name not in main_results['predictions']:
                    main_results['predictions'][model_name] = []
                main_results['predictions'][model_name].extend(chunk_results['predictions'][model_name])
            return main_results    
    
    ##END CHANGE

    def _default_config(self) -> Dict:
        return {
            'rolling_window': 252,
            'test_window': 22,
            'random_state': 42,
            'log_transform': True
        }
    
    def prepare_features(self, df_prices: pd.DataFrame, df_iv: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Combine RV and IV features"""
        # Compute RV features
        rv_features = RVFeatureEngine.compute_rv_features(df_prices)
        df_iv = None
        if df_iv is not None and not df_iv.empty:
            # Compute IV features
            iv_features = IVFeatureEngine.compute_iv_features(df_iv)
            
            # Merge features on date
            print("RV features index:", rv_features.index.names)

            print("IV features index:", iv_features.index.names)  

            features = full_feature_integration(rv_features, iv_features)
        else:
            features = rv_features
        self.feature_names = list(features.columns)
        return features
    
    def create_target(self, features: pd.DataFrame, target_col: str = 'RV') -> pd.Series:
        """Create next-day RV target"""
        target = features[target_col].shift(-1)  # Next day RV
        
        if self.config['log_transform']:
            target = np.log(target + 1e-8)  # Add small constant to avoid log(0)
        return target      

    def rolling_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Perform rolling window forecasting"""
        results = {
            'dates': [],
            'actual': [],
            'predictions': {model: [] for model in ['lasso', 'pcr', 'random_forest', 'xgboost', 'neural_network', 'ensemble']},
            'har_baseline': []
        }
        ##CHANGE
        features = features.apply(pd.to_numeric, errors='coerce')
        target = pd.to_numeric(target, errors='coerce')
        print("Features shape:", features.shape)
        print("Target shape:", target.shape)
        ##  
        #       
        feature_matrix = features.values
        target_values = target.values




        valid_idx = ~(np.isnan(feature_matrix).any(axis=1) | np.isnan(target_values))
        print("Drop invalid rows:", np.sum(~valid_idx), "out of", len(valid_idx))
        feature_matrix = feature_matrix[valid_idx]
        target_values = target_values[valid_idx]
        dates = features.index[valid_idx]
        
        # Rolling window forecasting
        print("Feature length: ", len(feature_matrix))
        print("Rolling window: ", self.config['rolling_window'])
        print("Doing ", len(feature_matrix) - self.config['rolling_window'], " Forecasts...")
        for i in range(self.config['rolling_window'], len(feature_matrix) - 1):
            # Training data
            X_train = feature_matrix[i-self.config['rolling_window']:i]
            y_train = target_values[i-self.config['rolling_window']:i]
            
            # Test data (single observation)
            X_test = feature_matrix[i:i+1]
            y_test = target_values[i]
            
            # Fit models
            self.ensemble.fit_all(X_train, y_train)
            
            # Generate predictions
            predictions = self.ensemble.predict(X_test)
            ensemble_pred = self.ensemble.ensemble_predict(X_test)
            
            # HAR baseline (simple autoregressive model)
            har_pred = X_train[:, 0][-5:].mean()  # Use recent RV average as baseline
            
            # Store results
            results['dates'].append(dates[i])
            results['actual'].append(y_test)
            results['har_baseline'].append(har_pred)
            
            for model_name, pred in predictions.items():
                results['predictions'][model_name].append(pred[0])
            
            results['predictions']['ensemble'].append(ensemble_pred[0] if len(ensemble_pred) > 0 else har_pred)
            
            if (i - self.config['rolling_window']) % 10 == 0:
                print(f"Processed {i - self.config['rolling_window'] + 1} forecasts...")
        
        return results
    
    ##CHANGE Start 

    def optimized_rolling_forecast(self, feature_matrix, target_values, dates, batch_size=50):
        """
        Process multiple forecasts in batches instead of one-by-one
        """
        results = {
            'dates': [],
            'actual': [],
            'predictions': {model: [] for model in ['lasso', 'pcr', 'random_forest', 'xgboost', 'neural_network', 'ensemble']},
            'har_baseline': []
        }

        # Pre-allocate arrays for better memory management
        n_forecasts = len(feature_matrix) - self.config['rolling_window']

        for batch_start in range(0, n_forecasts, batch_size):
            batch_end = min(batch_start + batch_size, n_forecasts)
            batch_predictions = []
            batch_actuals = []
            batch_dates = []

            # Process batch
            for i in range(batch_start, batch_end):
                idx = i + self.config['rolling_window']
            
                # Training data
                X_train = feature_matrix[idx-self.config['rolling_window']:idx]
                y_train = target_values[idx-self.config['rolling_window']:idx]

                # Only retrain if significant data change (adaptive retraining)
                if i % 10 == 0 or self._should_retrain(X_train, y_train):
                    self.ensemble.fit_all(X_train, y_train)

                # Test data
                X_test = feature_matrix[idx:idx+1]
                batch_predictions.append(self.ensemble.predict(X_test))
                batch_actuals.append(target_values[idx])
                batch_dates.append(dates[idx])

            # Store batch results
            self._store_batch_results(results, batch_predictions, batch_actuals, batch_dates)

        print(f"Processed batch {batch_start//batch_size + 1}, forecasts: {batch_end}")

        return results
    
    def _should_retrain(self, X_train, y_train, threshold=0.1):
        """
        Only retrain when data distribution changes significantly
        """
        if not hasattr(self, '_last_train_stats'):
            self._last_train_stats = {'mean': np.mean(y_train), 'std': np.std(y_train)}
            return True

        current_mean = np.mean(y_train)
        current_std = np.std(y_train)

        # Check if distribution changed significantly
        mean_change = abs(current_mean - self._last_train_stats['mean']) / self._last_train_stats['mean']
        std_change = abs(current_std - self._last_train_stats['std']) / self._last_train_stats['std']
    
        if mean_change > threshold or std_change > threshold:
            self._last_train_stats = {'mean': current_mean, 'std': current_std}
            return True
    
        return False

    def parallel_rolling_forecast(self, feature_matrix, target_values, dates, n_workers=4):
        """
        Use parallel processing for independent model training
        """
        results = {
            'dates': [],
            'actual': [],
            'predictions': {model: [] for model in ['lasso', 'pcr', 'random_forest', 'xgboost', 'neural_network', 'ensemble']},
            'har_baseline': []
        }

        # Split work into chunks
        n_forecasts = len(feature_matrix) - self.config['rolling_window']
        chunk_size = max(1, n_forecasts // n_workers)

        def process_chunk(start_idx, end_idx):
            chunk_results = {
            'dates': [],
            'actual': [],
            'predictions': {model: [] for model in ['lasso', 'pcr', 'random_forest', 'xgboost', 'neural_network', 'ensemble']},
            'har_baseline': []
        }
            local_ensemble = copy.deepcopy(self.ensemble)  # Each thread gets its own models


            total = end_idx - start_idx
            start_time = time.time()

            for i in range(start_idx, end_idx):
                idx = i + self.config['rolling_window']
                X_train = feature_matrix[idx - self.config['rolling_window']:idx]
                y_train = target_values[idx - self.config['rolling_window']:idx]
                X_test = feature_matrix[idx:idx + 1]
                # Train and predict
                local_ensemble.fit_all(X_train, y_train)
                predictions = local_ensemble.predict(X_test)

                # Store results
                self._store_single_result(chunk_results, predictions, target_values[idx], dates[idx])

                # Progress bar
                progress = i - start_idx + 1
                percent = 100 * progress // total
                bar_len = 40
                filled_len = percent * bar_len // 100
                bar = '#' * filled_len + '-' * (bar_len - filled_len)

                elapsed = time.time() - start_time
                if progress > 0:
                    est_total = elapsed / progress * total
                    eta = est_total - elapsed
                else:
                    eta = 0

                # Format ETA as minutes:seconds
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                print(f"\rProgress: |{bar}| {percent}% ({progress}/{total}) ETA: {eta_min}m {eta_sec}s", end='', flush=True)

            # Optional: move to next line after loop
            print()

            return chunk_results

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(0, n_forecasts, chunk_size):
                end_idx = min(i + chunk_size, n_forecasts)
                future = executor.submit(process_chunk, i, end_idx)
                futures.append(future)
            total = len(futures)
            completed = 0
            # Collect results
            for future in as_completed(futures):
                chunk_results = future.result()
                self._merge_results(results, chunk_results)
                completed += 1
                if completed == total:
                    self.save_model(str(model_file))
        return results
    
    def vectorized_rolling_forecast(self, feature_matrix, target_values, dates):
        """
        Use numpy views and vectorized operations where possible
        """
        results = {
            'dates': [],
            'actual': [],
            'predictions': {model: [] for model in ['lasso', 'pcr', 'random_forest', 'xgboost', 'neural_network', 'ensemble']},
            'har_baseline': []
        }
        window_size = self.config['rolling_window']

        # Pre-compute all training windows using stride tricks
        from numpy.lib.stride_tricks import sliding_window_view
        
        #NO PROBLEM
        # Create sliding windows for features and targets
        X_windows = sliding_window_view(feature_matrix, window_size, axis=0)[:-1]  # Remove last incomplete window
        y_windows = sliding_window_view(target_values, window_size, axis=0)[:-1]
        
        # Batch process with periodic retraining
        retrain_frequency = 20  # Retrain every 20 steps

        for i in range(0, len(X_windows), retrain_frequency):
            batch_end = min(i + retrain_frequency, len(X_windows))

            # Process batch with same model
            for j in range(i, batch_end):
                if j % retrain_frequency == 0:  # Retrain periodically
                    self.ensemble.fit_all(X_windows[j], y_windows[j])

                # Predict
                X_test = feature_matrix[j + window_size:j + window_size + 1]
                y_actual = target_values[j + window_size]

                predictions = self.ensemble.predict(X_test)
                self._store_single_result(results, predictions, y_actual, dates[j + window_size])

            print(f"Processed {batch_end} forecasts...")

        return results
    
    def feature_selection_pipeline(self, feature_matrix, target_values):
        """
        Select most important features to speed up training
        """
        from sklearn.feature_selection import SelectKBest, f_regression

        # Select top K features based on correlation with target
        if feature_matrix.shape[1] > 100:  # Only if we have many features
            selector = SelectKBest(score_func=f_regression, k=min(100, feature_matrix.shape[1]//2))

            # Fit on a sample of data
            sample_size = min(10000, len(feature_matrix))
            sample_idx = np.random.choice(len(feature_matrix), sample_size, replace=False)
            
            selector.fit(feature_matrix[sample_idx], target_values[sample_idx])
            feature_matrix = selector.transform(feature_matrix)

            print(f"Selected {feature_matrix.shape[1]} most important features")

        return feature_matrix
    
    def fast_rolling_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        print("Starting fast rolling forecast...")
        results = {
            'dates': [],
            'actual': [],
            'predictions': {model: [] for model in ['lasso', 'pcr', 'random_forest', 'xgboost', 'neural_network', 'ensemble']},
            'har_baseline': []
        }
        
        ##CHANGE
        features = features.apply(pd.to_numeric, errors='coerce')
        target = pd.to_numeric(target, errors='coerce')
        print("Features shape:", features.shape)
        print("Target shape:", target.shape)
        ##        
        feature_matrix = features.values
        target_values = target.values
        valid_idx = ~(np.isnan(feature_matrix).any(axis=1) | np.isnan(target_values))
        print("Drop invalid rows:", np.sum(~valid_idx), "out of", len(valid_idx))
        feature_matrix = feature_matrix[valid_idx]
        target_values = target_values[valid_idx]
        dates = features.index[valid_idx]
        """
        Complete optimized pipeline combining all improvements
        """
        # 1. Feature selection
        feature_matrix = self.feature_selection_pipeline(feature_matrix, target_values)

        # 2. Choose best method based on data size
        n_forecasts = len(feature_matrix) - self.config['rolling_window']
        print(f"Total forecasts to process: {n_forecasts}")
        if n_forecasts > 1000:
            # Use parallel processing for large datasets
            return self.optimized_rolling_forecast(feature_matrix, target_values, dates)
        elif n_forecasts > 100:
            # Use batch processing for medium datasets
            return self.optimized_rolling_forecast(feature_matrix, target_values, dates)
        else:
            # Original method for small datasets
            return self.rolling_forecast(features, target)

    ##CHANGE End
    
    def evaluate_forecasts(self, results: Dict) -> Dict:
        """Evaluate forecast performance"""
        evaluation = {}
        actual = np.array(results['actual'])
        har_baseline = np.array(results['har_baseline'])
        
        # Calculate R² out-of-sample for each model
        for model_name, predictions in results['predictions'].items():
            pred_array = np.array(predictions)
            
            # R² out-of-sample relative to HAR baseline
            mse_model = mean_squared_error(actual, pred_array)
            mse_baseline = mean_squared_error(actual, har_baseline)
            r2_oos = 1 - (mse_model / mse_baseline)
            
            # Standard R²
            r2_standard = r2_score(actual, pred_array)
            
            evaluation[model_name] = {
                'R2_OOS': r2_oos,
                'R2_standard': r2_standard,
                'MSE': mse_model,
                'RMSE': np.sqrt(mse_model)
            }
        
        return evaluation
    
    def plot_results(self, results: Dict, evaluation: Dict, save_path: str = None):
        """Plot forecasting results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(results['dates'])
        actual = results['actual']
        
        # Time series plot
        axes[0, 0].plot(dates, actual, label='Actual', alpha=0.7)
        axes[0, 0].plot(dates, results['predictions']['ensemble'], label='Ensemble', alpha=0.7)
        axes[0, 0].plot(dates, results['har_baseline'], label='HAR Baseline', alpha=0.7)
        axes[0, 0].set_title('Realized Volatility Forecasts')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        models = list(evaluation.keys())
        r2_oos = [evaluation[model]['R2_OOS'] for model in models]
        
        axes[0, 1].bar(models, r2_oos)
        axes[0, 1].set_title('Out-of-Sample R² by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Prediction scatter plot
        ensemble_pred = results['predictions']['ensemble']
        axes[1, 0].scatter(actual, ensemble_pred, alpha=0.6)
        axes[1, 0].plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted (Ensemble)')
        axes[1, 0].set_title('Actual vs Predicted')
        
        # Residuals plot
        residuals = np.array(actual) - np.array(ensemble_pred)
        axes[1, 1].scatter(ensemble_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        
        """Save trained models"""
        model_data = {
        'ensemble': self.ensemble,
        'config': self.config,
        'feature_names': self.feature_names,
        'models': self.ensemble.models,  # Explicitly save the models dict
        'scalers': self.ensemble.scalers,  # And scalers
        'pca': self.ensemble.pca  # And PCA if needed
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    
        self.ensemble = model_data['ensemble']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']

        # Explicitly restore the models if they're empty
        if not self.ensemble.models and 'models' in model_data:
            self.ensemble.models = model_data['models']
            self.ensemble.scalers = model_data['scalers']
            self.ensemble.pca = model_data['pca']

    def forecast_current_day_rv(self, data_file_path: str, symbol: str = 'SPY') -> Dict:
        """
        Forecast RV for the current day based on the provided data file.

        Args:
            data_file_path (str): Path to the data file (e.g., 'path/to/SPY.csv')
            symbol (str, optional): Stock symbol (default: 'SPY')

        Returns:
            Dict: Forecast results with 'rv_daily', 'rv_annualized', and 'prediction_date'
        """
        try:
            # Load price data
            df_prices = pd.read_csv(data_file_path)
            df_prices = downsample_prices(df_prices, interval)  # Downsample to 5-minute intervals

            # Load or fetch IV data (if available)
            iv_file_path = Path(data_file_path).parent / 'iv_data.csv'
            if iv_file_path.exists():
                df_iv = pd.read_csv(iv_file_path, index_col=0, parse_dates=True)
            else:
                df_iv = IVFeatureEngine.fetch_options_data(symbol, '2022-05-31', datetime.today().strftime('%Y-%m-%d'))

            # Prepare features
            features = self.prepare_features(df_prices, df_iv)

            # Get the last available date in the feature set (assuming it's the most recent data)
            last_date = features.index[-1]
            print(last_date)
            print(datetime.today().date())
            # Check if the last date is today (if not, use the last available date)
            if last_date.date()!= datetime.today().date():
                print(f"Warning: Last available date ({last_date}) is not today. Using last available data.")
            # Select the last row of features (for the most recent date)
            X_test = features.tail(1)
            # Generate prediction
            predictions = self.ensemble.predict(X_test.values)
            ensemble_pred = self.ensemble.ensemble_predict(X_test.values)
            # Extract the predicted RV value
            predicted_rv = ensemble_pred[0]
            print(6)
            # Calculate annualized RV (assuming 252 trading days)
            predicted_rv_annualized = predicted_rv * np.sqrt(252)
            print(7)
            # Return forecast results
            return {
                'rv_daily': predicted_rv,
                'rv_annualized': predicted_rv_annualized,
                'prediction_date': last_date
            }

        except Exception as e:
            print(f"Error forecasting current day RV: {e}")
            return None

def inspect_dict(predictions):
    print(f"Dict type: {type(predictions).__name__}")
    if not predictions:
        print("Dict is empty.")
        return

    print(f"Number of keys: {len(predictions)}")
    for i, (k, v) in enumerate(predictions.items()):
        print(f"\nKey {i}: {repr(k)} ({type(k).__name__})")

        if isinstance(v, np.ndarray):
            print(f"  -> Value is numpy.ndarray, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  -> Value is list, length={len(v)}, sample={v[:3]}")
        elif isinstance(v, (float, int, str)):
            print(f"  -> Value is {type(v).__name__}: {v}")
        elif isinstance(v, dict):
            print(f"  -> Value is nested dict with {len(v)} keys")
        else:
            print(f"  -> Value is {type(v).__name__}")
    
## START OF INTEGRATION FUNCTIONS
def integrate_rv_iv_features(rv_features, iv_features, method='forward_fill'):
    """
    Integrate RV features (5-minute) with IV features (daily) while preserving high frequency
    
    Args:
        rv_features: DataFrame with RV features at 5-minute frequency (297207, 34)
        iv_features: DataFrame with IV features at daily frequency (799, 279)  
        method: Integration method ('forward_fill', 'interpolate', 'intraday_expand', 'hybrid')
    
    Returns:
        DataFrame with integrated features at 5-minute frequency
    """
    
    if method == 'forward_fill':
        return forward_fill_integration(rv_features, iv_features)
    elif method == 'interpolate':
        return interpolate_integration(rv_features, iv_features)
    elif method == 'intraday_expand':
        return intraday_expand_integration(rv_features, iv_features)
    elif method == 'hybrid':
        return hybrid_integration(rv_features, iv_features)
    else:
        raise ValueError("Method must be 'forward_fill', 'interpolate', 'intraday_expand', or 'hybrid'")

def forward_fill_integration(rv_features, iv_features):
    """
    Forward fill daily IV features to match 5-minute RV frequency
    Best for: Features that remain constant throughout the day
    """
    # Ensure both have datetime index
    rv_features = rv_features.copy()
    iv_features = iv_features.copy()
    
    # Create date column for merging
    rv_features['merge_date'] = pd.to_datetime(rv_features.index.date)
    iv_features_reset = iv_features.reset_index()
    iv_features_reset['merge_date'] = pd.to_datetime(iv_features_reset['date'])
    merged = rv_features.merge(iv_features_reset.drop('date', axis=1), on='merge_date', how='left')

    # Forward fill IV features within each day
    iv_cols = [col for col in merged.columns if col.endswith('_iv') or col in iv_features.columns]
    merged[iv_cols] = merged.groupby('merge_date')[iv_cols].fillna(method='ffill')
    
    # Remove helper column
    #merged = merged.drop('merge_date', axis=1)
    return merged

def interpolate_integration(rv_features, iv_features):
    """
    Interpolate IV features to create smooth intraday transitions
    Best for: Features that should transition smoothly during the day
    """
    # Align indices
    rv_features = rv_features.copy()
    iv_features = iv_features.copy()
    
    # Create full timeline
    full_timeline = pd.concat([rv_features[[]], iv_features[[]]], axis=0).sort_index()
    
    # Reindex IV features to full timeline and interpolate
    iv_interpolated = iv_features.reindex(full_timeline.index).interpolate(method='time')
    
    # Merge with RV features
    merged = rv_features.join(iv_interpolated, how='left', rsuffix='_iv')
    
    return merged

def intraday_expand_integration(rv_features, iv_features):
    """
    Create intraday variations of IV features based on RV patterns
    Best for: Creating more realistic intraday IV dynamics
    """
    rv_features = rv_features.copy()
    iv_features = iv_features.copy()
    
    # Start with forward fill
    merged = forward_fill_integration(rv_features, iv_features)
    
    # Create intraday adjustments based on RV patterns
    rv_features['merge_date'] = rv_features.index.date
    daily_rv = rv_features.groupby('merge_date')['RV'].agg(['mean', 'std', 'min', 'max'])
    
    # Add intraday IV adjustments
    merged['merge_date'] = merged.index.merge_date
    merged = merged.merge(daily_rv, on='merge_date', how='left', suffixes=('', '_daily'))
    
    # Adjust IV features based on current vs daily RV
    iv_base_cols = [col for col in iv_features.columns if not col.endswith('_iv')]
    
    for col in iv_base_cols:
        if col in merged.columns:
            # Scale IV based on relative RV
            rv_ratio = merged['RV'] / (merged['RV_mean'] + 1e-8)  # Avoid division by zero
            merged[f'{col}_adjusted'] = merged[col] * (1 + 0.1 * (rv_ratio - 1))  # 10% adjustment factor
    
    # Clean up helper columns
    merged = merged.drop(['merge_date', 'RV_mean', 'RV_std', 'RV_min', 'RV_max'], axis=1)
    
    return merged

def hybrid_integration(rv_features, iv_features):
    """
    Combine multiple methods for optimal integration
    Best for: Maximum information preservation
    """
    # Method 1: Forward fill for baseline
    ff_merged = forward_fill_integration(rv_features, iv_features)
    
    # Method 2: Interpolate for smooth transitions
    interp_merged = interpolate_integration(rv_features, iv_features)
    
    ##CHANGE
    # Method 3: Intraday adjustments
    #intraday_merged = intraday_expand_integration(rv_features, iv_features)
    
    # Combine methods
    final_merged = ff_merged.copy()
    
    # Use interpolated values for continuous features
    continuous_features = ['vix', 'term_structure', 'skew'] # Adjust based on your IV features
    for feature in continuous_features:
        matching_cols = [col for col in interp_merged.columns if feature in col.lower()]
        for col in matching_cols:
            if col in interp_merged.columns:
                final_merged[f'{col}_smooth'] = interp_merged[col]
    ##CHANGE
    # Add intraday adjusted features
    #adjusted_cols = [col for col in intraday_merged.columns if col.endswith('_adjusted')]
    #for col in adjusted_cols:
    #    final_merged[col] = intraday_merged[col]
    
    return final_merged

def create_lagged_iv_features(merged_features, iv_feature_names, lags=[1, 2, 3, 5]):
    """
    Create lagged versions of IV features at 5-minute frequency
    This gives you more predictive power
    """
    merged_features = merged_features.copy()
    
    for feature in iv_feature_names:
        if feature in merged_features.columns:
            for lag in lags:
                merged_features[f'{feature}_lag_{lag}'] = merged_features[feature].shift(lag)
    
    return merged_features

def add_rv_iv_interactions(merged_features, rv_cols, iv_cols):
    """
    Create interaction features between RV and IV
    These can capture important relationships
    """
    merged_features = merged_features.copy()
    
    # Basic interactions
    for rv_col in rv_cols[:5]:  # Limit to avoid too many features
        for iv_col in iv_cols[:10]:  # Limit to avoid too many features
            if rv_col in merged_features.columns and iv_col in merged_features.columns:
                # Ratio features
                merged_features[f'{rv_col}_{iv_col}_ratio'] = (
                    merged_features[rv_col] / (merged_features[iv_col] + 1e-8)
                )
                
                # Difference features  
                merged_features[f'{rv_col}_{iv_col}_diff'] = (
                    merged_features[rv_col] - merged_features[iv_col]
                )
    
    return merged_features

# Example usage function
def full_feature_integration(rv_features, iv_features):
    """
    Complete feature integration pipeline
    """
    print(f"Starting integration:")
    print(f"RV features shape: {rv_features.shape}")

    print(f"IV features shape: {iv_features.shape}")
    
    # Step 1: Basic integration
    merged = integrate_rv_iv_features(rv_features, iv_features, method='hybrid')
    print(f"After integration: {merged.shape}")
    ##CHANGE 
    # Step 2: Add lagged IV features
    iv_feature_names = iv_features.columns.tolist()
    """
    merged = create_lagged_iv_features(merged, iv_feature_names, lags=[1, 2, 3, 5])
    print(f"After lagged features: {merged.shape}")
    """
    # Step 3: Add interaction features
    rv_cols = [col for col in merged.columns if col.startswith('RV')]
    iv_cols = [col for col in merged.columns if col in iv_feature_names]
    merged = add_rv_iv_interactions(merged, rv_cols[:5], iv_cols[:5])  # Limit for demo
    print(f"After interactions: {merged.shape}")
    merged.set_index('merge_date', inplace=True)
    # Step 4: Handle remaining NaNs
    merged = merged.fillna(method='ffill').fillna(method='bfill')
    print(f"Final shape: {merged.shape}")
    
    return merged

## END OF INTEGRATION FUNCTIONS

## DOWNSAMPLING FUNCTIONS
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

def main():
    parser = argparse.ArgumentParser(description='Realized Volatility Forecasting System')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'forecast'], required=True,
                        help='Mode: train, predict, or evaluate')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='results/',
                        help='Path to output directory')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='Path to save/load models')
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Stock symbol for options data')
    parser.add_argument('--start_date', type=str, default='2022-05-31',
                        help='Start date for data')
    parser.add_argument('--end_date', type=str, default= datetime.today().strftime('%Y-%m-%d'),
                        help='End date for data')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        print("Starting training mode...")
        global model_file
        model_file = Path(args.model_path) / 'rv_forecaster.pkl'
        # Load price data
        price_file = Path(args.data_path) ##CHANGE THIS TO YOUR PRICE DATA
        if price_file.exists():
            df_prices = pd.read_csv(price_file)
            df_prices = downsample_prices(df_prices, interval)  # Downsample to 5-minute intervals
            print(f"Loaded price data: {df_prices.shape}")
        else:
            print(f"Price data file not found: {price_file}")
            return
        # Load or fetch IV data
        iv_file = Path(args.data_path) / 'iv_data.csv'
        if iv_file.exists():
            df_iv = pd.read_csv(iv_file, index_col=0, parse_dates=True)
            print(f"Loaded IV data: {df_iv.shape}")
        else:
            print("IV data not found, fetching from yfinance...")
            df_iv = IVFeatureEngine.fetch_options_data(args.symbol, args.start_date, args.end_date)
            if not df_iv.empty:
                df_iv.to_csv(iv_file)
                print(f"Saved IV data: {df_iv.shape}")
            else:
                print("Could not fetch IV data, proceeding with RV features only...")
                df_iv = None
        
        # Initialize forecaster
        config = {
            'rolling_window': 252,
            'test_window': 22,
            'random_state': 42,
            'log_transform': True
        }
        forecaster = RVForecaster(config)
        
        # Prepare features
        print("Preparing features...")
        features = forecaster.prepare_features(df_prices, df_iv)
        print(f"Feature matrix shape: {features.shape}")
        features.to_csv('featurematrix.csv', index=True)

        # Create target
        target = forecaster.create_target(features)
        print(f"Target shape: {target.shape}")
        
        # Perform rolling forecasting
        print("Starting rolling forecast...")
        #results = forecaster.rolling_forecast(features, target)
        # CHANGE NXT
        results = forecaster.fast_rolling_forecast(features, target)
        # Evaluate results
        print("Evaluating forecasts...")
        evaluation = forecaster.evaluate_forecasts(results)
        
        # Print evaluation results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for model_name, metrics in evaluation.items():
            print(f"\n{model_name.upper()}:")
            print(f"  R² Out-of-Sample: {metrics['R2_OOS']:.4f}")
            print(f"  R² Standard: {metrics['R2_standard']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
        
        # Save results
        results_file = Path(args.output_path) / 'forecast_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump({'results': results, 'evaluation': evaluation}, f)
        
        # Save evaluation as CSV
        eval_df = pd.DataFrame(evaluation).T
        eval_df.to_csv(Path(args.output_path) / 'evaluation_results.csv')
        
        # Plot results
        plot_path = Path(args.output_path) / 'forecast_plots.png'
        forecaster.plot_results(results, evaluation, str(plot_path))
        
        # Save model
        #forecaster.ensemble.fit_all(features.values, target.values)
        forecaster.save_model(str(model_file))
        print(f"Model saved to: {model_file}")
        
        # Feature importance for tree-based models
        """if 'random_forest' in forecaster.ensemble.models:
            rf_model = forecaster.ensemble.models['random_forest']
            feature_importance = pd.DataFrame({
                'feature': forecaster.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(Path(args.output_path) / 'feature_importance_rf.csv', index=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Features - Random Forest')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(Path(args.output_path) / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
        """
        print("Training completed successfully!")
    
    elif args.mode == 'predict':
        print("Starting prediction mode...")
        
        # Load model
        model_file = Path(args.model_path) / 'rv_forecaster.pkl'
        if not model_file.exists():
            print(f"Model file not found: {model_file}")
            return
        
        forecaster = RVForecaster()
        forecaster.load_model(str(model_file))
        print("Model loaded successfully")
        print("Available models:", list(forecaster.ensemble.models.keys()))
        print("Models dict:", forecaster.ensemble.models)
        # Load new data
        price_file = Path(args.data_path)  # Change this to your new price data file
        if price_file.exists():
            df_prices = pd.read_csv(price_file)
            df_prices = downsample_prices(df_prices, interval)  # Downsample to 5-minute intervals
            print(f"Loaded new price data: {df_prices.shape}")
        else:
            print(f"New price data file not found: {price_file}")
            return

        # Load IV data if available
        iv_file = Path(args.data_path) / 'iv_data.csv' ##Change
        if iv_file.exists():
            df_iv = pd.read_csv(iv_file, index_col=0, parse_dates=True)
            print(f"Loaded new IV data: {df_iv.shape}")
        else:
            print("New IV data not found, using RV features only...")
            df_iv = None
        
        # Prepare features
        features = forecaster.prepare_features(df_prices, df_iv)
        print(f"Feature matrix shape: {features.shape}")
        
        # Generate predictions
        X_test = features.values
        X_test = pd.DataFrame(X_test)  # Ensure it's a DataFrame if it's not already
        X_test = X_test.apply(pd.to_numeric, errors='coerce')
        valid_idx = ~np.isnan(X_test).any(axis=1)
        X_test_clean = X_test[valid_idx]

        predictions = forecaster.ensemble.predict(X_test_clean)
        ensemble_pred = forecaster.ensemble.ensemble_predict(X_test_clean)
        # Create prediction dataframe
        pred_dates = features.index[valid_idx]
        pred_df = pd.DataFrame({
            'date': pred_dates,
            'ensemble_prediction': ensemble_pred
        })
        
        # Add individual model predictions
        for model_name, pred in predictions.items():
            pred_df[f'{model_name}_prediction'] = pred
        
        # Save predictions
        pred_file = Path(args.output_path) / 'predictions.csv'
        pred_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(pred_dates, ensemble_pred, label='Ensemble Prediction', linewidth=2)
        for model_name, pred in predictions.items():
            plt.plot(pred_dates, pred, label=f'{model_name.title()}', alpha=0.7)
        
        plt.title('Realized Volatility Predictions')
        plt.xlabel('Date')
        plt.ylabel('Predicted RV')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_file = Path(args.output_path) / 'predictions_plot.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Prediction completed successfully!")
    
    elif args.mode == 'evaluate':
        print("Starting evaluation mode...")
        
        # Load results
        results_file = Path(args.output_path) / 'forecast_results.pkl'
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            return
        
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        results = data['results']
        evaluation = data['evaluation']
        
        # Print detailed evaluation
        print("\n" + "="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        
        # Summary table
        summary_data = []
        for model_name, metrics in evaluation.items():
            summary_data.append({
                'Model': model_name.title(),
                'R² OOS': f"{metrics['R2_OOS']:.4f}",
                'R² Standard': f"{metrics['R2_standard']:.4f}",
                'RMSE': f"{metrics['RMSE']:.6f}",
                'MSE': f"{metrics['MSE']:.6f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Utility gains calculation (simplified mean-variance investor)
        print("\n" + "="*60)
        print("UTILITY GAINS ANALYSIS")
        print("="*60)
        
        actual = np.array(results['actual'])
        har_baseline = np.array(results['har_baseline'])
        
        # Risk aversion parameter
        gamma = 3
        
        for model_name, predictions in results['predictions'].items():
            pred_array = np.array(predictions)
            
            # Calculate portfolio returns (simplified)
            # Assume we use volatility forecasts for position sizing
            baseline_returns = -0.5 * gamma * har_baseline  # Utility from baseline
            model_returns = -0.5 * gamma * pred_array  # Utility from model
            
            utility_gain = np.mean(model_returns) - np.mean(baseline_returns)
            utility_gain_pct = (utility_gain / np.abs(np.mean(baseline_returns))) * 100
            
            print(f"{model_name.title()}: {utility_gain_pct:.2f}% utility improvement")
        
        # Additional statistics
        print("\n" + "="*60)
        print("ADDITIONAL STATISTICS")
        print("="*60)
        
        # Directional accuracy
        actual_diff = np.diff(actual)
        har_diff = np.diff(har_baseline)
        
        for model_name, predictions in results['predictions'].items():
            pred_array = np.array(predictions)
            pred_diff = np.diff(pred_array)
            
            # Direction accuracy
            correct_direction = np.sum(np.sign(actual_diff) == np.sign(pred_diff))
            direction_accuracy = correct_direction / len(actual_diff) * 100
            
            # Correlation
            correlation = np.corrcoef(actual, pred_array)[0, 1]
            
            print(f"{model_name.title()}:")
            print(f"  Direction Accuracy: {direction_accuracy:.2f}%")
            print(f"  Correlation: {correlation:.4f}")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        dates = pd.to_datetime(results['dates'])
        
        # Time series comparison
        axes[0, 0].plot(dates, actual, label='Actual', linewidth=2)
        axes[0, 0].plot(dates, results['predictions']['ensemble'], label='Ensemble', alpha=0.8)
        axes[0, 0].plot(dates, har_baseline, label='HAR Baseline', alpha=0.8)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Model comparison
        models = list(evaluation.keys())
        r2_oos = [evaluation[model]['R2_OOS'] for model in models]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = axes[0, 1].bar(models, r2_oos, color=colors)
        axes[0, 1].set_title('Out-of-Sample R² Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, r2_oos):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Residual analysis
        ensemble_pred = np.array(results['predictions']['ensemble'])
        residuals = actual - ensemble_pred
        
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[0, 2].axvline(x=0, color='red', linestyle='--')
        axes[0, 2].set_title('Residuals Distribution (Ensemble)')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Density')
        
        # Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot - Residuals Normality')
        
        # Prediction intervals
        pred_std = np.std(residuals)
        axes[1, 1].plot(dates, actual, label='Actual', alpha=0.8)
        axes[1, 1].plot(dates, ensemble_pred, label='Prediction')
        axes[1, 1].fill_between(dates, 
                               ensemble_pred - 1.96*pred_std,
                               ensemble_pred + 1.96*pred_std,
                               alpha=0.3, label='95% Prediction Interval')
        axes[1, 1].set_title('Predictions with Confidence Intervals')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Rolling R² analysis
        window_size = 50
        rolling_r2 = []
        rolling_dates = []
        
        for i in range(window_size, len(actual)):
            actual_window = actual[i-window_size:i]
            pred_window = ensemble_pred[i-window_size:i]
            r2_window = r2_score(actual_window, pred_window)
            rolling_r2.append(r2_window)
            rolling_dates.append(dates[i])
        
        axes[1, 2].plot(rolling_dates, rolling_r2)
        axes[1, 2].set_title(f'Rolling R² ({window_size}-day window)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save comprehensive evaluation plot
        eval_plot_file = Path(args.output_path) / 'comprehensive_evaluation.png'
        plt.savefig(eval_plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nComprehensive evaluation plot saved to: {eval_plot_file}")
        print("Evaluation completed successfully!")
    
    elif args.mode == 'forecast':
        print("Starting current day RV forecast...")
        forecaster = RVForecaster()
        # Load model
        model_file = Path(args.model_path) / 'rv_forecaster.pkl'
        if not model_file.exists():
            print(f"Model file not found: {model_file}")
            return
        forecaster.load_model(str(model_file))
        print("Model loaded successfully")
        print("Available models:", list(forecaster.ensemble.models.keys()))
        price_file = Path(args.data_path) ##CHANGE THIS TO YOUR PRICE DATA
        def app_log(msg):
            print(f"APPLOG: {msg}")
        forecast_results = forecaster.forecast_current_day_rv(price_file, args.symbol)
        if forecast_results:
            print(f"Forecast Results for {forecast_results['prediction_date']}:")
            print(f"  Daily RV: {forecast_results['rv_daily']:.6f}")
            print(f"  Annualized RV: {forecast_results['rv_daily']:.6f}")
            app_log(f"Annualized Volatility (%): {forecast_results['rv_daily']:.6f}")
        else:
            print("Forecast failed. Check logs for errors.")
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()


"""
Realized Volatility Forecasting System
Based on Li & Tang (2021) - Automatic System for Forecasting Realized Volatility

Usage:
    python zGen4\\rv_forecast.py --mode train --data_path data/ --output_path results/
    python zGen4\\rv_forecast.py --mode predict --model_path models/ --data_path data/
    python zGen4\\rv_forecast.py --mode forecast --data_path data/
"""