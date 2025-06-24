import xgboost as xgb
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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
    
    def fit_pcr(self, X_train: np.ndarray, y_train: np.ndarray, n_components: int = 20) -> None:
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
        #print("Fitting Lasso...")
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
        print("Ensemble Predictions:", predictions)
        if predictions:
            pred_array = np.column_stack(list(predictions.values()))
            return np.mean(pred_array, axis=1)
        return np.array([])
    
class RVForecaster:
    """Main forecasting system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.ensemble = ModelEnsemble(self.config)
        self.feature_names = []   
    
    def _default_config(self) -> Dict:
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
    
    def load_model(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(model_data)
        self.ensemble = model_data['ensemble']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']

        # Explicitly restore the models if they're empty
        if not self.ensemble.models and 'models' in model_data:
            self.ensemble.models = model_data['models']
            self.ensemble.scalers = model_data['scalers']
            self.ensemble.pca = model_data['pca']
        print("Model loaded successfully")
        print("Available models:", list(self.ensemble.models.keys()))
        print("Models dict:", self.ensemble.models)

model_path = "models/rv_forecaster.pkl"
Forecaster = RVForecaster()
Forecaster.load_model(str(model_path))
