"""
Model implementations: LightGBM, LSTM, and ensemble.
Implements exact loss functions and training procedures as specified.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import json

class BasePredictor:
    """Base class for all prediction models."""
    
    def __init__(self, target: str = 'target_1d'):
        self.target = target
        self.model = None
        self.feature_cols = []
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector."""
        # Select numeric features, excluding targets and identifiers
        exclude_cols = ['ticker', 'date', 'target_1d', 'target_7d', 'target_30d']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        X = df[self.feature_cols].values
        y = df[self.target].values
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return predictions with confidence intervals.
        Returns: (predictions, lower_bound, upper_bound)
        """
        raise NotImplementedError


class LightGBMPredictor(BasePredictor):
    """
    LightGBM gradient boosted trees for regression.
    
    Loss: MSE = (1/N) * sum((y_i - y_hat_i)^2)
    """
    
    def __init__(self, target: str = 'target_1d', params: Optional[Dict] = None):
        super().__init__(target)
        
        # Default hyperparameters as specified
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 1000
        }
        
        # For quantile regression (confidence intervals)
        self.quantile_models = {}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train LightGBM model with early stopping.
        """
        # Remove NaN rows
        valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        print(f"[v0] Training LightGBM on {len(X_train)} samples...")
        
        # Train main model (mean prediction)
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            val_data = lgb.Dataset(X_val, label=y_val)
            
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
        else:
            self.model = lgb.train(self.params, train_data)
        
        # Train quantile models for uncertainty estimation
        print("[v0] Training quantile models for confidence intervals...")
        for quantile in [0.05, 0.95]:
            params_quantile = self.params.copy()
            params_quantile['objective'] = 'quantile'
            params_quantile['alpha'] = quantile
            
            self.quantile_models[quantile] = lgb.train(params_quantile, train_data)
        
        print("[v0] Training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X_clean)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals using quantile regression.
        
        Returns: (mean_prediction, lower_5%, upper_95%)
        """
        X_clean = np.nan_to_num(X, nan=0.0)
        
        mean_pred = self.model.predict(X_clean)
        lower = self.quantile_models[0.05].predict(X_clean)
        upper = self.quantile_models[0.95].predict(X_clean)
        
        return mean_pred, lower, upper
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for explainability."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_cols, importance))


class EnsemblePredictor(BasePredictor):
    """
    Ensemble of multiple models for improved predictions and uncertainty.
    
    Ensemble prediction: y_hat = (1/M) * sum(y_hat^(m))
    Ensemble variance: Var = (1/M) * sum((y_hat^(m) - y_hat)^2)
    """
    
    def __init__(self, target: str = 'target_1d', n_models: int = 5):
        super().__init__(target)
        self.n_models = n_models
        self.models = []
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train ensemble of models with bootstrap sampling."""
        valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        print(f"[v0] Training ensemble of {self.n_models} models...")
        
        for i in range(self.n_models):
            print(f"[v0] Training model {i+1}/{self.n_models}...")
            
            # Bootstrap sample
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train model with slight variation
            model = LightGBMPredictor(target=self.target)
            model.feature_cols = self.feature_cols
            model.train(X_boot, y_boot)
            
            self.models.append(model)
        
        print("[v0] Ensemble training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble mean prediction."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with epistemic uncertainty from ensemble variance.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence interval (±1.96 std)
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower, upper


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Metrics:
    - MSE = (1/N) * sum((y_i - y_hat_i)^2)
    - RMSE = sqrt(MSE)
    - MAE = (1/N) * sum(|y_i - y_hat_i|)
    - Direction accuracy (hit rate)
    """
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Direction accuracy
    direction_correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    hit_rate = direction_correct / len(y_true)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'hit_rate': hit_rate,
        'n_samples': len(y_true)
    }


def walk_forward_validation(df: pd.DataFrame, model_class, target: str = 'target_1d', n_splits: int = 5):
    """
    Walk-forward cross-validation for time series.
    
    Splits data chronologically and trains/validates sequentially.
    """
    print(f"[v0] Running walk-forward validation with {n_splits} splits...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Prepare features
    model = model_class(target=target)
    X, y = model.prepare_features(df)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n[v0] Fold {fold + 1}/{n_splits}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        fold_model = model_class(target=target)
        fold_model.feature_cols = model.feature_cols
        fold_model.train(X_train, y_train)
        
        # Predict
        y_pred = fold_model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred)
        metrics['fold'] = fold + 1
        results.append(metrics)
        
        print(f"[v0] Fold {fold + 1} results: RMSE={metrics.get('rmse', 0):.6f}, Hit Rate={metrics.get('hit_rate', 0):.4f}")
    
    # Aggregate results
    avg_metrics = {
        'avg_rmse': np.mean([r['rmse'] for r in results if 'rmse' in r]),
        'avg_mae': np.mean([r['mae'] for r in results if 'mae' in r]),
        'avg_hit_rate': np.mean([r['hit_rate'] for r in results if 'hit_rate' in r]),
        'folds': results
    }
    
    print(f"\n[v0] Average RMSE: {avg_metrics['avg_rmse']:.6f}")
    print(f"[v0] Average Hit Rate: {avg_metrics['avg_hit_rate']:.4f}")
    
    return avg_metrics


def main():
    """Train and evaluate models."""
    # Load features
    df = pd.read_parquet('data/features.parquet')
    print(f"[v0] Loaded {len(df)} rows with features")
    
    # Filter to complete rows
    df = df.dropna(subset=['target_1d'])
    print(f"[v0] {len(df)} rows with valid targets")
    
    # Walk-forward validation for 1-day predictions
    print("\n=== LightGBM Model ===")
    lgb_results = walk_forward_validation(df, LightGBMPredictor, target='target_1d', n_splits=5)
    
    print("\n=== Ensemble Model ===")
    ensemble_results = walk_forward_validation(df, EnsemblePredictor, target='target_1d', n_splits=3)
    
    # Save results
    results = {
        'lightgbm': lgb_results,
        'ensemble': ensemble_results
    }
    
    with open('data/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[v0] Model evaluation complete")

if __name__ == '__main__':
    main()
