"""
Semi-Parametric Structural Model for Counterfactual Demand Prediction
Modified version for markdown-only channel with base prices
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for Semi-Parametric Model"""
    # LightGBM parameters
    lgb_params: Dict = None
    lgb_num_rounds: int = 100
    lgb_early_stopping_rounds: int = 10
    
    # Elasticity model parameters
    regularization_lambda: float = 0.5
    convergence_tolerance: float = 1e-6
    
    # Data split
    validation_split: float = 0.2
    random_state: int = 42
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
    
    @classmethod
    def from_config_file(cls, config_path: str = "config.json"):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_config = config.get('semi_parametric_model', {})
            return cls(**model_config)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()


class SemiParametricDemandModel:
    """
    Semi-Parametric Model for demand prediction without normal channel sales
    
    Model equation:
    E[ln(Y)|r, L] = g(r; L, θ) + h(r_o, x)
    
    Where:
    - Y: sales in markdown channel
    - r: price ratio (actual_price / base_price)
    - r_o: reference price ratio
    - L: category encoding vector
    - θ: price elasticity parameters
    - h(r_o, x): base sales forecast
    - g(r; L, θ): price elasticity adjustment
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the semi-parametric model"""
        self.config = config or ModelConfig()
        
        # Price elasticity parameters
        self.theta = None  # [θ_1, θ_kan5_1, ..., θ_kan5_n]
        
        # Base forecasting model (LightGBM)
        self.base_model = None
        self.scaler = StandardScaler()
        
        # Category mappings
        self.kan5_mapping = {}  # Map KAN5 category names to indices
        self.n_kan5 = 0
        
        # Model performance metrics
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Store feature importance
        self.feature_importance = None
        
    def _build_category_mappings(self, data: pd.DataFrame):
        """Build mappings for KAN5 categories to indices"""
        if 'kan5' not in data.columns:
            logger.warning("No 'kan5' column found, skipping category mapping")
            return
            
        unique_kan5 = data['kan5'].unique()
        self.kan5_mapping = {cat: i for i, cat in enumerate(unique_kan5)}
        self.n_kan5 = len(unique_kan5)
        logger.info(f"Found {self.n_kan5} KAN5 categories")
    
    def _create_category_encoding(self, kan5: str) -> np.ndarray:
        """Create one-hot encoding for KAN5 category"""
        if not self.kan5_mapping:
            return np.array([])
            
        L_i = np.zeros(self.n_kan5)
        if kan5 in self.kan5_mapping:
            L_i[self.kan5_mapping[kan5]] = 1
        return L_i
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for base model"""
        feature_cols = []
        
        # Core features
        core_features = [
            'base_price_ratio',  # Reference price ratio
            'inventory',
            'days_to_expiry',
            'historical_avg_sales',
            'sales_trend'
        ]
        
        # Time features
        time_features = [
            'day_of_week',
            'month',
            'week_of_month',
            'is_weekend',
            'is_holiday'
        ]
        
        # Lagged sales features
        lag_features = [f'sales_lag_{lag}' for lag in [1, 3, 7, 14, 28]]
        
        # Competition features
        competition_features = [
            'competitor_avg_price_ratio',
            'competitor_min_price_ratio'
        ]
        
        # Combine all features
        all_features = core_features + time_features + lag_features + competition_features
        
        # Filter to only include existing columns
        feature_cols = [col for col in all_features if col in data.columns]
        
        if not feature_cols:
            # Fallback: use all numeric columns except target and identifiers
            exclude_cols = ['sales', 'sku_id', 'kan5', 'price_ratio', 'actual_price']
            feature_cols = [col for col in data.columns 
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])]
        
        logger.info(f"Using {len(feature_cols)} features for base model")
        return data[feature_cols] if feature_cols else pd.DataFrame(np.ones((len(data), 1)))
    
    def fit(self, data: pd.DataFrame):
        """
        Fit the semi-parametric model
        
        Required columns:
        - sku_id: SKU identifier
        - base_price: Base price for each SKU
        - actual_price: Actual selling price
        - price_ratio: actual_price / base_price (will be computed if not present)
        - sales: Actual sales
        - kan5: KAN5 category (optional)
        
        Optional columns:
        - Various features for base model
        """
        logger.info("Starting model training...")
        
        # Compute price ratio if not present
        if 'price_ratio' not in data.columns:
            if 'actual_price' in data.columns and 'base_price' in data.columns:
                data['price_ratio'] = data['actual_price'] / data['base_price']
            else:
                raise ValueError("Need either 'price_ratio' or both 'actual_price' and 'base_price'")
        
        # Compute base price ratio (historical average) for each SKU
        data['base_price_ratio'] = data.groupby('sku_id')['price_ratio'].transform('mean')
        
        # Build category mappings if available
        self._build_category_mappings(data)
        
        # Split data for validation
        if self.config.validation_split > 0:
            train_data, val_data = train_test_split(
                data, 
                test_size=self.config.validation_split,
                random_state=self.config.random_state
            )
            logger.info(f"Training set: {len(train_data)} samples, Validation set: {len(val_data)} samples")
        else:
            train_data = data
            val_data = None
        
        # Step 1: Train base forecasting model h(r_o, x)
        self._train_base_model(train_data, val_data)
        
        # Step 2: Estimate price elasticity parameters θ
        self._estimate_price_elasticity(train_data)
        
        # Step 3: Evaluate on validation set if available
        if val_data is not None:
            self._evaluate_model(val_data)
        
        logger.info("Model training completed")
        return self
    
    def _train_base_model(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None):
        """
        Train the non-parametric base model h(r_o, x)
        Predicts: ln(Y) at base price ratio r_o
        """
        logger.info("Training base forecasting model with LightGBM...")
        
        # Prepare features
        X_train = self._prepare_features(train_data)
        
        # Target: log sales (absolute, not ratio)
        epsilon = 1e-8
        y_train = np.log(train_data['sales'] + epsilon)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
        
        # Prepare validation set if available
        valid_sets = [lgb_train]
        valid_names = ['train']
        
        if val_data is not None:
            X_val = self._prepare_features(val_data)
            y_val = np.log(val_data['sales'] + epsilon)
            X_val_scaled = self.scaler.transform(X_val)
            lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)
            valid_sets.append(lgb_val)
            valid_names.append('valid')
        
        # Train LightGBM model
        self.base_model = lgb.train(
            self.config.lgb_params,
            lgb_train,
            num_boost_round=self.config.lgb_num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.config.lgb_early_stopping_rounds),
                lgb.log_evaluation(period=10)
            ]
        )
        
        # Store feature importance
        importance = self.base_model.feature_importance(importance_type='gain')
        feature_names = [f'feature_{i}' for i in range(len(importance))]
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Calculate training metrics
        y_pred_train = self.base_model.predict(X_train_scaled, num_iteration=self.base_model.best_iteration)
        mse = np.mean((y_train - y_pred_train) ** 2)
        mae = np.mean(np.abs(y_train - y_pred_train))
        
        self.training_metrics['base_model_mse'] = mse
        self.training_metrics['base_model_mae'] = mae
        
        logger.info(f"Base model trained - MSE: {mse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Best iteration: {self.base_model.best_iteration}")
    
    def _estimate_price_elasticity(self, data: pd.DataFrame):
        """
        Estimate price elasticity parameters using regularized least squares
        Model: g(r; L, θ) = (θ_1 + θ_2^T * L) * ln(r)
        """
        logger.info("Estimating price elasticity parameters...")
        
        n_samples = len(data)
        n_params = 1 + self.n_kan5  # θ_1 + KAN5 category parameters
        
        # Design matrix
        X = np.zeros((n_samples, n_params))
        y = np.zeros(n_samples)
        
        for idx, (_, row) in enumerate(data.iterrows()):
            # Create augmented L vector [1, L_i]
            L_i = self._create_category_encoding(row.get('kan5', ''))
            L_aug = np.concatenate([[1], L_i])
            
            # Feature vector: L_aug * ln(r)
            ln_r = np.log(row['price_ratio'] + 1e-8)
            X[idx, :n_params] = L_aug * ln_r
            
            # Target: ln(Y)
            epsilon = 1e-8
            y[idx] = np.log(row['sales'] + epsilon)
        
        # Solve regularized least squares
        # min ||y - X*θ||^2 + λ*||θ||^2
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add L2 regularization
        reg_matrix = self.config.regularization_lambda * np.eye(n_params)
        
        # Solve normal equations
        try:
            self.theta = np.linalg.solve(XtX + reg_matrix, Xty)
        except np.linalg.LinAlgError:
            logger.warning("Matrix singular, using pseudo-inverse")
            self.theta = np.linalg.pinv(XtX + reg_matrix) @ Xty
        
        # Calculate training metrics
        y_pred = X @ self.theta
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        
        self.training_metrics['elasticity_mse'] = mse
        self.training_metrics['elasticity_mae'] = mae
        
        # Log elasticity estimates
        base_elasticity = self.theta[0]
        logger.info(f"Base price elasticity: {base_elasticity:.3f}")
        logger.info(f"Elasticity model - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Show KAN5-specific elasticities if available
        if self.n_kan5 > 0:
            for kan5, idx in self.kan5_mapping.items():
                elasticity = base_elasticity + self.theta[1 + idx]
                logger.info(f"  KAN5 '{kan5}' total elasticity: {elasticity:.3f}")
    
    def _evaluate_model(self, val_data: pd.DataFrame):
        """Evaluate model on validation data"""
        logger.info("Evaluating model on validation set...")
        
        predictions = []
        actuals = []
        
        for _, row in val_data.iterrows():
            pred = self.predict_counterfactual_demand(
                target_price_ratio=row['price_ratio'],
                base_price_ratio=row.get('base_price_ratio', row['price_ratio']),
                base_sales=row['sales'],
                kan5=row.get('kan5', ''),
                features=row.to_dict()
            )
            predictions.append(pred)
            actuals.append(row['sales'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mae = np.mean(np.abs(actuals - predictions))
        
        self.validation_metrics = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Validation metrics - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    def predict_counterfactual_demand(
        self,
        target_price_ratio: float,
        base_price_ratio: float,
        base_sales: float,
        kan5: str = '',
        features: Optional[Dict] = None
    ) -> float:
        """
        Predict counterfactual demand at a given price ratio
        
        Formula: Y(r) = Y_o * (r/r_o)^(θ_1 + θ_2^T * L)
        
        Args:
            target_price_ratio: Target price / base_price
            base_price_ratio: Reference price / base_price
            base_sales: Observed sales at base_price_ratio
            kan5: Category (optional)
            features: Additional features for base model (optional, not used in current implementation)
        
        Returns:
            Predicted demand at target price ratio
        """
        if self.theta is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Create augmented category encoding
        L_i = self._create_category_encoding(kan5)
        L_aug = np.concatenate([[1], L_i])
        
        # Calculate product-specific elasticity
        elasticity = np.dot(self.theta[:len(L_aug)], L_aug)
        
        # Apply counterfactual prediction formula
        epsilon = 1e-8
        price_ratio_change = (target_price_ratio + epsilon) / (base_price_ratio + epsilon)
        
        # Cap elasticity to prevent extreme predictions
        capped_elasticity = np.clip(elasticity, -5.0, 0.0)  # Elasticity between -5 and 0
        
        # Cap multiplier to prevent explosive growth
        demand_multiplier = np.power(price_ratio_change, capped_elasticity)
        demand_multiplier = np.clip(demand_multiplier, 0.01, 100)  # Multiplier between 0.01x and 100x
        
        demand = base_sales * demand_multiplier
        
        return max(0, demand)  # Ensure non-negative
    
    def predict_demand_curve(
        self,
        base_price_ratio: float,
        base_sales: float,
        kan5: str = '',
        price_ratio_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate demand curve for a range of price ratios
        
        Args:
            base_price_ratio: Reference price ratio
            base_sales: Sales at reference price ratio
            kan5: Category (optional)
            price_ratio_range: Array of price ratios to evaluate
        
        Returns:
            Tuple of (price_ratios, predicted_demands)
        """
        if price_ratio_range is None:
            # Default range from 50% to 120% of base price
            price_ratio_range = np.linspace(0.5, 1.2, 50)
        
        demands = []
        for ratio in price_ratio_range:
            demand = self.predict_counterfactual_demand(
                target_price_ratio=ratio,
                base_price_ratio=base_price_ratio,
                base_sales=base_sales,
                kan5=kan5
            )
            demands.append(demand)
        
        return price_ratio_range, np.array(demands)
    
    def get_elasticity(self, kan5: str = '') -> float:
        """
        Get price elasticity for given category
        
        Args:
            kan5: Category name
        
        Returns:
            Price elasticity value
        """
        if self.theta is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        L_i = self._create_category_encoding(kan5)
        L_aug = np.concatenate([[1], L_i])
        
        return np.dot(self.theta[:len(L_aug)], L_aug)
    
    def save_model(self, filepath: str):
        """Save model to file"""
        import pickle
        
        model_state = {
            'theta': self.theta,
            'kan5_mapping': self.kan5_mapping,
            'n_kan5': self.n_kan5,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'scaler': self.scaler,
            'base_model': self.base_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        self.theta = model_state['theta']
        self.kan5_mapping = model_state['kan5_mapping']
        self.n_kan5 = model_state['n_kan5']
        self.training_metrics = model_state['training_metrics']
        self.validation_metrics = model_state['validation_metrics']
        self.feature_importance = model_state.get('feature_importance')
        self.config = model_state['config']
        self.scaler = model_state['scaler']
        self.base_model = model_state['base_model']
        
        logger.info(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'sku_id': np.random.choice(['SKU1', 'SKU2', 'SKU3'], n_samples),
        'kan5': np.random.choice(['CAT1', 'CAT2'], n_samples),
        'base_price': np.random.uniform(50, 200, n_samples),
        'actual_price': np.random.uniform(40, 180, n_samples),
        'inventory': np.random.uniform(10, 100, n_samples),
        'days_to_expiry': np.random.uniform(1, 30, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples)
    })
    
    # Compute price ratio and sales (with price elasticity effect)
    sample_data['price_ratio'] = sample_data['actual_price'] / sample_data['base_price']
    # Simulate sales with elasticity = -1.5
    sample_data['sales'] = 100 * np.power(sample_data['price_ratio'], -1.5) + np.random.normal(0, 10, n_samples)
    sample_data['sales'] = sample_data['sales'].clip(lower=0)
    
    # Initialize and train model
    config = ModelConfig()
    model = SemiParametricDemandModel(config)
    model.fit(sample_data)
    
    # Test prediction
    test_pred = model.predict_counterfactual_demand(
        target_price_ratio=0.8,
        base_price_ratio=1.0,
        base_sales=100,
        kan5='CAT1'
    )
    print(f"Predicted demand at 80% of base price: {test_pred:.2f}")
    
    # Generate demand curve
    ratios, demands = model.predict_demand_curve(
        base_price_ratio=1.0,
        base_sales=100,
        kan5='CAT1'
    )
    print(f"Demand curve generated with {len(ratios)} points")