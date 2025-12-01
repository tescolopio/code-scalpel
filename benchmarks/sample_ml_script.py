"""
Sample Machine Learning script for benchmarking CodeAnalyzer.

This simulates a real-world ML training pipeline with:
- Data preprocessing
- Model training
- Evaluation
- Some intentional dead code
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class ModelType(Enum):
    """Types of models supported."""
    LINEAR = 'linear'
    POLYNOMIAL = 'polynomial'
    NEURAL = 'neural'


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    num_samples: int
    num_features: int
    mean: List[float]
    std: List[float]
    min_val: List[float]
    max_val: List[float]


class DataPreprocessor:
    """Preprocess data for ML training."""
    
    def __init__(self, normalize: bool = True, handle_missing: bool = True):
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'DataPreprocessor':
        """Fit preprocessor to data."""
        if self.handle_missing:
            X = self._fill_missing(X)
        
        if self.normalize:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()
        
        if self.handle_missing:
            X = self._fill_missing(X)
        
        if self.normalize:
            X = (X - self.mean_) / self.std_
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def _fill_missing(self, X: np.ndarray) -> np.ndarray:
        """Fill missing values with column means."""
        X = X.copy()
        for i in range(X.shape[1]):
            col = X[:, i]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmean(col)
        return X
    
    def get_stats(self, X: np.ndarray) -> DatasetStats:
        """Get dataset statistics."""
        return DatasetStats(
            num_samples=X.shape[0],
            num_features=X.shape[1],
            mean=list(np.mean(X, axis=0)),
            std=list(np.std(X, axis=0)),
            min_val=list(np.min(X, axis=0)),
            max_val=list(np.max(X, axis=0))
        )
    
    # Dead code: unused method
    def deprecated_normalize(self, X: np.ndarray) -> np.ndarray:
        """Old normalization method - deprecated."""
        return (X - X.min()) / (X.max() - X.min())


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pass


class LinearRegressor(BaseModel):
    """Simple linear regression model."""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.training_history: List[float] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressor':
        """Train the model using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self._forward(X)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Track loss
            loss = self._mse_loss(y, y_pred)
            self.training_history.append(loss)
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        return self._forward(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return np.dot(X, self.weights) + self.bias
    
    def _mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error."""
        return np.mean((y_true - y_pred) ** 2)
    
    # Dead code: unused method
    def unused_regularization(self, lambda_: float = 0.01) -> float:
        """Calculate L2 regularization term - not used."""
        if self.weights is None:
            return 0.0
        return lambda_ * np.sum(self.weights ** 2)


class NeuralNetwork(BaseModel):
    """Simple neural network with one hidden layer."""
    
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.01, 
                 epochs: int = 100):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        
        self.training_history: List[float] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetwork':
        """Train the neural network."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.W1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        
        y = y.reshape(-1, 1)
        
        for epoch in range(self.epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            y_pred = z2
            
            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            self.training_history.append(loss)
            
            # Backward pass
            m = n_samples
            dz2 = y_pred - y
            dW2 = (1 / m) * np.dot(a1.T, dz2)
            db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self._relu_derivative(z1)
            dW1 = (1 / m) * np.dot(X.T, dz1)
            db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2.flatten()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)


# Dead code: unused class
class DeprecatedOptimizer:
    """Old optimizer class - no longer used."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def step(self, gradients: np.ndarray) -> np.ndarray:
        """Compute update step."""
        self.step_count += 1
        return -self.learning_rate * gradients
    
    def reset(self):
        """Reset optimizer state."""
        self.step_count = 0


def train_test_split(X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def cross_validate(model: BaseModel, X: np.ndarray, y: np.ndarray, 
                   n_folds: int = 5) -> Dict[str, float]:
    """Perform k-fold cross-validation."""
    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    scores = []
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        
        # Create validation set
        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        
        # Create training set
        X_train = np.vstack([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Train and evaluate
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10, 
                           noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    X = np.random.randn(n_samples, n_features)
    
    # True weights
    true_weights = np.random.randn(n_features)
    
    # Generate targets
    y = np.dot(X, true_weights) + np.random.randn(n_samples) * noise
    
    return X, y


# Dead code: unused function
def unused_feature_selection(X: np.ndarray, y: np.ndarray, 
                             k: int = 5) -> List[int]:
    """Select top k features by correlation - not used."""
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((i, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in correlations[:k]]


def evaluate_model(model: BaseModel, X_test: np.ndarray, 
                   y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate model on test data."""
    y_pred = model.predict(X_test)
    
    # Mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(y_test - y_pred))
    
    # R-squared
    r2 = model.score(X_test, y_test)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def save_model(model: BaseModel, filepath: str) -> None:
    """Save model to file (simulated)."""
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> BaseModel:
    """Load model from file (simulated)."""
    logger.info(f"Model loaded from {filepath}")
    return LinearRegressor()


def run_training_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    """Run complete training pipeline."""
    logger.info("Starting training pipeline...")
    start_time = time.time()
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=1000, n_features=10)
    
    # Preprocess
    preprocessor = DataPreprocessor(normalize=True)
    X = preprocessor.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=config.validation_split)
    
    # Train model
    model = LinearRegressor(learning_rate=config.learning_rate, 
                           epochs=config.epochs)
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    elapsed = time.time() - start_time
    
    results = {
        'metrics': metrics,
        'training_time': elapsed,
        'n_epochs': config.epochs,
        'final_loss': model.training_history[-1] if model.training_history else None
    }
    
    logger.info(f"Training complete in {elapsed:.2f}s")
    logger.info(f"Test R²: {metrics['r2']:.4f}")
    
    return results


# ----- Main Entry Point -----

if __name__ == '__main__':
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    results = run_training_pipeline(config)
    print(f"\nFinal Results:")
    print(f"  MSE: {results['metrics']['mse']:.4f}")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  MAE: {results['metrics']['mae']:.4f}")
    print(f"  R²: {results['metrics']['r2']:.4f}")
