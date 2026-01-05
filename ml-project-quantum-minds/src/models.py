# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from numba import jit
from sklearn.neural_network import MLPClassifier




# ============================================================
# Utilities
# ============================================================

def _as_1d(y: np.ndarray) -> np.ndarray:
    """Ensure y is shape (N,) float."""
    y = np.asarray(y)
    return y.reshape(-1).astype(float)


def _add_intercept(tx: np.ndarray) -> np.ndarray:
    """Add bias column of ones to tx."""
    tx = np.asarray(tx, dtype=float)
    return np.c_[np.ones((tx.shape[0], 1)), tx]


def _to_zero_one(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to {0,1}.
    Accepts {0,1} or {-1,1}. Returns float array in {0,1}.
    """
    y = _as_1d(y)
    uniq = set(np.unique(y).tolist())
    if uniq.issubset({0.0, 1.0}):
        return y
    if uniq.issubset({-1.0, 1.0}):
        return (y + 1.0) / 2.0
    raise ValueError(f"Unsupported label values for logistic regression: {sorted(uniq)}")


def _to_minus_one_plus_one(y01: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities/0-1 labels into {-1, 1}."""
    y01 = np.asarray(y01).reshape(-1)
    return np.where(y01 >= threshold, 1, -1).astype(int)


def sigmoid(t: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    pos = t >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[neg])
    out[neg] = exp_t / (1.0 + exp_t)
    return out


def batch_iter(
    y: np.ndarray,
    tx: np.ndarray,
    batch_size: int,
    num_batches: int = 1,
    shuffle: bool = True,
):
    """
    Minibatch iterator. Same spirit as your original helper, but parameterized.
    """
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=float)

    data_size = len(y)
    batch_size = int(min(max(1, batch_size), data_size))
    max_batches = int(data_size / batch_size)
    remainder = data_size - max_batches * batch_size

    if max_batches == 0:
        yield y, tx
        return

    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        end = start + batch_size
        yield y[start:end], tx[start:end]


def _mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """0.5 * mean squared error (same convention as your least_squares)."""
    y = _as_1d(y)
    e = y - tx @ w
    return 0.5 * float(np.mean(e**2))


def _logistic_loss(y01: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Stable binary cross entropy loss for y in {0,1}.
    Uses logaddexp for numerical stability:
      loss = mean( log(1+exp(z)) - y*z )
    """
    y01 = _as_1d(y01)
    z = tx @ w
    return float(np.mean(np.logaddexp(0.0, z) - y01 * z))


# ============================================================
# Original function-style API (kept for compatibility)
# ============================================================

def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least squares via pseudo-inverse. Returns (w_opt, mse)."""
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=float)

    w_opt = np.linalg.pinv(tx.T @ tx) @ (tx.T @ y)
    mse = _mse(y, tx, w_opt)
    return w_opt, mse


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """Linear regression via GD minimizing MSE. Returns (w, final_loss)."""
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    n = len(y)
    for _ in range(int(max_iters)):
        err = y - tx @ w
        grad = -(tx.T @ err) / n
        w -= float(gamma) * grad

    return w, _mse(y, tx, w)


def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    *,
    batch_size: int = 1,
) -> Tuple[np.ndarray, float]:
    """Linear regression via SGD minimizing MSE. Returns (w, final_loss)."""
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    max_iters = int(max_iters)
    gamma = float(gamma)

    for _ in range(max_iters):
        for mini_y, mini_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1, shuffle=True):
            err = mini_y - mini_tx @ w
            grad = -(mini_tx.T @ err) / len(err)
            w -= gamma * grad

    return w, _mse(y, tx, w)


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> Tuple[np.ndarray, float]:
    """Closed-form ridge regression. Returns (w_opt, mse_loss_only)."""
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=float)

    n, d = tx.shape
    lam = float(lambda_)
    a = tx.T @ tx + 2.0 * n * lam * np.eye(d)
    b = tx.T @ y
    w_opt = np.linalg.solve(a, b)
    mse_loss = _mse(y, tx, w_opt)
    return w_opt, mse_loss


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """
    Logistic regression via GD for y in {0,1} or {-1,1}.
    Returns (w, final_loss).
    """
    y01 = _to_zero_one(y)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    n = len(y01)
    for _ in range(int(max_iters)):
        p = sigmoid(tx @ w)
        grad = (tx.T @ (p - y01)) / n
        w -= float(gamma) * grad

    return w, _logistic_loss(y01, tx, w)


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """
    Regularized logistic regression via GD for y in {0,1} or {-1,1}.
    L2 penalty: lambda * ||w||^2 (bias is also penalized here; can be changed if desired).
    Returns (w, final_loss_without_penalty) to match your original behavior.
    """
    y01 = _to_zero_one(y)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    n = len(y01)
    lam = float(lambda_)
    for _ in range(int(max_iters)):
        p = sigmoid(tx @ w)
        grad = (tx.T @ (p - y01)) / n + 2.0 * lam * w
        w -= float(gamma) * grad

    # match your original: report unregularized logistic loss
    return w, _logistic_loss(y01, tx, w)

# ============================================================
# HPC / Numba Optimized Functions
# ============================================================

@jit(nopython=True, cache=True)
def _sigmoid_numba(t: np.ndarray) -> np.ndarray:
    """
    Numba-optimized sigmoid function.
    Compiled to machine code for maximum performance.
    """
    # Clip values to prevent overflow, compatible with Numba
    t = np.minimum(np.maximum(t, -500), 500)
    return 1.0 / (1.0 + np.exp(-t))

@jit(nopython=True, cache=True)
def _grad_descent_numba(y: np.ndarray, tx: np.ndarray, w: np.ndarray, 
                        lambda_: float, gamma: float, max_iters: int) -> np.ndarray:
    """
    Gradient Descent loop compiled with Numba JIT (Just-In-Time).
    Replaces the slow Python loop with optimized C-like machine code.
    """
    n = len(y)
    for _ in range(max_iters):
        z = tx @ w
        pred = 1.0 / (1.0 + np.exp(-z)) # Inline sigmoid for performance
        
        # Compute gradient with L2 penalty
        err = pred - y
        grad = (tx.T @ err) / n + 2.0 * lambda_ * w
        
        # Update weights
        w -= gamma * grad
        
    return w


# ============================================================
# Class-based API (for clean model comparison in main.py)
# ============================================================

@dataclass
class BaseModel(ABC):
    """
    Minimal interface for model comparison.
    """
    add_intercept: bool = True
    w: Optional[np.ndarray] = None

    def _prep_x(self, x: np.ndarray) -> np.ndarray:
        if not self.add_intercept: return x
        return np.c_[np.ones((x.shape[0], 1)), x]

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LeastSquaresModel(BaseModel):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LeastSquaresModel":
        tx = self._prep_x(x)
        self.w, _ = least_squares(_as_1d(y), tx)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        tx = self._prep_x(x)
        return tx @ self.w


@dataclass
class RidgeRegressionModel(BaseModel):
    lambda_: float = 1e-6

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        tx = self._prep_x(x)
        self.w, _ = ridge_regression(_as_1d(y), tx, self.lambda_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        tx = self._prep_x(x)
        return tx @ self.w


@dataclass
class MSEGDModel(BaseModel):
    max_iters: int = 1000
    gamma: float = 1e-2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MSEGDModel":
        tx = self._prep_x(x)
        d = tx.shape[1]
        w0 = np.zeros(d)
        self.w, _ = mean_squared_error_gd(_as_1d(y), tx, w0, self.max_iters, self.gamma)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        return self._prep_x(x) @ self.w


@dataclass
class MSESGDModel(BaseModel):
    max_iters: int = 10
    gamma: float = 1e-2
    batch_size: int = 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MSESGDModel":
        tx = self._prep_x(x)
        d = tx.shape[1]
        w0 = np.zeros(d)
        self.w, _ = mean_squared_error_sgd(
            _as_1d(y), tx, w0, self.max_iters, self.gamma, batch_size=self.batch_size
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        return self._prep_x(x) @ self.w


@dataclass
class LogisticRegressionGDModel(BaseModel):
    max_iters: int = 1000
    gamma: float = 1e-2
    threshold: float = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGDModel":
        tx = self._prep_x(x)
        y01 = _to_zero_one(y)
        d = tx.shape[1]
        w0 = np.zeros(d)
        self.w, _ = logistic_regression(y01, tx, w0, self.max_iters, self.gamma)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        tx = self._prep_x(x)
        return sigmoid(tx @ self.w)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _to_minus_one_plus_one(self.predict_proba(x), threshold=self.threshold)


@dataclass
class RegLogisticRegressionGDModel(LogisticRegressionGDModel):
    lambda_: float = 1e-2
    max_iters: int = 300
    gamma: float = 0.1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegLogisticRegressionGDModel":
        # Ensure float64 type for Numba compatibility
        tx = self._prep_x(x).astype(np.float64)
        
        # Convert labels {-1, 1} to {0, 1} and cast to float64
        y_01 = ((y + 1) / 2).astype(np.float64) if np.any(y == -1) else y.astype(np.float64)
        
        w0 = np.zeros(tx.shape[1], dtype=np.float64)
        
        # Call the JIT-compiled kernel (HPC)
        self.w = _grad_descent_numba(y_01, tx, w0, self.lambda_, self.gamma, self.max_iters)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None: 
            raise RuntimeError("Model not trained")
            
        tx = self._prep_x(x).astype(np.float64)
        # Use the Numba-optimized sigmoid
        prob = _sigmoid_numba(tx @ self.w)
        return np.where(prob >= 0.5, 1, -1)


@dataclass
class RandomForestModel(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 0
    # Random Forest does not require a bias/intercept column
    add_intercept: bool = False 
    clf: Optional[RandomForestClassifier] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        # Scikit-learn natively handles {-1, 1} labels, no conversion needed
        # We assume self._prep_x handles the intercept logic (returns x as-is because add_intercept is False)
        tx = self._prep_x(x)
        
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1 # Use all available CPU cores for performance
        )
        self.clf.fit(tx, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model has not been trained yet.")
        tx = self._prep_x(x)
        return self.clf.predict(tx)
    
@dataclass
class XGBoostModel(BaseModel):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 0
    add_intercept: bool = False
    clf: Optional[XGBClassifier] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        # Map labels from {-1, 1} to {0, 1} for XGBoost
        y_mapped = np.where(y == -1, 0, 1)

        self.clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.clf.fit(x, y_mapped)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model not trained.")
        # Predict 0/1 and map back to -1/1
        pred_01 = self.clf.predict(x)
        return np.where(pred_01 == 0, -1, 1)
    

@dataclass
class NeuralNetModel(BaseModel):
    """
    Week 10: Deep Learning / Multi-Layer Perceptron.
    """
    hidden_layer_sizes: Tuple[int, ...] = (100, 50)
    activation: str = 'relu'
    learning_rate_init: float = 0.001
    max_iter: int = 200
    random_state: int = 43504
    add_intercept: bool = False
    clf: Optional[MLPClassifier] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NeuralNetModel":
        # Scikit-learn requires {0, 1} or {-1, 1} (it handles mapping automatically)
        self.clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True, # Prevents overfitting (Week 10/11)
            validation_fraction=0.1
        )
        self.clf.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.clf is None: raise RuntimeError("Model not trained.")
        return self.clf.predict(x)


# ============================================================
# Simple model factory (helps main.py stay clean)
# ============================================================

MODEL_REGISTRY: Dict[str, Callable[..., BaseModel]] = {
    "least_squares": LeastSquaresModel,
    "ridge": RidgeRegressionModel,
    "mse_gd": MSEGDModel,
    "mse_sgd": MSESGDModel,
    "logreg_gd": LogisticRegressionGDModel,
    "reg_logreg_gd": RegLogisticRegressionGDModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "neural_net": NeuralNetModel,
}


def make_model(name: str, **kwargs) -> BaseModel:
    """
    Create a model from MODEL_REGISTRY.
    Example:
      model = make_model("ridge", lambda_=1e-3)
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
