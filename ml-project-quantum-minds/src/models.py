# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple, Any
import numpy as np


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
# Class-based API (for clean model comparison in main.py)
# ============================================================

@dataclass
class BaseModel:
    """
    Minimal interface for model comparison.
    """
    add_intercept: bool = True
    w: Optional[np.ndarray] = None

    def _prep_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return _add_intercept(x) if self.add_intercept else x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

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
    lambda_: float = 1e-6

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegLogisticRegressionGDModel":
        tx = self._prep_x(x)
        y01 = _to_zero_one(y)
        d = tx.shape[1]
        w0 = np.zeros(d)
        self.w, _ = reg_logistic_regression(y01, tx, self.lambda_, w0, self.max_iters, self.gamma)
        return self


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
