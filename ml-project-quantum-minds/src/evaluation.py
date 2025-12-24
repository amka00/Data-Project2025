# src/evaluation.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Callable, Optional

import numpy as np

from src.models import BaseModel, make_model, sigmoid  # sigmoid only used for the "best-F1 reg log" routine


# ============================================================
# I/O helpers
# ============================================================

def create_csv_submission(ids, y_pred, name: str | Path):
    """
    Create a submission csv with columns: Id, Prediction.
    y_pred must contain only -1 and 1.
    """
    ids = np.asarray(ids).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if ids.shape[0] != y_pred.shape[0]:
        raise ValueError("ids and y_pred must have the same length.")

    if not np.all(np.isin(y_pred, [-1, 1])):
        raise ValueError("y_pred can only contain values -1, 1")

    name = Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)

    with name.open("w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def save_json(obj: Any, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Label utilities + metrics
# ============================================================

def to01(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to {0,1}. Accepts {0,1} or {-1,1}.
    """
    y = np.asarray(y).reshape(-1)
    uniq = set(np.unique(y).tolist())
    if uniq.issubset({0, 1}):
        return y.astype(int)
    if uniq.issubset({-1, 1}):
        return ((y + 1) // 2).astype(int)
    raise ValueError(f"Unsupported labels: {sorted(uniq)}")


def to_pm1(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to {-1,1}. Accepts {0,1} or {-1,1}.
    """
    y = np.asarray(y).reshape(-1)
    uniq = set(np.unique(y).tolist())
    if uniq.issubset({-1, 1}):
        return y.astype(int)
    if uniq.issubset({0, 1}):
        return (2 * y - 1).astype(int)
    raise ValueError(f"Unsupported labels: {sorted(uniq)}")


def precision_recall_f1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes precision/recall/f1/accuracy for binary classification.
    Treats label "1" as positive.
    """
    y_true01 = to01(y_true)
    y_pred01 = to01(y_pred)

    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))
    tn = int(np.sum((y_true01 == 0) & (y_pred01 == 0)))

    eps = 1e-15
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================
# Splitting + balancing
# ============================================================

def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    split_ratio: float = 0.9,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random split into train/val.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    cut = int(split_ratio * n)
    tr_idx = perm[:cut]
    va_idx = perm[cut:]
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def balance_binary(
    X: np.ndarray,
    y: np.ndarray,
    positive_ratio: float = 0.22,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance a binary dataset to reach approximately `positive_ratio` in the returned sample.
    Keeps all positives; samples negatives (without replacement if possible) and oversamples positives if needed.

    Works for y in {-1,1} or {0,1}.
    """
    rng = np.random.default_rng(seed)
    y_pm1 = to_pm1(y)

    pos_idx = np.where(y_pm1 == 1)[0]
    neg_idx = np.where(y_pm1 == -1)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # nothing to balance
        return X, y

    n_pos = len(pos_idx)
    n_total_target = int(np.ceil(n_pos / max(1e-12, positive_ratio)))
    n_neg_target = max(0, n_total_target - n_pos)

    # sample negatives
    if n_neg_target <= len(neg_idx):
        neg_sample = rng.choice(neg_idx, size=n_neg_target, replace=False)
        pos_sample = pos_idx
    else:
        # use all negatives and oversample positives to hit desired total
        neg_sample = neg_idx
        extra_pos = n_total_target - len(neg_sample)
        pos_sample = rng.choice(pos_idx, size=extra_pos, replace=True)

    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ============================================================
# Prediction adaptation (regression vs classification)
# ============================================================

def predict_labels(model: BaseModel, X: np.ndarray, *, regression_threshold: float = 0.0) -> np.ndarray:
    """
    Convert model outputs into {-1,1} labels:
    - If model.predict returns labels already (0/1 or -1/1), normalize to -1/1.
    - Otherwise treat output as a score and threshold at regression_threshold.
    """
    raw = model.predict(X)
    raw = np.asarray(raw).reshape(-1)

    uniq = set(np.unique(raw).tolist())
    if uniq.issubset({-1, 1}):
        return raw.astype(int)
    if uniq.issubset({0, 1}):
        return (2 * raw - 1).astype(int)

    # regression-style scores
    return np.where(raw >= regression_threshold, 1, -1).astype(int)


# ============================================================
# Model evaluation loop
# ============================================================

def evaluate_one(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    balance: bool = True,
    positive_ratio: float = 0.22,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Fit on train (optionally balanced), evaluate on val.
    Returns metrics dict.
    """
    if balance:
        Xb, yb = balance_binary(X_train, y_train, positive_ratio=positive_ratio, seed=seed)
    else:
        Xb, yb = X_train, y_train

    model.fit(Xb, yb)
    y_pred = predict_labels(model, X_val)
    metrics = precision_recall_f1_accuracy(y_val, y_pred)
    return metrics


def evaluate_many(
    model_specs: Dict[str, Callable[[], BaseModel]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    balance: bool = True,
    positive_ratio: float = 0.22,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate many models. Returns dict: model_name -> metrics
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, ctor in model_specs.items():
        model = ctor()
        results[name] = evaluate_one(
            model, X_train, y_train, X_val, y_val,
            balance=balance, positive_ratio=positive_ratio, seed=seed
        )
    return results


def best_by_f1(results: Dict[str, Dict[str, Any]]) -> str:
    return max(results.keys(), key=lambda k: results[k]["f1"])


# ============================================================
# Special: "best-F1" regularized logistic (your notebook variation)
# This uses validation during training to keep the best weights.
# ============================================================

def reg_logistic_best_f1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_iters: int = 150,
    gamma: float = 0.1,
    lambda_: float = 0.01,
    tol: float = 1e-8,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Re-implements your "logistic_regression_variation" cleanly:
    - adds intercept
    - uses y in {0,1}
    - regularizes w[1:] (no penalty on bias)
    - tracks best validation F1 and returns best weights

    Returns:
      w_best (D+1,),
      metrics_best (dict)
    """
    rng = np.random.default_rng(seed)

    # labels to {0,1}
    ytr01 = to01(y_train).astype(float)
    yva01 = to01(y_val).astype(float)

    # add bias
    tx_tr = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    tx_va = np.c_[np.ones((X_val.shape[0], 1)), X_val]

    w = np.zeros(tx_tr.shape[1], dtype=float)

    best_f1 = -1.0
    w_best = w.copy()
    prev_loss = None

    for _ in range(int(max_iters)):
        z = tx_tr @ w
        p = sigmoid(z)

        # gradient (no reg on bias)
        grad = (tx_tr.T @ (p - ytr01)) / len(ytr01)
        grad[1:] += float(lambda_) * w[1:]

        w -= float(gamma) * grad

        # early stop based on change in loss
        # stable logistic loss + reg term
        loss = float(np.mean(np.logaddexp(0.0, z) - ytr01 * z) + 0.5 * float(lambda_) * np.sum(w[1:] ** 2))
        if prev_loss is not None and abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

        # validate F1
        p_val = sigmoid(tx_va @ w)
        y_pred01 = (p_val >= 0.5).astype(int)
        y_pred_pm1 = to_pm1(y_pred01)

        metrics = precision_recall_f1_accuracy(y_val, y_pred_pm1)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            w_best = w.copy()

    # final metrics with best weights
    p_val_best = sigmoid(tx_va @ w_best)
    y_pred01_best = (p_val_best >= 0.5).astype(int)
    y_pred_pm1_best = to_pm1(y_pred01_best)
    metrics_best = precision_recall_f1_accuracy(y_val, y_pred_pm1_best)
    return w_best, metrics_best


# ============================================================
# Optional plotting (matplotlib only)
# ============================================================

def plot_f1_bar(results: Dict[str, Dict[str, Any]], title: str = "F1 scores"):
    """
    Simple bar plot of F1. Uses matplotlib only.
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    f1s = [results[k]["f1"] for k in names]

    plt.figure(figsize=(10, 5))
    plt.bar(names, f1s)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("F1")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pearson_correlations(scores, title: str = "Pearson Correlation Coefficients"):
    """
    Simple bar plot for Pearson correlations (matplotlib only).
    """
    scores = np.asarray(scores).ravel()
    x = np.arange(len(scores))

    plt.figure(figsize=(12, 6))
    plt.bar(x, scores)
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Correlation coefficient (r)")
    plt.tight_layout()
    plt.show()
