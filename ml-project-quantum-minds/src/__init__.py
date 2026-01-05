# src/__init__.py
from __future__ import annotations

# Importation des fonctions clés pour faciliter l'accès depuis la racine du projet
from .data_loader import (
    load_and_preprocess,
    load_csv_data,
    PreprocessingState
)

from .models import (
    make_model,
    sigmoid,
    BaseModel,
    RegLogisticRegressionGDModel,
    RidgeRegressionModel
)

from .evaluation import (
    train_val_split,
    balance_binary,
    evaluate_many,
    best_by_f1,
    precision_recall_f1_accuracy,
    create_csv_submission,
    save_json
)

__all__ = [
    "load_and_preprocess",
    "load_csv_data",
    "PreprocessingState",
    "make_model",
    "sigmoid",
    "BaseModel",
    "RegLogisticRegressionGDModel",
    "RidgeRegressionModel",
    "train_val_split",
    "balance_binary",
    "evaluate_many",
    "best_by_f1",
    "precision_recall_f1_accuracy",
    "create_csv_submission",
    "save_json"
]