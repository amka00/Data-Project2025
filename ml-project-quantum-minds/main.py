# main.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.data_loader import load_and_preprocess
from src.models import make_model
from src.evaluation import (
    train_val_split,
    balance_binary,
    evaluate_many,
    best_by_f1,
    save_json,
    create_csv_submission,
    plot_f1_bar,
    reg_logistic_best_f1,
    precision_recall_f1_accuracy,
    to_pm1,
)
from src.models import sigmoid


def _resolve_data_paths():
    """
    Try common folder layouts:
      - data/raw/{x_train.csv, x_test.csv, y_train.csv, variable_descriptions.txt}
      - dataset/{x_train.csv, x_test.csv, y_train.csv} + variable_descriptions.txt in root
      - dataset/ + dataset/variable_descriptions.txt
    """
    print("CWD:", Path.cwd())
    print("main.py location:", Path(__file__).resolve())
    print("Project root guess:", Path(__file__).resolve().parent)
    candidates = [
        (Path("data/raw"), Path("data/raw/variable_descriptions.txt")),
        (Path("dataset"), Path("variable_descriptions.txt")),
        (Path("dataset"), Path("dataset/variable_descriptions.txt")),
        (Path("data"), Path("data/variable_descriptions.txt")),
    ]

    for data_dir, desc_path in candidates:
        if (data_dir / "x_train.csv").exists() and (data_dir / "x_test.csv").exists() and (data_dir / "y_train.csv").exists():
            if desc_path.exists():
                return data_dir, desc_path

    raise FileNotFoundError(
        "Could not locate data files. Expected x_train.csv/x_test.csv/y_train.csv and variable_descriptions.txt "
        "in either data/raw/ or dataset/."
    )


def main():
    # ---------------------------
    # Config (edit here)
    # ---------------------------
    RANDOM_SEED = 43504
    TRAIN_SPLIT_RATIO = 0.90
    BALANCE_TRAIN = True
    TRAIN_POSITIVE_RATIO = 0.22

    PEARSON_R_MIN = 0.0

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)

    # ---------------------------
    # Load + preprocess
    # ---------------------------
    data_dir, desc_path = _resolve_data_paths()

    X_train, X_test, y_train, train_ids, test_ids, state, pearson_scores = load_and_preprocess(
        data_dir=data_dir,
        variable_description_path=desc_path,
        pearson_r_min=PEARSON_R_MIN,
    )

    # ---------------------------
    # Train/Val split
    # ---------------------------
    X_tr, y_tr, X_val, y_val = train_val_split(
        X_train, y_train, split_ratio=TRAIN_SPLIT_RATIO, seed=RANDOM_SEED
    )

    # ---------------------------
    # Define model candidates
    # (easy to extend later with xgboost/trees/rf)
    # ---------------------------
    model_specs = {
        "ridge(lambda=1e-2)": lambda: make_model("ridge", lambda_=1e-2),
        "mse_gd(it=150,g=0.1)": lambda: make_model("mse_gd", max_iters=150, gamma=0.1),
        "mse_sgd(it=20,g=0.05,b=8)": lambda: make_model("mse_sgd", max_iters=20, gamma=0.05, batch_size=8),
        "logreg_gd(it=300,g=0.1)": lambda: make_model("logreg_gd", max_iters=300, gamma=0.1),
        "reg_logreg_gd(l=1e-2,it=300,g=0.1)": lambda: make_model("reg_logreg_gd", lambda_=1e-2, max_iters=300, gamma=0.1),
        "least_squares": lambda: make_model("least_squares"),
    }

    # ---------------------------
    # Evaluate standard models
    # ---------------------------
    results = evaluate_many(
        model_specs, X_tr, y_tr, X_val, y_val,
        balance=BALANCE_TRAIN,
        positive_ratio=TRAIN_POSITIVE_RATIO,
        seed=RANDOM_SEED,
    )

    # ---------------------------
    # Also evaluate your "best-F1 reg logistic" routine (uses val during training)
    # ---------------------------
    if BALANCE_TRAIN:
        X_tr_bal, y_tr_bal = balance_binary(X_tr, y_tr, positive_ratio=TRAIN_POSITIVE_RATIO, seed=RANDOM_SEED)
    else:
        X_tr_bal, y_tr_bal = X_tr, y_tr

    w_best, metrics_best = reg_logistic_best_f1(
        X_tr_bal, y_tr_bal, X_val, y_val,
        max_iters=150, gamma=0.1, lambda_=0.01, tol=1e-8, seed=RANDOM_SEED
    )
    results["reg_logreg_bestF1(it=150,g=0.1,l=0.01)"] = metrics_best

    # ---------------------------
    # Pick best and refit on full training
    # ---------------------------
    best_name = best_by_f1(results)

    # Refit:
    # - If best is our special routine, refit it on full train with an internal split (small val) OR just train normally.
    # Here: we refit normal reg_logreg_gd on full balanced set with same hyperparams for simplicity.
    if best_name.startswith("reg_logreg_bestF1"):
        # Use same hyperparams as bestF1 routine, but train on full (balanced) with reg_logreg_gd model.
        best_model = make_model("reg_logreg_gd", lambda_=0.01, max_iters=300, gamma=0.1)
    else:
        best_model = model_specs[best_name]()

    if BALANCE_TRAIN:
        X_full, y_full = balance_binary(X_train, y_train, positive_ratio=TRAIN_POSITIVE_RATIO, seed=RANDOM_SEED)
    else:
        X_full, y_full = X_train, y_train

    best_model.fit(X_full, y_full)

    # Predict test labels in {-1,1}
    y_test_pred = best_model.predict(X_test)
    y_test_pred = np.asarray(y_test_pred).reshape(-1)

    # Some models output regression scores; convert to {-1,1}
    uniq = set(np.unique(y_test_pred).tolist())
    if not uniq.issubset({-1, 1}):
        # treat as score
        y_test_pred = np.where(y_test_pred >= 0.0, 1, -1).astype(int)

    # Save submission + metrics
    submission_path = RESULTS_DIR / "submission.csv"
    create_csv_submission(test_ids, y_test_pred, submission_path)

    metrics_path = RESULTS_DIR / "metrics.json"
    payload = {
        "best_model": best_name,
        "results": results,
        "data_dir": str(data_dir),
        "variable_descriptions": str(desc_path),
        "config": {
            "seed": RANDOM_SEED,
            "train_split_ratio": TRAIN_SPLIT_RATIO,
            "balance_train": BALANCE_TRAIN,
            "train_positive_ratio": TRAIN_POSITIVE_RATIO,
            "pearson_r_min": PEARSON_R_MIN,
        },
    }
    save_json(payload, metrics_path)

    # Optional: quick plot (comment out if you want CLI-only)
    # plot_f1_bar(results, title="Validation F1 scores")

    print(f"Best model: {best_name}")
    print(f"Saved: {submission_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
