
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Teen_Stress.csv"
MODEL_PATH = BASE_DIR / "trained_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Encoding — kept consistent with the original code
    df["gender"] = df["gender"].map({
        "male": 1,
        "female": 0
    })

    df["social_interaction_level"] = df[
        "social_interaction_level"
    ].map({
        "low": 0,
        "medium": 1,
        "high": 2
    })

    df = pd.get_dummies(
        df,
        columns=["platform_usage"],
        dtype=int
    )

    # Features and target
    x = df.drop("depression_label", axis=1)
    y = df["depression_label"]

    # Train test split — same settings as original code
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42
    )

    # Random Forest Model
    rf = RandomForestClassifier(
        random_state=42
    )

    # Hyperparameter grid — same settings as original code
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    # GridSearchCV
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Train model
    print("Starting GridSearchCV: 81 parameter combinations × 5 folds = 405 fits.")
    grid.fit(x_train, y_train)

    # Best model
    best_model = grid.best_estimator_

    # Prediction
    y_pred = best_model.predict(x_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Store the exact training feature order. The app must use this order.
    model_bundle = {
        "model": best_model,
        "feature_columns": x.columns.tolist(),
        "target_classes": [int(label) for label in best_model.classes_],
    }
    joblib.dump(model_bundle, MODEL_PATH)

    # Feature importance from the fitted Random Forest
    importances = [
        {"feature": str(name), "importance": float(value)}
        for name, value in zip(
            x.columns,
            best_model.feature_importances_
        )
    ]
    importances.sort(key=lambda item: item["importance"], reverse=True)

    class_counts = {
        str(label): int(count)
        for label, count in y.value_counts().sort_index().items()
    }

    metrics = {
        "best_params": grid.best_params_,
        "cv_accuracy": float(grid.best_score_),
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": importances,
        "dataset_records": int(len(df)),
        "feature_count": int(x.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "class_counts": class_counts,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_labels": {"0": "No", "1": "Yes"},
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    # Display results, matching the original code's output
    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest Cross Validation Score:")
    print(grid.best_score_)

    print("\nTest Accuracy:")
    print(acc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nSaved model bundle:")
    print(MODEL_PATH)

    print("\nSaved metrics:")
    print(METRICS_PATH)


if __name__ == "__main__":
    main()
