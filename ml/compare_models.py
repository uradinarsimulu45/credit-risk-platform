"""
Day 5 - Credit Risk Model Comparison

Compares:
1. Logistic Regression
2. Random Forest

Dataset:
data/processed/loans_clean.parquet
"""

import os
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/processed/loans_clean.parquet"
MODEL_PATH = "ml/models/best_credit_risk_model.joblib"

TARGET = "default"


def load_data():
    print("Loading processed dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def prepare_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def create_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return results


def main():

    print("=" * 60)
    print("DAY 5 - CREDIT RISK MODEL COMPARISON")
    print("=" * 60)

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", f"{len(X_train):,}")
    print("Testing samples:", f"{len(X_test):,}")

    preprocessor = create_preprocessor(X_train)

    # -----------------------------
    # Logistic Regression
    # -----------------------------

    print("\n" + "=" * 60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("=" * 60)

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    start = time.time()

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_time = time.time() - start

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test,
    )

    print(
        f"Training time: {logistic_time:.2f} seconds"
    )

    # -----------------------------
    # Random Forest
    # -----------------------------

    print("\n" + "=" * 60)
    print("MODEL 2: RANDOM FOREST")
    print("=" * 60)

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    start = time.time()

    random_forest_model.fit(
        X_train,
        y_train,
    )

    random_forest_time = time.time() - start

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test,
    )

    print(
        f"Training time: {random_forest_time:.2f} seconds"
    )

    # -----------------------------
    # Compare results
    # -----------------------------

    results = pd.DataFrame(
        [
            logistic_results,
            random_forest_results,
        ]
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Choose based on ROC-AUC
    best_index = results["roc_auc"].idxmax()

    best_name = results.loc[
        best_index,
        "model",
    ]

    if best_name == "Logistic Regression":
        best_model = logistic_model
    else:
        best_model = random_forest_model

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Selected model: {best_name}")

    print(
        f"ROC-AUC: "
        f"{results.loc[best_index, 'roc_auc']:.4f}"
    )

    # Save best model
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    print("\nBest model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()