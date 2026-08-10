"""
Credit Risk Prediction Model Training

Input:
    data/processed/loans_clean.parquet

Output:
    ml/models/credit_risk_model.joblib
"""

import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/processed/loans_clean.parquet"
MODEL_PATH = "ml/models/credit_risk_model.joblib"

TARGET = "default"


def load_data():
    """Load the processed dataset."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    print("Loading processed dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def prepare_data(df):
    """Separate features and target."""

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def build_pipeline(X):
    """Build preprocessing + Logistic Regression pipeline."""

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

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n========== MODEL RESULTS ==========")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"ROC-AUC:  {auc:.4f}")

    print("\n========== CONFUSION MATRIX ==========")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Good",
                "Default",
            ],
        )
    )


def save_model(model):
    """Save trained model."""

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nModel saved to:")

    print(MODEL_PATH)


def main():

    print("=" * 50)
    print("Credit Risk Model Training")
    print("=" * 50)

    df = load_data()

    X, y = prepare_data(df)

    print("\n========== TARGET DISTRIBUTION ==========")

    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n========== DATA SPLIT ==========")

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")

    pipeline = build_pipeline(X_train)

    print("\nTraining Logistic Regression...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    save_model(pipeline)


if __name__ == "__main__":
    main()