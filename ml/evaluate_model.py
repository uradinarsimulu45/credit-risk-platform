"""
Day 6 - Credit Risk Model Evaluation & Explainability

Evaluates the Random Forest credit risk model.

Outputs:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Feature importance
- Evaluation report
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/loans_clean.parquet"
MODEL_PATH = "ml/models/best_credit_risk_model.joblib"

OUTPUT_DIR = "ml/evaluation"


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

    target = "default"

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Low Risk",
                "High Risk"
            ],
            zero_division=0
        )
    )

    return predictions, probabilities


def save_confusion_matrix(y_test, predictions):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Low Risk",
            "High Risk"
        ]
    )

    display.plot()

    plt.title("Credit Risk Confusion Matrix")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(path)

    plt.close()

    print(f"Saved: {path}")


def save_roc_curve(
    model,
    X_test,
    y_test
):

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(
        "Credit Risk ROC Curve"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )

    plt.savefig(path)

    plt.close()

    print(f"Saved: {path}")


def save_feature_importance(
    model
):

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        classifier = model.named_steps[
            "model"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importances = (
            classifier
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .head(20)
        )

        print(
            importance_df.to_string(
                index=False
            )
        )

        path = os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        )

        importance_df.to_csv(
            path,
            index=False
        )

        print(f"\nSaved: {path}")

        plt.figure(
            figsize=(10, 8)
        )

        importance_df.sort_values(
            "importance"
        ).plot(
            x="feature",
            y="importance",
            kind="barh",
            legend=False
        )

        plt.title(
            "Top 20 Credit Risk Features"
        )

        plt.xlabel(
            "Importance"
        )

        plt.tight_layout()

        chart_path = os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        )

        plt.savefig(
            chart_path
        )

        plt.close()

        print(
            f"Saved: {chart_path}"
        )

    except Exception as error:

        print(
            "Could not calculate feature importance:"
        )

        print(error)


def main():

    print("=" * 60)
    print("DAY 6 - CREDIT RISK MODEL EVALUATION")
    print("=" * 60)

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    print("\nLoading best model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Random Forest model loaded."
    )

    predictions, probabilities = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_confusion_matrix(
        y_test,
        predictions
    )

    save_roc_curve(
        model,
        X_test,
        y_test
    )

    save_feature_importance(
        model
    )

    print("\n" + "=" * 60)
    print("DAY 6 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()