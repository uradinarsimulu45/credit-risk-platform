"""
LendingClub Credit Risk Data Preprocessing

Input:
    data/raw/loans.csv.gz

Output:
    data/processed/loans_clean.parquet
"""

import os
import numpy as np
import pandas as pd


RAW_PATH = "data/raw/loans.csv.gz"
OUTPUT_PATH = "data/processed/loans_clean.parquet"


DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
}

PAID_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}


FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
]

TARGET_COLUMN = "loan_status"


def load_dataset(path=RAW_PATH):
    """Load the required columns from the LendingClub dataset."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    print("Loading LendingClub dataset...")

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    df = pd.read_csv(
        path,
        usecols=lambda column: column in columns,
        low_memory=False,
        compression="gzip"
    )

    print(f"Loaded {len(df):,} rows.")

    return df


def create_target(df):
    """Keep finalized loans and create binary default target."""

    print("\nCreating binary target...")

    valid_statuses = DEFAULT_STATUSES | PAID_STATUSES

    df = df[df[TARGET_COLUMN].isin(valid_statuses)].copy()

    df["default"] = (
        df[TARGET_COLUMN]
        .isin(DEFAULT_STATUSES)
        .astype(int)
    )

    df = df.drop(columns=[TARGET_COLUMN])

    print(f"Rows after filtering: {len(df):,}")
    print("\nTarget distribution:")
    print(df["default"].value_counts())
    print("\nTarget percentage:")
    print(
        df["default"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    return df


def clean_numeric_columns(df):
    """Convert string-based numeric columns into numeric values."""

    print("\nCleaning numeric columns...")

    # Example: "36 months" -> 36
    df["term"] = (
        df["term"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
    )

    df["term"] = pd.to_numeric(
        df["term"],
        errors="coerce"
    )

    # Example: "13.99" or "13.99%" -> 13.99
    df["int_rate"] = (
        df["int_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df["int_rate"] = pd.to_numeric(
        df["int_rate"],
        errors="coerce"
    )

    # Example: "29.7" -> 29.7
    df["revol_util"] = (
        df["revol_util"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df["revol_util"] = pd.to_numeric(
        df["revol_util"],
        errors="coerce"
    )

    # Employment length
    employment_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }

    df["emp_length"] = df["emp_length"].map(
        employment_map
    )

    return df


def engineer_features(df):
    """Create additional credit-risk features."""

    print("\nEngineering features...")

    # Loan amount compared with annual income
    df["loan_to_income"] = (
        df["loan_amnt"]
        / df["annual_inc"].replace(0, np.nan)
    )

    # Annual installment burden compared with income
    df["installment_to_income"] = (
        (df["installment"] * 12)
        / df["annual_inc"].replace(0, np.nan)
    )

    # Missing income indicator
    df["income_missing"] = (
        df["annual_inc"]
        .isna()
        .astype(int)
    )

    return df


def handle_missing_values(df):
    """Handle missing numerical and categorical values."""

    print("\nHandling missing values...")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    return df


def save_dataset(df, path=OUTPUT_PATH):
    """Save the processed dataset."""

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df.to_parquet(
        path,
        index=False
    )

    print("\nProcessed dataset saved to:")
    print(path)


def main():
    print("=" * 50)
    print("LendingClub Credit Risk Preprocessing")
    print("=" * 50)

    df = load_dataset()

    df = create_target(df)

    df = clean_numeric_columns(df)

    df = engineer_features(df)

    df = handle_missing_values(df)

    print("\n========== FINAL DATASET ==========")
    print(f"Shape: {df.shape}")

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum().sum())

    print("\n========== FINAL COLUMNS ==========")
    print(df.columns.tolist())

    save_dataset(df)


if __name__ == "__main__":
    main()