"""
LendingClub Dataset Analysis
Credit Risk Prediction Project

Analyzes the raw LendingClub loan dataset without modifying it.

Dataset:
data/raw/loans.csv
"""

import os
import time
import pandas as pd


DATA_PATH = "data/raw/loans.csv.gz"


def load_dataset(path=DATA_PATH):
    """Load the LendingClub dataset."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    print("Loading LendingClub dataset...")
    print(f"Dataset path: {path}")

    start_time = time.time()

    columns = [
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
        "loan_status",
    ]

    df = pd.read_csv(
        path,
        usecols=lambda column: column in columns,
        low_memory=False,
	compression="gzip"
    )

    elapsed = time.time() - start_time

    print(f"Dataset loaded successfully in {elapsed:.2f} seconds.")

    return df


def analyze_dataset(df):
    """Print basic dataset analysis."""

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing)

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())

    print("\n========== LOAN STATUS DISTRIBUTION ==========")
    print(
        df["loan_status"]
        .value_counts(dropna=False)
    )

    print("\n========== TOP 10 LOAN PURPOSES ==========")
    print(
        df["purpose"]
        .value_counts()
        .head(10)
    )

    print("\n========== LOAN GRADE DISTRIBUTION ==========")
    print(
        df["grade"]
        .value_counts()
        .sort_index()
    )

    print("\n========== NUMERICAL SUMMARY ==========")

    print(
        df[
            [
                "loan_amnt",
                "int_rate",
                "installment",
                "annual_inc",
                "dti",
                "delinq_2yrs",
                "open_acc",
                "pub_rec",
                "revol_bal",
                "total_acc",
            ]
        ].describe()
    )


def main():

    print("======================================")
    print(" LendingClub Credit Risk Analysis")
    print("======================================")

    df = load_dataset()

    analyze_dataset(df)

    print("\n======================================")
    print(" Analysis completed successfully")
    print("======================================")


if __name__ == "__main__":
    main()