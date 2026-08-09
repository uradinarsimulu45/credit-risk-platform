import pandas as pd
from sklearn.datasets import fetch_openml


def main():
    print("Loading German Credit Risk dataset...")

    data = fetch_openml(
        name="credit-g",
        version=1,
        as_frame=True
    )

    df = data.frame

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["class"].value_counts())

    print("\n========== TARGET PERCENTAGE ==========")
    print(df["class"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()