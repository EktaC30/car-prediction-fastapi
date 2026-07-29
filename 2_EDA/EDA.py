import pandas as pd
import numpy as np


# ==========================================================
# CONSTANTS
# ==========================================================

HEADERS = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

RAW_DATA_PATH = "../1_Data/imports-85.data"
CLEAN_DATA_PATH = "../1_Data/car_cleaned.csv"


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(file_path):
    """
    Load raw automobile dataset.
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = HEADERS

    print(f"Dataset Loaded Successfully")
    print(f"Shape : {df.shape}")

    return df


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

def replace_question_marks(df):
    """
    Replace '?' with NaN.
    """
    df.replace("?", np.nan, inplace=True)
    return df


def convert_datatypes(df):
    """
    Convert required columns to numeric.
    """

    numeric_columns = [
        "normalized-losses",
        "bore",
        "stroke",
        "horsepower",
        "peak-rpm",
        "price"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fill_numeric_missing_values(df):
    """
    Fill missing numeric values using mean.
    """

    numeric_columns = [
        "normalized-losses",
        "bore",
        "stroke",
        "horsepower",
        "peak-rpm",
        "price"
    ]

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    return df


def fill_categorical_missing_values(df):
    """
    Fill missing categorical values using mode.
    """

    most_common_door = df["num-of-doors"].mode()[0]

    df["num-of-doors"] = (
        df["num-of-doors"]
        .fillna(most_common_door)
    )

    return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def create_fuel_efficiency_features(df):
    """
    Create L/100km columns.
    """

    df["city-L/100 km"] = 235 / df["city-mpg"]

    df["highway-L/100 km"] = 235 / df["highway-mpg"]

    df.drop(
        ["city-mpg", "highway-mpg"],
        axis=1,
        inplace=True
    )

    return df


def rename_columns(df):
    """
    Rename columns.
    """

    df.rename(
        columns={
            "num-of-doors": "doors"
        },
        inplace=True
    )

    return df


def create_horsepower_bins(df):
    """
    Categorize horsepower.
    """

    bins = np.linspace(
        df["horsepower"].min(),
        df["horsepower"].max(),
        4
    )

    labels = ["Low", "Medium", "High"]

    df["horsepower-binned"] = pd.cut(
        df["horsepower"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


def create_dummy_variables(df):
    """
    Create dummy variables for fuel type.
    """

    fuel_dummies = pd.get_dummies(
        df["fuel-type"],
        dtype=int
    )

    fuel_dummies.rename(
        columns={
            "diesel": "fuel-type-diesel",
            "gas": "fuel-type-gas"
        },
        inplace=True
    )

    df = pd.concat(
        [df, fuel_dummies],
        axis=1
    )

    df.drop(
        "fuel-type",
        axis=1,
        inplace=True
    )

    return df


# ==========================================================
# CLEANING PIPELINE
# ==========================================================

def clean_data(df):

    df = replace_question_marks(df)

    df = convert_datatypes(df)

    df = fill_numeric_missing_values(df)

    df = fill_categorical_missing_values(df)

    df = create_fuel_efficiency_features(df)

    df = rename_columns(df)

    df = create_horsepower_bins(df)

    df = create_dummy_variables(df)

    return df


# ==========================================================
# SAVE DATA
# ==========================================================

def save_clean_data(df, output_path):

    df.to_csv(
        output_path,
        index=False
    )

    print("\nCleaned Dataset Saved Successfully")
    print(f"Location : {output_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    try:

        df = load_data(RAW_DATA_PATH)

        print("\nCleaning Dataset...")

        cleaned_df = clean_data(df)

        print("\nFinal Shape :", cleaned_df.shape)

        print("\nMissing Values Remaining:")
        print(cleaned_df.isnull().sum().sum())

        save_clean_data(
            cleaned_df,
            CLEAN_DATA_PATH
        )

        print("\nEDA Pipeline Completed Successfully ✅")

    except Exception as e:

        print(f"\nError : {e}")


if __name__ == "__main__":
    main()