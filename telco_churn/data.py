"""Dataset loading, validation, and missing-value handling."""

from pathlib import Path

import pandas as pd

from telco_churn.config import REQUIRED_COLUMNS, TARGET_COLUMN


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a Telco churn CSV file."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Provide a CSV with --data."
        )

    dataframe = pd.read_csv(data_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataframe["TotalCharges"] = pd.to_numeric(
        dataframe["TotalCharges"], errors="coerce"
    )
    churn_values = dataframe[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if churn_values.isna().any():
        invalid_values = dataframe.loc[
            churn_values.isna(), TARGET_COLUMN
        ].drop_duplicates().tolist()
        raise ValueError(
            f"Unexpected values in {TARGET_COLUMN}: {invalid_values}. "
            "Expected 'Yes' or 'No'."
        )
    dataframe[TARGET_COLUMN] = churn_values.astype("int8")

    return dataframe


def impute_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with their column median."""
    result = dataframe.copy()
    numeric_columns = result.select_dtypes(include="number").columns

    for column in numeric_columns:
        if result[column].isna().any():
            median = result[column].median()
            if pd.isna(median):
                raise ValueError(
                    f"Cannot impute '{column}': the column has no numeric values."
                )
            result[column] = result[column].fillna(median)

    return result
