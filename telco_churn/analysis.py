"""Exploratory analysis and report generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from telco_churn.config import ID_COLUMN, TARGET_COLUMN


def get_feature_columns(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return categorical and numeric feature columns, excluding ID and target."""
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    categorical_columns = dataframe[feature_columns].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_columns = dataframe[feature_columns].select_dtypes(
        include="number"
    ).columns.tolist()
    return categorical_columns, numeric_columns


def create_analysis_reports(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    make_plots: bool = False,
) -> None:
    """Write reusable EDA tables and, optionally, plots to disk."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    categorical_columns, numeric_columns = get_feature_columns(dataframe)

    overview = pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "missing_count": dataframe.isna().sum(),
            "missing_percent": dataframe.isna().mean().mul(100).round(2),
            "unique_count": dataframe.nunique(dropna=False),
        }
    )
    overview.to_csv(report_dir / "data_overview.csv", index_label="column")

    if numeric_columns:
        dataframe[numeric_columns].describe().T.to_csv(
            report_dir / "numeric_summary.csv", index_label="column"
        )
        target_numeric = dataframe.groupby(TARGET_COLUMN)[numeric_columns].mean().T
        target_numeric.to_csv(report_dir / "numeric_by_churn.csv", index_label="column")

        correlation_columns = numeric_columns + [TARGET_COLUMN]
        dataframe[correlation_columns].corr().to_csv(
            report_dir / "correlations.csv", index_label="column"
        )

    category_rows: list[pd.DataFrame] = []
    target_rows: list[pd.DataFrame] = []
    for column in categorical_columns:
        counts = dataframe[column].value_counts(dropna=False)
        summary = counts.rename("count").to_frame()
        summary["ratio_percent"] = (counts / len(dataframe) * 100).round(2)
        summary.insert(0, "column", column)
        summary.index.name = "value"
        category_rows.append(summary.reset_index())

        grouped = dataframe.groupby(column, dropna=False)[TARGET_COLUMN].agg(
            churn_rate="mean", count="size"
        )
        grouped.insert(0, "column", column)
        grouped.index.name = "value"
        target_rows.append(grouped.reset_index())

    if category_rows:
        pd.concat(category_rows, ignore_index=True).to_csv(
            report_dir / "categorical_summary.csv", index=False
        )
        pd.concat(target_rows, ignore_index=True).to_csv(
            report_dir / "categorical_by_churn.csv", index=False
        )

    if make_plots:
        _save_plots(dataframe, numeric_columns, categorical_columns, report_dir)


def _save_plots(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if numeric_columns:
        correlation_columns = numeric_columns + [TARGET_COLUMN]
        figure, axis = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            dataframe[correlation_columns].corr(),
            annot=True,
            fmt=".2f",
            cmap="magma",
            ax=axis,
        )
        axis.set_title("Numeric feature correlations")
        figure.tight_layout()
        figure.savefig(plot_dir / "correlation_matrix.png", dpi=150)
        plt.close(figure)

    for column in numeric_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(data=dataframe, x=column, bins=20, ax=axis)
        axis.set_title(f"Distribution of {column}")
        figure.tight_layout()
        figure.savefig(plot_dir / f"{column}_distribution.png", dpi=150)
        plt.close(figure)

    for column in categorical_columns:
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.countplot(data=dataframe, x=column, ax=axis)
        axis.set_title(f"Counts for {column}")
        axis.tick_params(axis="x", labelrotation=30)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{column}_counts.png", dpi=150)
        plt.close(figure)
