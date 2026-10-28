"""Command-line entry point."""

import argparse
from pathlib import Path

from telco_churn.analysis import create_analysis_reports
from telco_churn.data import impute_missing_values, load_dataset
from telco_churn.features import engineer_features
from telco_churn.modeling import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Telco customer churn and train a CatBoost baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Telco-Customer-Churn.csv"),
        help="Path to the Telco churn CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated reports (default: %(default)s).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save exploratory charts in the report directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataframe = load_dataset(args.data)
    create_analysis_reports(
        dataframe,
        args.output_dir / "eda",
        make_plots=args.plots,
    )
    prepared = impute_missing_values(dataframe)
    featured = engineer_features(prepared)
    metrics = train_and_evaluate(featured, args.output_dir / "metrics.json")

    print(f"Analysis reports saved to: {args.output_dir / 'eda'}")
    print(f"Model metrics saved to: {args.output_dir / 'metrics.json'}")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")
    return 0
