# Telco Customer Churn

Modular Python project for exploratory data analysis, feature engineering, and a
baseline CatBoost model using the Telco Customer Churn dataset.

## Project structure

```text
telco_churn/
├── telco_churn/
│   ├── analysis.py       # EDA tables and optional charts
│   ├── cli.py            # Command-line arguments and pipeline orchestration
│   ├── config.py         # Dataset column definitions
│   ├── data.py           # CSV validation and missing-value handling
│   ├── features.py       # Derived feature creation
│   └── modeling.py       # CatBoost training and holdout evaluation
├── tests/
│   └── test_pipeline.py
├── data/
│   └── raw/              # Place the dataset here (not tracked by Git)
├── reports/              # Generated reports (not tracked by Git)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10–3.13
- The packages listed in `requirements.txt`
- A Telco Customer Churn CSV file with the expected columns, including
  `customerID`, `TotalCharges`, and `Churn`

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put `Telco-Customer-Churn.csv` in `data/raw/`. The data is intentionally not
included in this repository; use a copy you are permitted to use.

## Run

From the project directory:

```powershell
python -m telco_churn
```

Or provide a different CSV path and request charts:

```powershell
python -m telco_churn --data "C:\path\to\Telco-Customer-Churn.csv" --output-dir reports --plots
```

The pipeline validates the CSV, writes EDA summaries, imputes missing numeric
values, creates churn features, and evaluates a stratified holdout split.
Generated files include `reports/metrics.json` and CSV tables under
`reports/eda/`; `--plots` also writes charts to `reports/eda/figures/`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- `TotalCharges` is parsed as numeric, with blank or invalid values reported in
  the initial EDA and median-imputed before feature engineering.
- `customerID` is retained for reporting but excluded from model features.
- ROC AUC is calculated from predicted probabilities; the other classification
  metrics use the holdout labels.
