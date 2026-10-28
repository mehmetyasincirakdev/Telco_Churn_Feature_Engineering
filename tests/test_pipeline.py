import unittest

import pandas as pd

from telco_churn.data import impute_missing_values
from telco_churn.features import engineer_features


def make_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 0],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["DSL", "No"],
            "OnlineSecurity": ["No", "No internet service"],
            "OnlineBackup": ["No", "No internet service"],
            "DeviceProtection": ["Yes", "No internet service"],
            "TechSupport": ["No", "No internet service"],
            "StreamingTV": ["Yes", "No internet service"],
            "StreamingMovies": ["No", "No internet service"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "MonthlyCharges": [75.0, 0.0],
            "TotalCharges": [900.0, None],
            "Churn": [1, 0],
        }
    )


class PipelineTests(unittest.TestCase):
    def test_imputation_fills_missing_numeric_values_without_mutating_input(self) -> None:
        original = make_sample()

        result = impute_missing_values(original)

        self.assertEqual(result["TotalCharges"].isna().sum(), 0)
        self.assertEqual(original["TotalCharges"].isna().sum(), 1)

    def test_feature_engineering_adds_expected_flags(self) -> None:
        result = engineer_features(impute_missing_values(make_sample()))

        self.assertEqual(result["NEW_TENURE_YEAR"].tolist(), ["0-1 Year", "0-1 Year"])
        self.assertEqual(result["NEW_noProt"].tolist(), [0, 1])
        self.assertEqual(result["NEW_TotalServices"].tolist(), [4, 0])
        self.assertEqual(result["NEW_FLAG_ANY_STREAMING"].tolist(), [1, 0])
        self.assertEqual(result["NEW_FLAG_AutoPayment"].tolist(), [0, 1])
        self.assertEqual(result["NEW_Increase"].iloc[1], 0)

    def test_feature_engineering_does_not_mutate_input(self) -> None:
        original = impute_missing_values(make_sample())

        result = engineer_features(original)

        self.assertNotIn("NEW_Engaged", original.columns)
        self.assertIn("NEW_Engaged", result.columns)


if __name__ == "__main__":
    unittest.main()
