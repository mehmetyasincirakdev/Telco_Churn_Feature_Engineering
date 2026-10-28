"""Feature engineering for the Telco customer churn dataset."""

import numpy as np
import pandas as pd


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the input with derived churn features."""
    result = dataframe.copy()

    result["NEW_TENURE_YEAR"] = pd.cut(
        result["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=[
            "0-1 Year",
            "1-2 Year",
            "2-3 Year",
            "3-4 Year",
            "4-5 Year",
            "5-6 Year",
        ],
        include_lowest=True,
    )
    result["NEW_Engaged"] = result["Contract"].isin(
        ["One year", "Two year"]
    ).astype("int8")

    protection_columns = ["OnlineBackup", "DeviceProtection", "TechSupport"]
    result["NEW_noProt"] = (~result[protection_columns].eq("Yes").any(axis=1)).astype(
        "int8"
    )
    result["NEW_Young_Not_Engaged"] = (
        result["NEW_Engaged"].eq(0) & result["SeniorCitizen"].eq(0)
    ).astype("int8")

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    subscribed_services = result[service_columns].eq("Yes").sum(axis=1)
    has_internet = result["InternetService"].ne("No").astype("int64")
    result["NEW_TotalServices"] = subscribed_services + has_internet
    result["NEW_FLAG_ANY_STREAMING"] = (
        result[["StreamingTV", "StreamingMovies"]].eq("Yes").any(axis=1)
    ).astype("int8")
    result["NEW_FLAG_AutoPayment"] = result["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype("int8")

    result["NEW_AVG_Charges"] = result["TotalCharges"] / (result["tenure"] + 1)
    result["NEW_Increase"] = np.divide(
        result["NEW_AVG_Charges"],
        result["MonthlyCharges"],
        out=np.zeros(len(result), dtype=float),
        where=result["MonthlyCharges"].to_numpy() != 0,
    )
    result["NEW_AVG_Service_Fee"] = result["MonthlyCharges"] / (
        result["NEW_TotalServices"] + 1
    )

    return result
