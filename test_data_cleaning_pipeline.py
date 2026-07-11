import pandas as pd

from database_normalisation import DataNormalizer
from schemas_detector import SchemaDetector


def _clean(df: pd.DataFrame, level: str = "advanced"):
    schema = SchemaDetector().detect_schema(df)
    normalizer = DataNormalizer()
    cleaned = normalizer.normalize_dataframe(df, schema, level=level)
    return cleaned, normalizer.get_last_report()


def test_generic_cleaning_pipeline_handles_common_uploaded_data_issues():
    df = pd.DataFrame(
        {
            " Order ID ": ["A1", "A1", "A2", None],
            "Sales Amount": ["1,200.50", "1,200.50", "", "300"],
            "Discount Rate": ["15%", "15%", "0.2", None],
            "Order Date": ["2024-01-31", "2024-01-31", "02/01/2024", None],
            "Zip Code": ["00123", "00123", "00456", None],
            "Empty": [None, None, None, None],
        }
    )

    cleaned, report = _clean(df)

    assert "empty" not in cleaned.columns
    assert cleaned["sales_amount"].iloc[0] == 1200.5
    assert str(cleaned["zip_code"].iloc[0]) == "00123"
    assert "order_date_year" in cleaned.columns
    assert "order_date" not in report["missing_values_filled"]
    assert report["empty_columns_removed"] == ["Empty"]
    assert report["duplicates_removed"] >= 1


def test_turnover_rate_is_not_scaled_as_percentage_but_discount_rate_is():
    df = pd.DataFrame({"turnover_rate": [3.5], "discount_rate": [15.0]})

    cleaned, _ = _clean(df, level="basic")

    assert cleaned.loc[0, "turnover"] == 3.5
    assert abs(cleaned.loc[0, "discount_rate"] - 0.15) < 1e-9

