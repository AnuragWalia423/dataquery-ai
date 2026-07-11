import pandas as pd

from database_connection import DatabaseConnection
from database_normalisation import DataNormalizer
from schemas_detector import SchemaDetector
from sql_agent import EnhancedSQLAgent


def main():
    source_path = "Sample_Superstore.csv"
    table_name = "sample_superstore_pipeline_test"

    raw_df = pd.read_csv(source_path)
    schema = SchemaDetector().detect_schema(raw_df)
    cleaned_df = DataNormalizer().normalize_dataframe(raw_df, schema, level="advanced")

    db = DatabaseConnection("sqlite")
    agent = EnhancedSQLAgent(db)
    loaded = agent.load_dataset(cleaned_df, table_name, already_normalized=True)

    if not loaded:
        raise RuntimeError(agent.last_error or "load_dataset returned False")

    row_count_df = db.execute_query(f"SELECT COUNT(*) AS row_count FROM {table_name}")
    totals_df = db.execute_query(
        f"SELECT ROUND(SUM(sales), 2) AS total_sales, "
        f"ROUND(SUM(profit), 2) AS total_profit FROM {table_name}"
    )

    loaded_rows = int(row_count_df.loc[0, "row_count"])
    if loaded_rows != len(cleaned_df):
        raise AssertionError(f"Expected {len(cleaned_df)} rows, got {loaded_rows}")

    print("PIPELINE_OK")
    print(f"python_rows_read={len(raw_df)}")
    print(f"cleaned_rows={len(cleaned_df)}")
    print(f"sqlite_rows_loaded={loaded_rows}")
    print(totals_df.to_string(index=False))


if __name__ == "__main__":
    main()
