
from pathlib import Path

import pandas as pd

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "temp-dataset" / "data" / "data-ingress-rate"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_and_clean_data():
    """Load and validate the five cloud ingress time series."""

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DATA_DIR}"
        )

    dataframes = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        required_columns = {"TimeStamp", "Value", "Label"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Unexpected columns in {file_path.name}")

        df["TimeStamp"] = pd.to_datetime(
            df["TimeStamp"], utc=True, errors="coerce"
        )
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        # Keep track of which time series each row came from.
        df["Series"] = file_path.stem

        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    # Remove rows with invalid timestamps, values, or labels.
    combined = combined.dropna(
        subset=["TimeStamp", "Value", "Label"]
    )
    combined = combined[combined["Label"].isin([0, 1])]

    # Sort records by series and time.
    combined = combined.sort_values(
        ["Series", "TimeStamp"]
    ).reset_index(drop=True)

    return combined


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()

    output_path = PROCESSED_DATA_DIR / "ingress_cleaned.parquet"
    df.to_parquet(output_path, index=False)

    print("Data preparation completed!")
    print("Rows and columns:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Label counts:\n", df["Label"].value_counts().sort_index())
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
