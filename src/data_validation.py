
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ingress_cleaned.parquet"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "eda" / "data_validation_report.csv"


def main():
    df = pd.read_parquet(DATA_PATH)

    checks = [
        ("Total observations", len(df)),
        ("Number of series", df["Series"].nunique()),
        ("Missing values", int(df.isnull().sum().sum())),
        ("Duplicate rows", int(df.duplicated().sum())),
        (
            "Duplicate series-timestamp pairs",
            int(df.duplicated(["Series", "TimeStamp"]).sum()),
        ),
        ("Invalid labels", int((~df["Label"].isin([0, 1])).sum())),
        ("Non-finite values", int((~np.isfinite(df["Value"])).sum())),
        ("Negative values", int((df["Value"] < 0).sum())),
    ]

    results = pd.DataFrame(checks, columns=["Check", "Result"])

    ordering_passed = all(
        group["TimeStamp"].is_monotonic_increasing
        for _, group in df.groupby("Series")
    )

    results.loc[len(results)] = [
        "Timestamps ordered within each series",
        "PASS" if ordering_passed else "FAIL",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(results.to_string(index=False))
    print(f"\nValidation report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
