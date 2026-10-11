
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ingress_cleaned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "eda"

def main():
    df = pd.read_parquet(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Count normal and anomalous observations.
    counts = df["Label"].value_counts().reindex([0, 1], fill_value=0)

    plt.figure(figsize=(7, 5))
    plt.bar(["Normal (0)", "Anomaly (1)"], counts.values)
    plt.title("Normal vs Anomalous Observations")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png", dpi=150)
    plt.close()

    # 2. Plot one time series and mark labelled anomalies.
    series_name = "ingress-01"
    series_df = df[df["Series"] == series_name].sort_values("TimeStamp")
    anomalies = series_df[series_df["Label"] == 1]

    plt.figure(figsize=(12, 5))
    plt.plot(series_df["TimeStamp"], series_df["Value"],
             label="Ingress rate", linewidth=1)
    plt.scatter(anomalies["TimeStamp"], anomalies["Value"],
                label="Labelled anomalies", s=15)
    plt.title(f"Cloud Ingress Rate and Labelled Anomalies: {series_name}")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Ingress rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ingress_01_anomalies.png", dpi=150)
    plt.close()

    print("EDA completed.")
    print("Charts saved in:", OUTPUT_DIR)
    print("Label counts:\n", counts)

if __name__ == "__main__":
    main()
