
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without opening graphical windows

import matplotlib.pyplot as plt
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ingress_cleaned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "eda"


def load_data():
    """Load and validate the processed dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    required_columns = {"TimeStamp", "Value", "Label", "Series"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return df


def plot_label_distribution(df):
    """Plot the number of normal and anomalous observations."""

    counts = df["Label"].value_counts().reindex([0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        ["Normal (0)", "Anomaly (1)"],
        counts.values,
        color=["steelblue", "tomato"],
    )

    ax.set_title("Normal vs Anomalous Observations")
    ax.set_xlabel("Observation type")
    ax.set_ylabel("Number of observations")
    ax.bar_label(bars, padding=3)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "label_distribution.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_time_series(df, series_name="ingress-01"):
    """Plot ingress values and highlight dataset-labelled anomalies."""

    series_df = (
        df[df["Series"] == series_name]
        .sort_values("TimeStamp")
        .copy()
    )

    if series_df.empty:
        raise ValueError(f"No observations found for {series_name}")

    anomalies = series_df[series_df["Label"] == 1]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        series_df["TimeStamp"],
        series_df["Value"],
        color="steelblue",
        label="Ingress rate",
        linewidth=1,
    )

    ax.scatter(
        anomalies["TimeStamp"],
        anomalies["Value"],
        color="red",
        label="Labelled anomalies",
        s=22,
        zorder=3,
    )

    ax.set_title(
        f"Cloud Ingress Rate and Labelled Anomalies: {series_name}"
    )
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Ingress rate")
    ax.legend()

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{series_name}_anomalies.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def create_data_quality_summary(df):
    """Save overall and per-series statistics as CSV files."""

    overall_summary = pd.DataFrame([{
        "total_observations": len(df),
        "number_of_series": df["Series"].nunique(),
        "normal_observations": int((df["Label"] == 0).sum()),
        "anomalous_observations": int((df["Label"] == 1).sum()),
        "anomaly_percentage": round((df["Label"] == 1).mean() * 100, 3),
        "missing_values": int(df.isnull().sum().sum()),
        "minimum_value": df["Value"].min(),
        "maximum_value": df["Value"].max(),
        "mean_value": df["Value"].mean(),
        "median_value": df["Value"].median(),
    }])

    overall_path = OUTPUT_DIR / "data_quality_summary.csv"
    overall_summary.to_csv(overall_path, index=False)
    print(f"Saved: {overall_path}")

    per_series_summary = (
        df.groupby("Series")
        .agg(
            observations=("Value", "size"),
            normal_observations=("Label", lambda x: (x == 0).sum()),
            anomalous_observations=("Label", lambda x: (x == 1).sum()),
            minimum_value=("Value", "min"),
            maximum_value=("Value", "max"),
            mean_value=("Value", "mean"),
            median_value=("Value", "median"),
            missing_values=("Value", lambda x: x.isnull().sum()),
        )
        .reset_index()
    )

    per_series_path = OUTPUT_DIR / "per_series_summary.csv"
    per_series_summary.to_csv(per_series_path, index=False)
    print(f"Saved: {per_series_path}")

    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))

    print("\nSummary by time series:")
    print(per_series_summary.to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("EDA started.")
    print("Dataset shape:", df.shape)

    plot_label_distribution(df)
    plot_time_series(df, "ingress-01")
    create_data_quality_summary(df)

    print("\nEDA completed successfully.")
    print("Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
