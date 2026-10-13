"""
Assignment 1 — training pipeline.

Dataset : UCI Bike Sharing, hourly file (data/hour.csv). One row = one hour.
Target  : cnt, total bike rentals in that hour (regression).
Model   : linear regression, trained three ways (scikit-learn, manual PyTorch
          loop, nn.Module + torch.optim) on the same split.
Scenario: operations team plans the next day, so features are calendar +
          weather only (no previous-hour demand).

Run with:  uv run python main.py train

STEP 1: load the data, build a timestamp, report data quality.
STEP 2 (this version): chronological split + cleaning decisions.
Next steps: features -> baseline -> three models -> evaluation -> save artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hour.csv"
SEED = 0
TARGET = "cnt"

# Columns that must never be used as features.
# casual + registered == cnt, and we would not know them in advance, so using
# them would leak the answer into the model.
LEAKAGE_COLS = ["casual", "registered"]
# Index and raw date: only used to build the timestamp and order the rows.
ID_COLS = ["instant", "dteday"]

# Chronological split: first ~60% of the time for train, next ~20% for
# validation, last ~20% for test. A random split would put near-identical
# neighbouring hours in both train and test and leak the future.
VAL_START = pd.Timestamp("2012-03-15")
TEST_START = pd.Timestamp("2012-08-07")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # One real timestamp per row: date + hour of the day.
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    # Sort by time: the split by date relies on this order.
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def data_quality_report(df):
    """Print the numbers that justify the cleaning decisions in REPORT.md."""
    print("rows, columns:", df.shape)
    print("period:", df["timestamp"].min(), "->", df["timestamp"].max())

    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    print("hours missing from the calendar:", len(full_range.difference(df["timestamp"])))
    print("duplicated timestamps:", df["timestamp"].duplicated().sum())

    print("cnt == casual + registered in every row:",
          bool((df["cnt"] == df["casual"] + df["registered"]).all()))

    n_hum0 = int((df["hum"] == 0).sum())
    n_ws0 = int((df["windspeed"] == 0).sum())
    print("hum == 0 rows:", n_hum0)
    print(f"windspeed == 0 rows: {n_ws0} ({n_ws0 / len(df):.1%})")
    print("weathersit counts:", df["weathersit"].value_counts().sort_index().to_dict())
    print("mean cnt per year:", df.groupby("yr")[TARGET].mean().round(1).to_dict())
    print("corr(temp, atemp):", round(df["temp"].corr(df["atemp"]), 3))


def split_by_date(df):
    train = df[df["timestamp"] < VAL_START].copy()
    val = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] < TEST_START)].copy()
    test = df[df["timestamp"] >= TEST_START].copy()
    return train, val, test


def split_report(train, val, test):
    n = len(train) + len(val) + len(test)
    print("\nchronological split:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(
            f"  {name:5s} {len(part):6d} rows ({len(part) / n:.0%})  "
            f"{part['timestamp'].min().date()} -> {part['timestamp'].max().date()}  "
            f"mean cnt {part[TARGET].mean():6.1f}  "
            f"hum==0: {(part['hum'] == 0).sum():2d}  weathersit==4: {(part['weathersit'] == 4).sum()}"
        )


def clean_data(train, val, test):
    """Apply the cleaning decisions (each one is explained in REPORT.md).

    1. hum == 0 is a sensor failure (22 rows, one single day) -> replace with
       the median humidity of the TRAIN rows. Computed on train only, so no
       information from val/test leaks into the preprocessing.
    2. weathersit == 4 has only 3 rows -> merge into category 3.
    3. windspeed == 0 is kept as it is: zeros look like real calm / readings
       too weak to measure, not clear errors.
    """
    median_hum = train.loc[train["hum"] > 0, "hum"].median()

    cleaned = []
    for part in (train, val, test):
        part = part.copy()
        part.loc[part["hum"] == 0, "hum"] = median_hum
        part.loc[part["weathersit"] == 4, "weathersit"] = 3
        cleaned.append(part)

    params = {"median_hum": float(median_hum)}
    print(f"\ncleaning: hum == 0 replaced with train median {median_hum:.2f}; "
          f"weathersit 4 merged into 3")
    return cleaned[0], cleaned[1], cleaned[2], params


def main():
    set_seed()
    df = load_data()
    data_quality_report(df)

    train, val, test = split_by_date(df)
    split_report(train, val, test)

    train, val, test, cleaning_params = clean_data(train, val, test)

    # TODO next step: build features (hour, working day, weather, ...).
    print("\nstep 2 done:", len(train), len(val), len(test))


if __name__ == "__main__":
    main()