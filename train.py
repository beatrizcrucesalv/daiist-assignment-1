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
STEP 2: chronological split + cleaning decisions.
STEP 3: feature engineering.
STEP 4 (this version): scaling, naive baselines, scikit-learn model.
Next steps: manual PyTorch -> nn.Module + optim -> final comparison on test
-> save artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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

# "Peak hours" for the signed-error check: the morning and evening commute
# windows on working days, where the operations team cannot afford to run short.
PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19]

# Regularization strengths tried for the scikit-learn model (chosen on the
# validation set, never on the test set).
ALPHAS = [0.01, 0.1, 1, 10, 100]


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


def one_hot(series, categories, prefix):
    """One 0/1 column per category, dropping the first one as the baseline.

    Categories are fixed in advance, so train, val and test always get exactly
    the same columns, even if a category is missing from one of the splits.
    """
    typed = series.astype(pd.CategoricalDtype(categories))
    return pd.get_dummies(typed, prefix=prefix, drop_first=True, dtype=float)


def build_features(df):
    """Turn a cleaned dataframe into the feature matrix X (all numeric).

    Kept on purpose: temp, hum, windspeed, yr, workingday, hour of day, month,
    weather type. Left out on purpose (each can be added back to measure its
    impact):
      - atemp    : correlation 0.99 with temp, nearly the same information
      - season   : fully determined by the month
      - holiday  : workingday already is 0 on holidays
      - weekday  : workingday already separates weekends from weekdays
    """
    X = pd.DataFrame(index=df.index)

    # Numeric columns used as they are (temp, hum, windspeed are already 0-1).
    for col in ["temp", "hum", "windspeed", "yr", "workingday"]:
        X[col] = df[col].astype(float)

    # Hour of day: one column per hour, so the model can draw any daily curve.
    hours = one_hot(df["hr"], list(range(24)), "hr")

    # Hour x workingday: a separate daily curve for working days vs weekends.
    hours_x_work = hours.mul(df["workingday"], axis=0).add_suffix("_x_work")

    months = one_hot(df["mnth"], list(range(1, 13)), "mnth")
    weather = one_hot(df["weathersit"], [1, 2, 3], "weather")

    X = pd.concat([X, hours, hours_x_work, months, weather], axis=1)

    # Safety nets: no forbidden column, no missing values.
    assert not set(LEAKAGE_COLS + ID_COLS) & set(X.columns)
    assert not X.isna().any().any()
    return X


def scale_features(X_train, X_val, X_test):
    """Standardise every column: (x - mean) / std, using TRAIN statistics only.

    The same scaled matrices feed all three models, so they see identical data.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def evaluate_predictions(name, y_true, y_pred, part):
    """RMSE and MAE in bikes, plus the signed error at peak hours.

    Signed error = prediction - actual, averaged over working-day commute
    hours. A negative value means the model runs short exactly when it hurts.
    """
    y_true = np.asarray(y_true, dtype=float)
    # A rental count cannot be negative, so negative predictions are clipped to 0.
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    peak = ((part["workingday"] == 1) & part["hr"].isin(PEAK_HOURS)).to_numpy()
    return {
        "model": name,
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "peak signed error": float(np.mean(y_pred[peak] - y_true[peak])),
    }


def baseline_predictions(train, part):
    """Two reference predictions that use no features beyond the hour."""
    overall_mean = np.full(len(part), train[TARGET].mean())
    hour_means = train.groupby("hr")[TARGET].mean()
    per_hour_mean = part["hr"].map(hour_means).to_numpy()
    return overall_mean, per_hour_mean


def fit_sklearn(Xs_train, y_train, Xs_val, y_val, val):
    """Ridge regression (linear regression + L2 penalty); alpha picked on validation."""
    best_model, best_rmse = None, np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(Xs_train, y_train)
        rmse = evaluate_predictions("tmp", y_val, model.predict(Xs_val), val)["RMSE"]
        print(f"  alpha {alpha:>6}: validation RMSE {rmse:.2f}")
        if rmse < best_rmse:
            best_model, best_rmse = model, rmse
    print(f"  chosen alpha: {best_model.alpha}")
    return best_model


def main():
    set_seed()
    df = load_data()
    data_quality_report(df)

    train, val, test = split_by_date(df)
    split_report(train, val, test)

    train, val, test, cleaning_params = clean_data(train, val, test)

    X_train, y_train = build_features(train), train[TARGET]
    X_val, y_val = build_features(val), val[TARGET]
    X_test, y_test = build_features(test), test[TARGET]

    # The three splits must have exactly the same columns, in the same order.
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)
    print(f"\nfeatures: {X_train.shape[1]} columns")
    print("shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("first columns:", list(X_train.columns[:8]))

    # Standardise (train statistics only), same matrices for every model.
    scaler, Xs_train, Xs_val, Xs_test = scale_features(X_train, X_val, X_test)

    # Baselines, evaluated on VALIDATION. The test set stays untouched until
    # all three models are ready, so nothing gets tuned on it.
    mean_val, hour_val = baseline_predictions(train, val)
    results = [
        evaluate_predictions("baseline: train mean", y_val, mean_val, val),
        evaluate_predictions("baseline: mean per hour", y_val, hour_val, val),
    ]

    print("\nscikit-learn (Ridge):")
    sk_model = fit_sklearn(Xs_train, y_train, Xs_val, y_val, val)
    results.append(evaluate_predictions("scikit-learn", y_val, sk_model.predict(Xs_val), val))

    print("\nvalidation results (bikes per hour):")
    print(pd.DataFrame(results).round(1).to_string(index=False))

    # TODO next step: manual PyTorch loop, then nn.Module + torch.optim.
    print("\nstep 4 done")


if __name__ == "__main__":
    main()