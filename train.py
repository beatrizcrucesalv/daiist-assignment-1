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
STEP 4: scaling, naive baselines, scikit-learn model.
STEP 5: manual PyTorch loop (raw tensors, autograd, hand update).
STEP 6: standard PyTorch workflow (nn.Module + torch.optim).
STEP 7 (this version): final comparison on the test set + save artifacts for app.py.
Next: app.py (dashboard that only loads what is saved here).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hour.csv"
ARTIFACT_DIR = ROOT / "artifacts"  # everything app.py needs is saved here
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

# Learning rates tried for the PyTorch models (also chosen on validation), and
# the number of full passes over the training data. The weights start at zero,
# so there is no randomness in the manual loop. With 156 correlated features,
# gradient descent needs many steps to finish converging, and a learning rate
# above about 0.23 (2 / the largest curvature of the loss) makes it blow up.
LEARNING_RATES = [0.01, 0.1, 0.2]
N_EPOCHS = 10000


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
    weather type, plus interactions of the hour with temperature and with the
    year (separately for working days and weekends). Left out on purpose (adding
    any of them back changed validation RMSE by less than 0.2):
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

    # The daily curve is not the same in every situation: commuters ride more
    # when it is warm, and the system grew between 2011 and 2012, so the peaks
    # get higher. A plain linear model adds one fixed effect per hour and cannot
    # show that. Multiplying each hour column by temperature and by the year
    # lets the curve change with both, for working days and weekends separately.
    interactions = []
    for modifier in ["yr", "temp"]:
        value = df[modifier].astype(float)
        interactions.append(hours.mul(value, axis=0).add_suffix(f"_x_{modifier}"))
        interactions.append(hours_x_work.mul(value, axis=0).add_suffix(f"_x_{modifier}"))

    X = pd.concat([X, hours, hours_x_work, months, weather] + interactions, axis=1)

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


def to_tensor(array):
    """NumPy array -> float32 tensor (PyTorch trains in float32)."""
    return torch.tensor(np.asarray(array), dtype=torch.float32)


def train_manual(X, y, lr, n_epochs=N_EPOCHS):
    """Linear regression by gradient descent, written by hand (Session 5).

    Model : y_pred = X @ w + b
    Loss  : mean squared error
    Update: w <- w - lr * dLoss/dw, computed by autograd, applied by us.
    """
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss_history = []

    for epoch in range(n_epochs):
        y_pred = X @ w + b
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():  # the update itself must not be tracked by autograd
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()  # backward() accumulates gradients, so reset them every step
        b.grad.zero_()
        loss_history.append(loss.item())

    return w.detach(), b.detach(), loss_history


def predict_manual(X, w, b):
    with torch.no_grad():
        return (X @ w + b).numpy()


def fit_manual(Xs_train, y_train, Xs_val, y_val, val):
    """Train the manual loop for each learning rate; keep the best on validation."""
    X_t, y_t, X_v = to_tensor(Xs_train), to_tensor(y_train), to_tensor(Xs_val)

    best = None
    best_rmse = np.inf
    for lr in LEARNING_RATES:
        w, b, history = train_manual(X_t, y_t, lr)
        rmse = evaluate_predictions("tmp", y_val, predict_manual(X_v, w, b), val)["RMSE"]
        print(f"  lr {lr:<6}: final train loss {history[-1]:8.1f}, validation RMSE {rmse:.2f}")
        if rmse < best_rmse:
            best, best_rmse = (w, b, history, lr), rmse
    print(f"  chosen lr: {best[3]}")
    return best


def train_nn(X, y, lr, n_epochs=N_EPOCHS):
    """The same linear regression, with the standard PyTorch workflow.

    nn.Linear holds w and b for us, torch.optim.SGD applies the update
    w <- w - lr * grad (the exact rule we wrote by hand), and
    optimizer.zero_grad() / optimizer.step() replace the manual bookkeeping.
    """
    torch.manual_seed(SEED)  # nn.Linear starts from random weights
    model = nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()                # replaces the manual .grad.zero_()
        y_pred = model(X).squeeze(-1)        # (n, 1) -> (n,)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()                     # replaces the manual no_grad update
        loss_history.append(loss.item())

    return model, loss_history


def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).squeeze(-1).numpy()


def fit_nn(Xs_train, y_train, Xs_val, y_val, val):
    """Train the nn.Module workflow for each learning rate; keep the best on validation."""
    X_t, y_t, X_v = to_tensor(Xs_train), to_tensor(y_train), to_tensor(Xs_val)

    best = None
    best_rmse = np.inf
    for lr in LEARNING_RATES:
        model, history = train_nn(X_t, y_t, lr)
        rmse = evaluate_predictions("tmp", y_val, predict_nn(model, X_v), val)["RMSE"]
        print(f"  lr {lr:<6}: final train loss {history[-1]:8.1f}, validation RMSE {rmse:.2f}")
        if rmse < best_rmse:
            best, best_rmse = (model, history, lr), rmse
    print(f"  chosen lr: {best[2]}")
    return best


# Short keys (used in files) -> readable names (used in printed tables).
MODEL_NAMES = {
    "baseline_mean": "baseline: train mean",
    "baseline_per_hour": "baseline: mean per hour",
    "sklearn": "scikit-learn",
    "manual": "manual PyTorch",
    "nn": "nn.Module + optim",
}


def results_table(part, preds):
    """One row of metrics per model, for one split."""
    return [
        evaluate_predictions(MODEL_NAMES[key], part[TARGET], pred, part)
        for key, pred in preds.items()
    ]


def save_models(sk_model, scaler, w_manual, b_manual, manual_history,
                nn_model, nn_history, feature_columns, cleaning_params,
                chosen_hyperparameters):
    """Save the trained models and everything needed to use them on new data."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(sk_model, ARTIFACT_DIR / "sklearn_model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    torch.save({"w": w_manual, "b": b_manual}, ARTIFACT_DIR / "manual_model.pt")
    torch.save(nn_model.state_dict(), ARTIFACT_DIR / "nn_model.pt")

    config = {
        "feature_columns": list(feature_columns),
        "cleaning_params": cleaning_params,
        "chosen_hyperparameters": chosen_hyperparameters,
        "peak_hours": PEAK_HOURS,
        "loss_history_manual": [round(x, 3) for x in manual_history],
        "loss_history_nn": [round(x, 3) for x in nn_history],
    }
    with open(ARTIFACT_DIR / "config.json", "w") as f:
        json.dump(config, f)


def save_predictions(val, test, val_preds, test_preds):
    """Actual and predicted values for every model, for the dashboard plots."""
    frames = []
    for split_name, part, preds in [("val", val, val_preds), ("test", test, test_preds)]:
        frame = part[["timestamp", "hr", "workingday"]].copy()
        frame["actual"] = part[TARGET]
        frame["split"] = split_name
        for key, pred in preds.items():
            frame[key] = np.clip(pred, 0, None)
        frames.append(frame)
    pd.concat(frames).to_csv(ARTIFACT_DIR / "predictions.csv", index=False)


def save_processed_data(train, val, test):
    """The cleaned data with its split label, for the distribution plots."""
    columns = ["timestamp", "yr", "mnth", "hr", "workingday", "weathersit",
               "temp", "hum", "windspeed", TARGET]
    parts = []
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        part = part[columns].copy()
        part["split"] = split_name
        parts.append(part)
    pd.concat(parts).to_csv(ARTIFACT_DIR / "processed_data.csv", index=False)


def save_metrics(val_results, test_results):
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump({"validation": val_results, "test": test_results}, f, indent=2)


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

    print("\nmanual PyTorch loop:")
    w_manual, b_manual, manual_history, manual_lr = fit_manual(
        Xs_train, y_train, Xs_val, y_val, val
    )
    manual_val_pred = predict_manual(to_tensor(Xs_val), w_manual, b_manual)
    results.append(evaluate_predictions("manual PyTorch", y_val, manual_val_pred, val))

    # Do the two implementations agree? Same model, same data: they should.
    sk_val_pred = sk_model.predict(Xs_val)
    print(f"  max |manual - scikit-learn| prediction gap on validation: "
          f"{np.abs(manual_val_pred - sk_val_pred).max():.2f} bikes")
    print(f"  max |manual - scikit-learn| weight gap: "
          f"{np.abs(w_manual.numpy() - sk_model.coef_).max():.3f}")

    print("\nnn.Module + torch.optim:")
    nn_model, nn_history, nn_lr = fit_nn(Xs_train, y_train, Xs_val, y_val, val)
    nn_val_pred = predict_nn(nn_model, to_tensor(Xs_val))
    results.append(evaluate_predictions("nn.Module + optim", y_val, nn_val_pred, val))

    print(f"  max |nn - scikit-learn| prediction gap on validation: "
          f"{np.abs(nn_val_pred - sk_val_pred).max():.2f} bikes")
    print(f"  max |nn - manual| prediction gap on validation: "
          f"{np.abs(nn_val_pred - manual_val_pred).max():.2f} bikes")
    print(f"  max |nn - scikit-learn| weight gap: "
          f"{np.abs(nn_model.weight.detach().numpy().ravel() - sk_model.coef_).max():.3f}")

    print("\nvalidation results (bikes per hour):")
    print(pd.DataFrame(results).round(1).to_string(index=False))

    # ---- Step 7: the test set, looked at for the first and only time. ----
    # Every choice (alpha, learning rates) was made on validation above, and the
    # models were trained on train only, so this number is an honest estimate.
    mean_test, hour_test = baseline_predictions(train, test)
    X_test_t = to_tensor(Xs_test)
    test_preds = {
        "baseline_mean": mean_test,
        "baseline_per_hour": hour_test,
        "sklearn": sk_model.predict(Xs_test),
        "manual": predict_manual(X_test_t, w_manual, b_manual),
        "nn": predict_nn(nn_model, X_test_t),
    }
    val_preds = {
        "baseline_mean": mean_val,
        "baseline_per_hour": hour_val,
        "sklearn": sk_val_pred,
        "manual": manual_val_pred,
        "nn": nn_val_pred,
    }
    val_results = results_table(val, val_preds)
    test_results = results_table(test, test_preds)

    print("\nTEST results (bikes per hour):")
    print(pd.DataFrame(test_results).round(1).to_string(index=False))
    print(f"  max |manual - scikit-learn| prediction gap on test: "
          f"{np.abs(test_preds['manual'] - test_preds['sklearn']).max():.2f} bikes")
    print(f"  max |nn - scikit-learn| prediction gap on test: "
          f"{np.abs(test_preds['nn'] - test_preds['sklearn']).max():.2f} bikes")

    # ---- Save everything app.py needs, so the app never retrains. ----
    save_models(
        sk_model, scaler, w_manual, b_manual, manual_history, nn_model, nn_history,
        X_train.columns, cleaning_params,
        {"alpha": sk_model.alpha, "manual_lr": manual_lr, "nn_lr": nn_lr,
         "n_epochs": N_EPOCHS},
    )
    save_predictions(val, test, val_preds, test_preds)
    save_processed_data(train, val, test)
    save_metrics(val_results, test_results)
    print("\nsaved to", ARTIFACT_DIR.name + "/:", sorted(p.name for p in ARTIFACT_DIR.iterdir()))

    print("\nstep 7 done")


if __name__ == "__main__":
    main()