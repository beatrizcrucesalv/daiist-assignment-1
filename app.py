"""
Assignment 1 — Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file — main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py app`, this file must build and launch a Gradio
app (a `demo` that calls `.launch()`) using only what your training stage
already produced — it must never retrain anything itself. At minimum, the
dashboard must let you:

- Compare your three trained models via a prediction-vs-actual plot.
- See feature and/or target distributions.
- For classification: move a decision-threshold slider and watch the
  confusion matrix, and a business-cost number tied to your REPORT.md's
  framing, change with it.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""
"""
Assignment 1 — Gradio dashboard.

Loads ONLY what train.py saved in artifacts/ (the three trained models, the
scaler, predictions, metrics and the cleaned data). Nothing is trained here.

Run with:  uv run python main.py app

Tabs:
  1. Prediction vs actual for the three models (validation or test set)
  2. Daily pattern: average rentals per hour, actual vs predicted
  3. Distributions of the target and the features, per split
  4. Metrics tables (validation and test)
  5. Try it: the saved models predict demand for a situation you choose
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

# Same feature code that built the training data, so inputs are encoded identically.
from train import ARTIFACT_DIR, MODEL_NAMES, build_features

# ---- Load everything train.py produced (no training happens here) ----
config = json.loads((ARTIFACT_DIR / "config.json").read_text())
metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
predictions = pd.read_csv(ARTIFACT_DIR / "predictions.csv")
processed = pd.read_csv(ARTIFACT_DIR / "processed_data.csv")

scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
sk_model = joblib.load(ARTIFACT_DIR / "sklearn_model.joblib")
manual_model = torch.load(ARTIFACT_DIR / "manual_model.pt")  # {"w": ..., "b": ...}
nn_model = nn.Linear(len(config["feature_columns"]), 1)
nn_model.load_state_dict(torch.load(ARTIFACT_DIR / "nn_model.pt"))
nn_model.eval()

MODEL_KEYS = ["sklearn", "manual", "nn"]
DIST_COLUMNS = ["cnt", "hr", "temp", "hum", "windspeed", "mnth", "weathersit"]


# ---- Tab 1: prediction vs actual ----
def plot_pred_vs_actual(split):
    data = predictions[predictions["split"] == split]
    top = float(max(data["actual"].max(), data[MODEL_KEYS].max().max()))

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=[MODEL_NAMES[k] for k in MODEL_KEYS],
    )
    for col, key in enumerate(MODEL_KEYS, start=1):
        fig.add_trace(
            go.Scatter(x=data["actual"], y=data[key], mode="markers",
                       marker=dict(size=4, opacity=0.4), showlegend=False),
            row=1, col=col,
        )
        # Perfect prediction = the diagonal.
        fig.add_trace(
            go.Scatter(x=[0, top], y=[0, top], mode="lines",
                       line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="actual rentals per hour", row=1, col=col)
    fig.update_yaxes(title_text="predicted rentals per hour", row=1, col=1)
    fig.update_layout(title=f"Predicted vs actual ({split} set)", height=420)
    return fig


# ---- Tab 2: daily pattern ----
def plot_hourly(split):
    data = predictions[predictions["split"] == split]
    fig = go.Figure()
    for flag, label in [(1, "working day"), (0, "weekend / holiday")]:
        by_hour = data[data["workingday"] == flag].groupby("hr")[["actual", "sklearn"]].mean()
        fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour["actual"],
                                 mode="lines", name=f"actual, {label}"))
        fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour["sklearn"],
                                 mode="lines", line=dict(dash="dash"),
                                 name=f"predicted, {label}"))
    fig.update_layout(
        title=f"Average rentals by hour of day ({split} set; the three models give the same curve)",
        xaxis_title="hour of day", yaxis_title="average rentals per hour", height=420,
    )
    return fig


# ---- Tab 3: distributions ----
def plot_distribution(column):
    fig = px.histogram(
        processed, x=column, color="split", barmode="overlay",
        histnorm="percent", opacity=0.6,
        category_orders={"split": ["train", "val", "test"]},
    )
    fig.update_layout(title=f"Distribution of {column} by split", yaxis_title="% of rows", height=420)
    return fig


# ---- Tab 4: metrics ----
def metrics_table(which):
    return pd.DataFrame(metrics[which]).round(1)


# ---- Tab 5: try it (uses the saved models) ----
def predict_demand(hour, working_day, month, weather, temp_c, humidity_pct, wind, year):
    # Convert what a person types into the raw columns the pipeline expects.
    row = pd.DataFrame({
        "yr": [1 if year == 2012 else 0],
        "mnth": [int(month)],
        "hr": [int(hour)],
        "workingday": [int(working_day)],
        "weathersit": [int(weather)],
        "temp": [(temp_c + 8) / 47],      # dataset scaling: (t + 8) / (39 + 8)
        "hum": [humidity_pct / 100],
        "windspeed": [wind / 67],
    })
    X = build_features(row)[config["feature_columns"]]
    Xs = scaler.transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32)

    with torch.no_grad():
        predicted = {
            "sklearn": float(sk_model.predict(Xs)[0]),
            "manual": float((Xt @ manual_model["w"] + manual_model["b"]).item()),
            "nn": float(nn_model(Xt).item()),
        }
    # A rental count cannot be negative.
    return pd.DataFrame({
        "model": [MODEL_NAMES[k] for k in MODEL_KEYS],
        "predicted rentals in that hour": [round(max(predicted[k], 0), 1) for k in MODEL_KEYS],
    })


# ---- Layout ----
with gr.Blocks(title="Bike Sharing demand dashboard") as demo:
    gr.Markdown(
        "# Bike Sharing: hourly demand\n"
        "One linear regression trained three ways (scikit-learn, manual PyTorch loop, "
        "`nn.Module` + `torch.optim`). Everything shown here was produced by `train.py`."
    )

    with gr.Tab("Prediction vs actual"):
        split_a = gr.Radio(["val", "test"], value="test", label="Split")
        plot_a = gr.Plot()
        split_a.change(plot_pred_vs_actual, split_a, plot_a)
        demo.load(plot_pred_vs_actual, split_a, plot_a)

    with gr.Tab("Daily pattern"):
        split_b = gr.Radio(["val", "test"], value="test", label="Split")
        plot_b = gr.Plot()
        split_b.change(plot_hourly, split_b, plot_b)
        demo.load(plot_hourly, split_b, plot_b)

    with gr.Tab("Distributions"):
        column = gr.Dropdown(DIST_COLUMNS, value="cnt", label="Column")
        plot_c = gr.Plot()
        column.change(plot_distribution, column, plot_c)
        demo.load(plot_distribution, column, plot_c)

    with gr.Tab("Metrics"):
        gr.Markdown("Validation (used to choose alpha and learning rates):")
        gr.Dataframe(value=metrics_table("validation"))
        gr.Markdown("Test (looked at once, at the end):")
        gr.Dataframe(value=metrics_table("test"))

    with gr.Tab("Try it"):
        gr.Markdown("The three saved models predict the rentals in one hour you describe.")
        with gr.Row():
            hour = gr.Slider(0, 23, value=8, step=1, label="Hour of day")
            working_day = gr.Checkbox(value=True, label="Working day")
            month = gr.Slider(1, 12, value=6, step=1, label="Month")
            year = gr.Radio([2011, 2012], value=2012, label="Year")
        with gr.Row():
            weather = gr.Radio([1, 2, 3], value=1, label="Weather (1 clear, 2 mist/cloudy, 3 rain/snow)")
            temp_c = gr.Slider(-8, 39, value=20, step=1, label="Temperature (°C)")
            humidity = gr.Slider(0, 100, value=60, step=1, label="Humidity (%)")
            wind = gr.Slider(0, 67, value=15, step=1, label="Wind speed (dataset units)")
        button = gr.Button("Predict")
        result = gr.Dataframe(label="Predictions")
        button.click(predict_demand, [hour, working_day, month, weather, temp_c, humidity, wind, year], result)


if __name__ == "__main__":
    demo.launch()