"""
forecast.py  -  shared helper functions for the FORECASTING part.
Used by traffic_app.py and mobility_management.py so the same code is not repeated.
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf

LOOKBACK, HORIZON, TARGET = 48, 6, "traffic_volume"      # must match the training script
DATA_FILE   = "Metro_Interstate_Traffic_Volume.csv.gz"
MODEL_FILE  = "best_traffic_model.keras"
SCALER_FILE = "scaler.json"
INPUT_COLUMNS = ["date_time", "traffic_volume", "temp_c", "rain_1h", "snow_1h",
                 "clouds_all", "weather_main", "is_holiday"]


def load_model_and_scaler():
    """Load the trained model and the numbers used to scale the data."""
    model = tf.keras.models.load_model(MODEL_FILE)
    with open(SCALER_FILE) as f:
        scaler = json.load(f)
    columns = scaler["columns"]
    return model, columns, pd.Series(scaler["mean"])[columns], pd.Series(scaler["std"])[columns]


def load_hourly_data(path=DATA_FILE):
    """Read the traffic dataset and create one clean row per hour."""

    print(f"Loading dataset: {path}")

    # ---------------------------------------------------------------
    # 1. Read dataset
    # ---------------------------------------------------------------
    raw = pd.read_csv(path)

    print("Dataset shape:", raw.shape)
    print("Columns:", raw.columns.tolist())

    # ---------------------------------------------------------------
    # 2. Check required columns
    # ---------------------------------------------------------------
    required = [
        "date_time",
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "holiday",
        "weather_main"
    ]

    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {raw.columns.tolist()}"
        )

    # ---------------------------------------------------------------
    # 3. Convert date_time
    # ---------------------------------------------------------------
    raw["date_time"] = pd.to_datetime(
        raw["date_time"],
        errors="coerce"
    )

    # Remove rows where date is invalid
    raw = raw.dropna(subset=["date_time"])

    # Sort chronologically
    raw = raw.sort_values("date_time")

    # ---------------------------------------------------------------
    # 4. Temperature: Kelvin -> Celsius
    # ---------------------------------------------------------------
    raw["temp_c"] = raw["temp"] - 273.15

    # Remove impossible temperature values
    raw.loc[raw["temp"] < 200, "temp_c"] = np.nan

    # ---------------------------------------------------------------
    # 5. Rain
    # ---------------------------------------------------------------
    raw["rain_1h"] = raw["rain_1h"].clip(
        lower=0,
        upper=100
    )

    # ---------------------------------------------------------------
    # 6. Holiday
    # ---------------------------------------------------------------
    raw["is_holiday"] = (
        raw["holiday"]
        .fillna("None")
        .astype(str)
        .str.lower()
        != "none"
    ).astype(int)

    # ---------------------------------------------------------------
    # 7. Convert to one row per hour
    # ---------------------------------------------------------------
    hourly = (
        raw.groupby("date_time")
        .agg(
            traffic_volume=("traffic_volume", "mean"),
            temp_c=("temp_c", "mean"),
            rain_1h=("rain_1h", "max"),
            snow_1h=("snow_1h", "max"),
            clouds_all=("clouds_all", "mean"),
            is_holiday=("is_holiday", "max"),
            weather_main=("weather_main", "first")
        )
        .asfreq("h")
    )

    # ---------------------------------------------------------------
    # 8. Fill missing numeric values
    # ---------------------------------------------------------------
    numeric = [
        "traffic_volume",
        "temp_c",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "is_holiday"
    ]

    hourly[numeric] = (
        hourly[numeric]
        .interpolate(
            method="linear",
            limit=6,
            limit_direction="both"
        )
    )

    # ---------------------------------------------------------------
    # 9. Traffic volume is essential.
    #    Fill any remaining missing traffic values.
    # ---------------------------------------------------------------
    hourly["traffic_volume"] = (
        hourly["traffic_volume"]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    # ---------------------------------------------------------------
    # 10. Weather
    # ---------------------------------------------------------------
    hourly["weather_main"] = (
        hourly["weather_main"]
        .ffill(limit=6)
        .bfill(limit=6)
    )

    # If weather still has missing values, use "Clear"
    hourly["weather_main"] = (
        hourly["weather_main"]
        .fillna("Clear")
    )

    # ---------------------------------------------------------------
    # 11. Final safety check
    # ---------------------------------------------------------------
    print("\nHourly dataset:")
    print("Shape:", hourly.shape)
    print("Start:", hourly.index.min())
    print("End:", hourly.index.max())

    print("\nRemaining missing values:")
    print(hourly.isna().sum())

    print("\nTraffic statistics:")
    print(hourly["traffic_volume"].describe())

    # VERY IMPORTANT
    return hourly

def get_input_window(hourly, now):
    """Return exactly the previous 48 hourly records."""

    now = pd.Timestamp(now)

    # Make sure index is datetime
    hourly = hourly.copy()
    hourly.index = pd.to_datetime(hourly.index)
    hourly = hourly.sort_index()

    # Check requested time
    if now not in hourly.index:
        print(f"Time {now} is not available in dataset.")
        return None

    # Get 48 hours including 'now'
    window = hourly.loc[:now].tail(LOOKBACK)

    # Check we actually got 48 rows
    if len(window) < LOOKBACK:
        print(
            f"Only {len(window)} rows found before {now}. "
            f"Need {LOOKBACK}."
        )
        return None

    # Reset index
    window = window.reset_index()

    # Rename columns
    window.columns = INPUT_COLUMNS

    # Convert holiday to bool
    window["is_holiday"] = window["is_holiday"].astype(bool)

    # Check missing values
    missing = window.drop(columns=["date_time"]).isna().sum()

    if missing.any():
        print("Missing values in input window:")
        print(missing[missing > 0])
        return None

    return window

def build_model_input(window, columns, mean, std):
    """Turn the 48-row table into the scaled array the model expects."""
    df = window.set_index("date_time").copy()
    df["is_holiday"] = df["is_holiday"].astype(int)

    for col in columns:                                   # weather type -> 0/1 columns
        if col.startswith("weather_"):
            df[col] = (df["weather_main"] == col[len("weather_"):]).astype(float)

    hour, day, month = df.index.hour, df.index.dayofweek, df.index.month   # time features
    df["hour_sin"],  df["hour_cos"]  = np.sin(2*np.pi*hour/24),  np.cos(2*np.pi*hour/24)
    df["day_sin"],   df["day_cos"]   = np.sin(2*np.pi*day/7),    np.cos(2*np.pi*day/7)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    df["is_weekend"] = (day >= 5).astype(int)

    scaled = ((df[columns].astype(float) - mean) / std).values.astype("float32")
    return scaled[np.newaxis]                              # shape (1, 48, features)


def predict_next_hours(model, window, columns, mean, std):
    """Forecast the next 6 hours (vehicles/hour) from a 48-hour window."""
    x = build_model_input(window, columns, mean, std)
    return np.clip(model.predict(x, verbose=0)[0] * std[TARGET] + mean[TARGET], 0, None)