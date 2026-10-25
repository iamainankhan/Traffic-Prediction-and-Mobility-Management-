import io, json, os, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks

tf.keras.utils.set_random_seed(42)          # same results every run


# =====================================================================
# STEP 1: SETTINGS
# =====================================================================
LOOKBACK = 48      # how many past hours the model sees
HORIZON  = 6       # how many future hours the model predicts
EPOCHS   = \
    30     # maximum training rounds (early stopping may stop sooner)
BATCH    = 64      # samples processed at once
TARGET   = "traffic_volume"

URL   = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
LOCAL = "Metro_Interstate_Traffic_Volume.csv"   # optional: put the CSV next to this file


# =====================================================================
# STEP 2: LOAD THE DATASET
# =====================================================================
def load_raw_data():
    """Read the CSV from disk if present, otherwise download it from UCI."""
    for path in (LOCAL, LOCAL + ".gz"):
        if os.path.exists(path):
            return pd.read_csv(path)
    print("Downloading dataset from UCI ...")
    with urllib.request.urlopen(URL) as response:
        zf = zipfile.ZipFile(io.BytesIO(response.read()))
    name = [n for n in zf.namelist() if "Traffic_Volume" in n][0]
    with zf.open(name) as f:
        return pd.read_csv(f, compression="gzip" if name.endswith(".gz") else None)

raw = load_raw_data()
raw["date_time"] = pd.to_datetime(raw["date_time"])
print("Raw dataset:", raw.shape[0], "rows")


# =====================================================================
# STEP 3: CLEAN THE DATA
# =====================================================================
# Problems in this dataset: duplicate timestamps, missing hours,
# sensor errors (temperature = 0 Kelvin) and one absurd rain value.
raw = raw.sort_values("date_time")
raw["temp_c"]     = raw["temp"] - 273.15                      # Kelvin -> Celsius
raw.loc[raw["temp"] < 200, "temp_c"] = np.nan                 # 0 K = sensor error
raw["rain_1h"]    = raw["rain_1h"].clip(upper=100)            # remove 9831 mm outlier
raw["is_holiday"] = (raw["holiday"].fillna("None") != "None").astype(int)

# Turn the weather type (Clear, Rain, Snow ...) into 0/1 columns
weather = pd.get_dummies(raw["weather_main"], prefix="weather").astype(float)
raw = pd.concat([raw, weather], axis=1)

# Several rows can exist for the same hour -> merge them into one row per hour
how_to_merge = {TARGET: "mean", "temp_c": "mean", "rain_1h": "max", "snow_1h": "max",
                "clouds_all": "mean", "is_holiday": "max"}
how_to_merge.update({col: "max" for col in weather.columns})
df = raw.groupby("date_time").agg(how_to_merge)

# Put the data on a perfect hourly timeline and fill SHORT gaps (up to 6 hours)
df = df.asfreq("h")
print("Missing hours found:", int(df[TARGET].isna().sum()))
df = df.interpolate(limit=6, limit_direction="both")

# Keep only the longest period without any remaining gap
good_row = df.notna().all(axis=1)
block_id = (~good_row).cumsum()                 # a new block starts after every bad row
longest  = good_row.groupby(block_id).sum().idxmax()
df = df[good_row & (block_id == longest)].copy()
print(f"Using {len(df)} continuous hours: {df.index[0]}  ->  {df.index[-1]}")


# =====================================================================
# STEP 4: FEATURE ENGINEERING (time features)
# =====================================================================
# Hour 23 and hour 0 are close in time but far as numbers (23 vs 0).
# Sine/cosine puts them next to each other on a circle, so the model understands "cycles".
hour, day, month = df.index.hour, df.index.dayofweek, df.index.month
df["hour_sin"],  df["hour_cos"]  = np.sin(2*np.pi*hour/24),  np.cos(2*np.pi*hour/24)
df["day_sin"],   df["day_cos"]   = np.sin(2*np.pi*day/7),    np.cos(2*np.pi*day/7)
df["month_sin"], df["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
df["is_weekend"] = (day >= 5).astype(int)

target_col = df.columns.get_loc(TARGET)          # position of traffic_volume column
print("Features used:", list(df.columns))


# =====================================================================
# STEP 5: SPLIT (by time!) AND SCALE
# =====================================================================
# We never shuffle time series: train = past, validation = middle, test = future.
n = len(df)
train_df = df.iloc[: int(n * 0.70)]
val_df   = df.iloc[int(n * 0.70): int(n * 0.85)]
test_df  = df.iloc[int(n * 0.85):]

# Scale every column to mean 0 / std 1 using ONLY training data (avoids data leakage)
mean = train_df.mean()
std  = train_df.std().replace(0, 1)
scale = lambda part: ((part - mean) / std).values.astype("float32")
train, val, test = scale(train_df), scale(val_df), scale(test_df)

# Save the scaling numbers + column order: the prediction script needs them later
with open("scaler.json", "w") as f:
    json.dump({"columns": list(df.columns), "mean": mean.to_dict(), "std": std.to_dict()}, f)

def to_vehicles(scaled):
    """Convert scaled traffic back to real 'vehicles per hour'."""
    return scaled * std[TARGET] + mean[TARGET]


# =====================================================================
# STEP 6: SLIDING WINDOWS
# =====================================================================
# One example = (past 48 hours of all features)  ->  (next 6 hours of traffic)
def make_windows(data):
    X, y = [], []
    for start in range(len(data) - LOOKBACK - HORIZON + 1):
        X.append(data[start : start + LOOKBACK])
        y.append(data[start + LOOKBACK : start + LOOKBACK + HORIZON, target_col])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train)
X_val,   y_val   = make_windows(val)
X_test,  y_test  = make_windows(test)
input_shape = (LOOKBACK, X_train.shape[-1])
print(f"Windows -> train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")


# =====================================================================
# STEP 7: THE 5 MODELS
# =====================================================================
# All models take the same input and output 6 numbers (the next 6 hours).

def build_mlp():
    """MLP: flattens the 48 hours into one long vector. Simple, ignores time order."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(HORIZON)])

def build_cnn():
    """1D-CNN: slides small filters over time to detect short local patterns (e.g. a rush-hour rise)."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation="relu"),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(HORIZON)])

def build_lstm():
    """LSTM: reads the hours one by one and keeps a memory (gates) of important past events."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(HORIZON)])

def build_gru():
    """GRU: like LSTM but with fewer gates -> fewer parameters, usually faster to train."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(HORIZON)])

def build_cnn_lstm():
    """CNN-LSTM: CNN extracts local patterns first, then LSTM learns how they evolve over time."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(HORIZON)])

MODEL_BUILDERS = {"MLP": build_mlp, "1D-CNN": build_cnn, "LSTM": build_lstm,
                  "GRU": build_gru, "CNN-LSTM": build_cnn_lstm}


# =====================================================================
# STEP 8: TRAIN ALL MODELS
# =====================================================================
trained, histories, train_seconds = {}, {}, {}

for name, builder in MODEL_BUILDERS.items():
    print(f"\n========== Training {name} ==========")
    model = builder()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    stop_early = callbacks.EarlyStopping(patience=6, restore_best_weights=True)   # stop when no improvement
    lower_lr   = callbacks.ReduceLROnPlateau(patience=3, factor=0.5)              # slow down learning when stuck

    start = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH, verbose=2,
                        callbacks=[stop_early, lower_lr])
    train_seconds[name] = time.time() - start
    trained[name], histories[name] = model, history.history


# =====================================================================
# STEP 9: EVALUATE AND COMPARE
# =====================================================================
def metrics(y_true, y_pred):
    """Four standard forecasting errors, all in real units (vehicles/hour)."""
    error = y_pred - y_true
    return {"MAE":  np.mean(np.abs(error)),                                   # average error
            "RMSE": np.sqrt(np.mean(error ** 2)),                             # punishes big mistakes
            "MAPE": np.mean(np.abs(error) / np.maximum(y_true, 50)) * 100,    # % error (ignores tiny volumes)
            "R2":   1 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)}

true_val, true_test = to_vehicles(y_val), to_vehicles(y_test)

# Baseline: copy the traffic of the same hours 24 hours earlier (positions 24..29 in the window)
def seasonal_naive(X):
    return to_vehicles(X[:, 24 : 24 + HORIZON, target_col])

predictions = {"Seasonal Naive": seasonal_naive(X_test)}
val_mae     = {"Seasonal Naive": metrics(true_val, seasonal_naive(X_val))["MAE"]}
for name, model in trained.items():
    predictions[name] = to_vehicles(model.predict(X_test, verbose=0))
    val_mae[name]     = metrics(true_val, to_vehicles(model.predict(X_val, verbose=0)))["MAE"]

rows = []
for name, pred in predictions.items():
    m = metrics(true_test, pred)
    rows.append({"Model": name, "Val MAE": val_mae[name], "Test MAE": m["MAE"],
                 "Test RMSE": m["RMSE"], "Test MAPE %": m["MAPE"], "Test R2": m["R2"],
                 "Parameters": trained[name].count_params() if name in trained else 0,
                 "Train time (s)": train_seconds.get(name, 0)})
results = pd.DataFrame(rows).set_index("Model").round(3)

print("\n================ FINAL COMPARISON (test set = unseen future data) ================")
print(results.sort_values("Test MAE").to_string())
results.to_csv("model_comparison_results.csv")


# =====================================================================
# STEP 10: CHOOSE THE BEST MODEL AND EXPLAIN WHY
# =====================================================================
# Rule: the best model is the one with the LOWEST VALIDATION MAE.
# (We choose on validation data, then report on test data -> an honest, unbiased score.)
deep = results.drop(index="Seasonal Naive")
best = deep["Val MAE"].idxmin()
ranking = deep.sort_values("Val MAE")
runner_up = ranking.index[1]

gain_vs_baseline = 100 * (1 - results.loc[best, "Test MAE"] / results.loc["Seasonal Naive", "Test MAE"])
gain_vs_runner   = 100 * (1 - results.loc[best, "Test MAE"] / results.loc[runner_up, "Test MAE"])
gap = abs(results.loc[best, "Test MAE"] - results.loc[best, "Val MAE"])

NOTES = {"MLP":      "treats the 48 hours as an unordered list, so it cannot model time order directly",
         "1D-CNN":   "captures short local patterns but has no long-term memory",
         "LSTM":     "uses gated memory to remember daily/weekly patterns over the whole window",
         "GRU":      "a lighter recurrent network with similar memory ability",
         "CNN-LSTM": "combines local pattern extraction (CNN) with sequence memory (LSTM)"}

print("\n================ WHY THIS MODEL? ================")
print(f"Selected model: {best}")
print(f"1. Lowest validation MAE of all 5 models ({results.loc[best,'Val MAE']:.1f} vehicles/hour).")
print(f"2. On unseen test data its MAE is {results.loc[best,'Test MAE']:.1f} vehicles/hour, "
      f"{gain_vs_baseline:.1f}% better than the seasonal-naive baseline.")
print(f"3. It beats the runner-up ({runner_up}) by {gain_vs_runner:.1f}% in test MAE.")
print(f"4. Validation and test MAE differ by only {gap:.1f}, so it generalises (no heavy overfitting).")
print(f"5. Architecture reason: {best} {NOTES[best]}.")
print(f"6. Cost: {results.loc[best,'Parameters']:,.0f} parameters, trained in {results.loc[best,'Train time (s)']:.0f} s.")

trained[best].save("best_traffic_model.keras")


# =====================================================================
# STEP 11: DIAGRAMS
# =====================================================================
COLORS = {"Seasonal Naive": "grey", "MLP": "tab:orange", "1D-CNN": "tab:green",
          "LSTM": "tab:blue", "GRU": "tab:red", "CNN-LSTM": "tab:purple"}
order = results.sort_values("Test MAE").index.tolist()

# ---- Figure 1: error comparison bar charts -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, title in zip(axes, ["Test MAE", "Test RMSE", "Test R2"],
                          ["MAE (lower is better)", "RMSE (lower is better)", "R-squared (higher is better)"]):
    values = results.loc[order, col]
    bars = ax.bar(order, values, color=[COLORS[m] for m in order], edgecolor="black")
    for m, bar in zip(order, bars):
        if m == best: bar.set_linewidth(3)                       # thick border on the winner
    ax.bar_label(bars, fmt="%.2f" if col == "Test R2" else "%.0f", fontsize=9)
    ax.set_title(title); ax.tick_params(axis="x", rotation=30)
fig.suptitle(f"Model performance on the test set (thick border = chosen model: {best})", fontweight="bold")
plt.tight_layout(); plt.savefig("fig1_model_comparison.png", dpi=150)

# ---- Figure 2: predictions vs actual traffic (first week of the test set) ----------------
fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
for ax, step, label in zip(axes, [0, HORIZON - 1], ["1 hour ahead", f"{HORIZON} hours ahead"]):
    ax.plot(true_test[:168, step], color="black", linewidth=2, label="Actual")
    ax.plot(predictions["Seasonal Naive"][:168, step], color="grey", linestyle="--", label="Seasonal Naive")
    ax.plot(predictions[best][:168, step], color=COLORS[best], linewidth=1.8, label=f"{best} (best)")
    ax.set_title(f"Forecast {label} - first 7 days of test data"); ax.set_ylabel("vehicles / hour"); ax.legend()
axes[1].set_xlabel("hours")
plt.tight_layout(); plt.savefig("fig2_actual_vs_predicted.png", dpi=150)

# ---- Figure 3: how error grows with forecast distance ---------------------------------------
plt.figure(figsize=(9, 5))
for name, pred in predictions.items():
    per_step_mae = np.mean(np.abs(pred - true_test), axis=0)
    plt.plot(range(1, HORIZON + 1), per_step_mae, marker="o", color=COLORS[name],
             linewidth=3 if name == best else 1.5, label=name)
plt.xlabel("Hours ahead"); plt.ylabel("MAE (vehicles/hour)")
plt.title("Error vs forecast distance"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig3_error_by_horizon.png", dpi=150)

# ---- Figure 4: learning curves of every model ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7)); axes = axes.ravel()
for ax, name in zip(axes, trained):
    ax.plot(histories[name]["loss"], label="train"); ax.plot(histories[name]["val_loss"], label="validation")
    ax.set_title(name); ax.set_xlabel("epoch"); ax.set_ylabel("loss (MSE)"); ax.legend()
axes[-1].axis("off")
plt.suptitle("Learning curves (gap between lines = overfitting)", fontweight="bold")
plt.tight_layout(); plt.savefig("fig4_learning_curves.png", dpi=150)

# ---- Figure 5: accuracy vs training cost -------------------------------------------------------
plt.figure(figsize=(8, 5))
for name in trained:
    plt.scatter(results.loc[name, "Train time (s)"], results.loc[name, "Test MAE"],
                s=results.loc[name, "Parameters"] / 300 + 60, color=COLORS[name], edgecolor="black", label=name)
plt.xlabel("Training time (s)"); plt.ylabel("Test MAE (lower is better)")
plt.title("Accuracy vs training cost (bubble size = parameters)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig5_accuracy_vs_cost.png", dpi=150)

plt.show()
print("\nSaved: model_comparison_results.csv, best_traffic_model.keras, scaler.json, fig1..fig5 (.png)")