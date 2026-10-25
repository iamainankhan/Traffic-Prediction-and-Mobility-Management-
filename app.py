"""
=====================================================================
 TRAFFIC FORECASTING + MOBILITY MANAGEMENT APP  (Streamlit interface)
=====================================================================
Tab 1: choose a date/hour (optionally edit the last 48 hours) -> model forecasts 6 hours
Tab 2: mobility management -> congestion level, diversion, actions, best departure time
Tab 3: comparison of the 5 trained models

Files needed in the same folder:
    forecast_core.py, mobility_management.py,
    best_traffic_model.keras, scaler.json, Metro_Interstate_Traffic_Volume.csv
    (+ model_comparison_results.csv and fig*.png for tab 3)

Run:   pip install streamlit
       streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import forecast as fc              # forecasting helpers
import mobility as mm        # mobility management logic

# Cache the heavy loading steps so they only happen once
load_model_and_scaler = st.cache_resource(fc.load_model_and_scaler)
load_hourly_data = st.cache_data(fc.load_hourly_data)

# ---------------------------------------------------------------------
# PAGE SET-UP
# ---------------------------------------------------------------------
st.set_page_config(page_title="Traffic Forecasting & Mobility Management", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Forecasting & Mobility Management")
st.caption("Deep-learning model trained on the Metro Interstate (I-94) dataset: it looks at the last 48 hours, "
           "predicts the next 6 hours, and the mobility module turns the forecast into traffic-control decisions.")

missing = [f for f in (fc.DATA_FILE, fc.MODEL_FILE, fc.SCALER_FILE) if not os.path.exists(f)]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Run train.py first "
             "and keep this app in the same folder.")
    st.stop()

model, columns, mean, std = load_model_and_scaler()
hourly = load_hourly_data(fc.DATA_FILE)
weather_types = [c[len("weather_"):] for c in columns if c.startswith("weather_")]

tab_predict, tab_mobility, tab_compare = st.tabs(
    ["🔮 1. Forecast traffic", "🚦 2. Mobility management", "📊 3. Model comparison"])

# ======================= TAB 1: FORECAST =======================
with tab_predict:
    # ---- Step 1: user chooses the moment ("now") ----
    st.sidebar.header("1. Choose the time")
    last_allowed, first_allowed = hourly.index[-1 - fc.HORIZON], hourly.index[fc.LOOKBACK]
    day = st.sidebar.date_input("Date", value=last_allowed.date() - pd.Timedelta(days=30),
                                min_value=first_allowed.date(), max_value=last_allowed.date())
    hour = st.sidebar.selectbox("Hour of the day (forecast starts after this hour)", range(24), index=15,
                                format_func=lambda h: f"{h:02d}:00")
    now = pd.Timestamp(day) + pd.Timedelta(hours=hour)

    st.sidebar.header("2. What-if mode")
    edit_mode = st.sidebar.checkbox("Edit the input data myself", value=False,
                                    help="Change traffic or weather values and see how the forecast reacts.")

    # ---- Take the 48 hours before 'now' ----
    window = fc.get_input_window(hourly, now) if now <= last_allowed else None
    if window is None:
        st.warning("This time is outside the dataset or has gaps in the previous 48 hours. "
                   "Please choose another date/hour.")
        st.stop()

    # ---- Step 2 (optional): the user edits the input ----
    if edit_mode:
        st.subheader("Input: last 48 hours (editable)")
        st.write("Double-click a cell to change it. Example: lower `traffic_volume` to simulate an accident, "
                 "or set `weather_main` to *Snow*.")
        window = st.data_editor(
            window, hide_index=True, width="stretch", height=300, disabled=["date_time"],
            column_config={
                "traffic_volume": st.column_config.NumberColumn("Traffic (veh/h)", min_value=0, max_value=10000),
                "temp_c":         st.column_config.NumberColumn("Temp (°C)"),
                "rain_1h":        st.column_config.NumberColumn("Rain (mm)", min_value=0),
                "snow_1h":        st.column_config.NumberColumn("Snow (mm)", min_value=0),
                "clouds_all":     st.column_config.NumberColumn("Clouds (%)", min_value=0, max_value=100),
                "weather_main":   st.column_config.SelectboxColumn("Weather", options=weather_types),
                "is_holiday":     st.column_config.CheckboxColumn("Holiday"),
            })

    # ---- Step 3: predict ----
    prediction = fc.predict_next_hours(model, window, columns, mean, std)
    future_hours = pd.date_range(now + pd.Timedelta(hours=1), periods=fc.HORIZON, freq="h")
    actual = hourly[fc.TARGET].reindex(future_hours)

    # ---- Show results ----
    st.subheader(f"Forecast for the next {fc.HORIZON} hours (after {now:%d %b %Y, %H:%M})")
    capacity = mm.estimate_capacity(hourly)
    cols = st.columns(fc.HORIZON)
    for col, t, p in zip(cols, future_hours, prediction):
        level = mm.congestion_level(p, capacity)
        col.metric(f"{t:%H:%M}", f"{p:,.0f}", help="vehicles per hour")
        col.write(f"{mm.LEVEL_ICONS[level]} {level}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(window["date_time"], window[fc.TARGET], color="black", label="Last 48 hours (input)")
    ax.plot(future_hours, prediction, "o-", color="tab:blue", linewidth=2, label="Predicted")
    show_actual = actual.notna().all() and not edit_mode
    if show_actual:
        ax.plot(future_hours, actual, "s--", color="tab:green", label="Actual (from dataset)")
    ax.axvline(now, color="grey", linestyle=":")
    ax.set_ylabel("vehicles / hour"); ax.grid(alpha=.3); ax.legend()
    st.pyplot(fig)

    if show_actual:
        error = np.abs(prediction - actual.values).mean()
        st.success(f"Average error on this forecast: **{error:,.0f} vehicles/hour** "
                   f"(real traffic was between {actual.min():,.0f} and {actual.max():,.0f}).")
    elif edit_mode:
        st.info("What-if mode: no 'actual' values are shown because you may have changed the input.")

# ======================= TAB 2: MOBILITY MANAGEMENT =======================
with tab_mobility:
    st.subheader("From forecast to decisions")
    st.write("The forecast from tab 1 is used to decide **before** congestion happens: "
             "how full the road is, how much traffic to divert, and what the traffic authority should do.")

    with st.expander("Assumptions (you can change them)", expanded=False):
        c1, c2, c3 = st.columns(3)
        mm.MAIN_FREE_FLOW_MIN = c1.number_input("Main road trip time when empty (min)", 5, 120, mm.MAIN_FREE_FLOW_MIN)
        mm.ALT_FREE_FLOW_MIN  = c2.number_input("Alternate route trip time when empty (min)", 5, 120, mm.ALT_FREE_FLOW_MIN)
        mm.ALT_CAPACITY       = c3.number_input("Alternate route capacity (veh/h)", 500, 10000, mm.ALT_CAPACITY, step=100)
        capacity = st.number_input("Main road capacity (veh/h)", 1000, 12000, int(capacity), step=100,
                                   help="By default: the busiest traffic seen in the dataset (99th percentile).")

    plan = mm.create_mobility_plan(prediction, future_hours, capacity)

    best, worst = mm.best_departure_hour(plan), plan.loc[plan["Predicted volume"].idxmax()]
    saved = (plan["Minutes saved per driver"] * plan["Predicted volume"]).sum() / 60
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak traffic", f"{worst['Time']:%H:%M}", f"{worst['Predicted volume']:,} veh/h", delta_color="off")
    m2.metric("Best departure time", f"{best['Time']:%H:%M}", f"{best['Time with control (min)']} min trip", delta_color="off")
    m3.metric("Time saved by control", f"{saved:,.0f} vehicle-hours", "over the next 6 hours", delta_color="off")

    fig2 = mm.draw_dashboard(plan, capacity)
    st.pyplot(fig2)

    st.markdown("#### Recommended actions")
    for _, row in plan.iterrows():
        st.write(f"**{row['Time']:%H:%M}** {mm.LEVEL_ICONS[row['Level']]} **{row['Level']}** "
                 f"({row['Divert to alternate (%)']}% diverted) - {row['Action']}")

    with st.expander("Show full table"):
        st.dataframe(plan.drop(columns="Action"), width="stretch", hide_index=True)

# ======================= TAB 3: MODEL COMPARISON =======================
with tab_compare:
    st.subheader("How the 5 models performed on unseen test data")
    if os.path.exists("model_comparison_results.csv"):
        st.dataframe(pd.read_csv("model_comparison_results.csv", index_col=0)
                     .sort_values("Test MAE"), width="stretch")
    for img, title in [("fig1_model_comparison.png", "Error comparison"),
                       ("fig3_error_by_horizon.png", "Error vs forecast distance"),
                       ("fig5_accuracy_vs_cost.png", "Accuracy vs training cost")]:
        if os.path.exists(img):
            st.image(img, caption=title)