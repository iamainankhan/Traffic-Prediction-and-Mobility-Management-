"""
=====================================================================
 MOBILITY MANAGEMENT  -  turning traffic FORECASTS into DECISIONS
=====================================================================
Idea in one sentence:
    "If we know how much traffic is coming in the next 6 hours, we can act BEFORE
     congestion happens instead of reacting after it has happened."

The forecasting model gives us the expected traffic volume. This module then:

  STEP 1  Measures how full the road will be           (volume / capacity)
  STEP 2  Classifies congestion                         (Free flow ... Congested)
  STEP 3  Estimates travel time on the main road        (BPR formula)
  STEP 4  Decides how much traffic to divert to an alternate route
  STEP 5  Recommends control actions + the best departure time
  STEP 6  Reports the benefit (minutes saved)

It is rule-based on purpose: every decision can be explained in one sentence.

Run:  python mobility_management.py                       (busy weekday afternoon)
      python mobility_management.py --at "2018-06-15 15:00"
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# ASSUMPTIONS (change these to fit your own scenario / explain them to your teacher)
# ---------------------------------------------------------------------
MAIN_FREE_FLOW_MIN = 20     # minutes to travel the I-94 section when the road is empty
ALT_FREE_FLOW_MIN  = 22     # alternate (parallel) route is 2 minutes longer when empty
ALT_CAPACITY       = 3000   # vehicles/hour the alternate route can handle (smaller road)
MAX_DIVERSION      = 0.50   # never divert more than 50% of the traffic

LEVEL_COLORS = {"Free flow": "tab:green", "Moderate": "gold", "Heavy": "darkorange", "Congested": "tab:red"}
LEVEL_ICONS  = {"Free flow": "🟢", "Moderate": "🟡", "Heavy": "🟠", "Congested": "🔴"}

# What the traffic authority should DO for each level
ACTIONS = {
    "Free flow": "No action needed. Normal signal timing.",
    "Moderate":  "Monitor the road. Keep normal signal timing.",
    "Heavy":     "Extend green time on main approaches, switch on ramp metering, "
                 "show 'heavy traffic ahead' on message signs.",
    "Congested": "Divert traffic to the alternate route, show route advice on message signs, "
                 "send app alerts, add buses / promote park-and-ride.",
}


# ---------------------------------------------------------------------
# STEP 1 & 2: HOW FULL IS THE ROAD?
# ---------------------------------------------------------------------
def estimate_capacity(hourly_data):
    """Estimate road capacity using the 99th percentile of valid traffic volumes."""

    traffic = pd.to_numeric(
        hourly_data["traffic_volume"],
        errors="coerce"
    ).dropna()

    if len(traffic) == 0:
        raise ValueError(
            "No valid traffic_volume values were found. "
            "Check the dataset."
        )

    capacity = float(np.percentile(traffic, 99))

    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError(
            f"Invalid road capacity calculated: {capacity}"
        )

    return capacity


def congestion_level(volume, capacity):
    """V/C ratio = volume / capacity.  Close to 1 means the road is full."""
    ratio = volume / capacity
    if ratio < 0.15: return "Free flow"
    if ratio < 0.55: return "Moderate"
    if ratio < 0.90: return "Heavy"
    return "Congested"


# ---------------------------------------------------------------------
# STEP 3: HOW LONG DOES THE TRIP TAKE?   (BPR formula, US Bureau of Public Roads)
#   time = free_flow_time * (1 + 0.15 * (volume / capacity) ^ 4)
#   The more cars, the slower everyone drives - and it gets worse very quickly.
# ---------------------------------------------------------------------
def travel_time(volume, capacity, free_flow_minutes):
    return free_flow_minutes * (1 + 0.15 * (volume / capacity) ** 4)


# ---------------------------------------------------------------------
# STEP 4: HOW MUCH TRAFFIC SHOULD TAKE THE ALTERNATE ROUTE?
#   Try every diversion share from 0% to 50% and keep the one with the
#   SHORTEST AVERAGE TRAVEL TIME for all drivers.
# ---------------------------------------------------------------------
def best_diversion(volume, main_capacity):
    best = None
    for share in np.arange(0, MAX_DIVERSION + 0.001, 0.01):
        t_main = travel_time(volume * (1 - share), main_capacity, MAIN_FREE_FLOW_MIN)
        t_alt  = travel_time(volume * share,       ALT_CAPACITY,  ALT_FREE_FLOW_MIN)
        average = (1 - share) * t_main + share * t_alt
        if best is None or average < best["average"] - 1e-9:      # strictly better only
            best = {"share": share, "t_main": t_main, "t_alt": t_alt, "average": average}
    return best


# ---------------------------------------------------------------------
# STEP 5 & 6: PUT EVERYTHING TOGETHER FOR THE NEXT 6 HOURS
# ---------------------------------------------------------------------
def create_mobility_plan(predicted_volumes, future_hours, capacity):
    """Returns one row per future hour with the level, travel times, diversion and action."""
    rows = []
    for time, volume in zip(future_hours, predicted_volumes):
        level = congestion_level(volume, capacity)
        time_without_control = travel_time(volume, capacity, MAIN_FREE_FLOW_MIN)   # everybody on main road
        plan = best_diversion(volume, capacity)

        rows.append({"Time": time, "Predicted volume": round(volume),
                     "V/C ratio": round(volume / capacity, 2), "Level": level,
                     "Time without control (min)": round(time_without_control, 1),
                     "Divert to alternate (%)": round(plan["share"] * 100),
                     "Time with control (min)": round(plan["average"], 1),
                     "Minutes saved per driver": round(time_without_control - plan["average"], 1),
                     "Action": ACTIONS[level]})
    return pd.DataFrame(rows)


def best_departure_hour(plan):
    """The hour in which a trip takes the least time."""
    return plan.loc[plan["Time with control (min)"].idxmin()]


# ---------------------------------------------------------------------
# REPORT + DASHBOARD
# ---------------------------------------------------------------------
def print_report(plan, capacity, now):
    print(f"\nMOBILITY MANAGEMENT REPORT  (forecast made at {now})")
    print(f"Road capacity used: {capacity:,.0f} vehicles/hour\n")
    table = plan.drop(columns="Action").copy()
    table["Time"] = table["Time"].dt.strftime("%H:%M")
    print(table.to_string(index=False))

    print("\nRECOMMENDED ACTIONS")
    for _, row in plan.iterrows():
        print(f"  {row['Time']:%H:%M}  {LEVEL_ICONS[row['Level']]} {row['Level']:<10} -> {row['Action']}")

    best = best_departure_hour(plan)
    worst = plan.loc[plan["Predicted volume"].idxmax()]
    saved = (plan["Minutes saved per driver"] * plan["Predicted volume"]).sum() / 60
    print(f"\nPeak traffic   : {worst['Time']:%H:%M}  ({worst['Predicted volume']:,} veh/h, {worst['Level']})")
    print(f"Best departure : {best['Time']:%H:%M}  (average trip {best['Time with control (min)']} min)")
    print(f"Total benefit  : about {saved:,.0f} vehicle-hours saved over these 6 hours")


def draw_dashboard(plan, capacity, path="mobility_dashboard.png"):
    hours = plan["Time"].dt.strftime("%H:%M")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) predicted volume vs capacity, coloured by congestion level
    axes[0].bar(hours, plan["Predicted volume"], color=[LEVEL_COLORS[l] for l in plan["Level"]], edgecolor="black")
    axes[0].axhline(capacity, color="black", linestyle="--", label=f"Capacity ({capacity:,.0f})")
    axes[0].set_title("Predicted traffic & congestion level"); axes[0].set_ylabel("vehicles / hour"); axes[0].legend()

    # 2) travel time before / after control
    x = np.arange(len(plan)); w = 0.38
    axes[1].bar(x - w/2, plan["Time without control (min)"], w, label="Without control", color="tab:red")
    axes[1].bar(x + w/2, plan["Time with control (min)"], w, label="With control", color="tab:green")
    axes[1].set_xticks(x); axes[1].set_xticklabels(hours)
    axes[1].set_ylim(plan["Time with control (min)"].min() * 0.9, plan["Time without control (min)"].max() * 1.05)
    axes[1].set_title("Average trip time (minutes)"); axes[1].legend()

    # 3) how much traffic is diverted
    axes[2].bar(hours, plan["Divert to alternate (%)"], color="tab:blue", edgecolor="black")
    axes[2].set_title("Traffic diverted to alternate route"); axes[2].set_ylabel("% of vehicles")

    plt.tight_layout(); plt.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------------
# RUN FROM THE COMMAND LINE: forecast -> mobility plan
# ---------------------------------------------------------------------
def main():
    import forecast as fc                      # the forecasting part we built earlier

    parser = argparse.ArgumentParser()
    parser.add_argument("--at", default=None, help='Forecast starts after this hour, e.g. "2018-06-15 15:00"')
    args = parser.parse_args()

    model, columns, mean, std = fc.load_model_and_scaler()
    hourly = fc.load_hourly_data()

    if args.at:
        now = pd.Timestamp(args.at)
    else:                                           # default: latest weekday 15:00 -> forecast covers the rush hour
        allowed = hourly.index[:-fc.HORIZON]

        valid_times = allowed[
            (allowed.hour == 15) &
            (allowed.dayofweek < 5)
            ]

        if len(valid_times) == 0:
            raise SystemExit(
                "No valid weekday 15:00 timestamp was found in the dataset."
            )

        now = valid_times[-1]

        print("Forecast starting time:", now)

    window = fc.get_input_window(hourly, now)
    if window is None:
        raise SystemExit("Not enough complete data before that time. Try another time.")

    prediction = fc.predict_next_hours(model, window, columns, mean, std)           # 1) FORECAST
    future_hours = pd.date_range(now + pd.Timedelta(hours=1), periods=fc.HORIZON, freq="h")
    capacity = estimate_capacity(hourly)
    plan = create_mobility_plan(prediction, future_hours, capacity)                 # 2) DECIDE

    print_report(plan, capacity, now)
    draw_dashboard(plan, capacity)
    print("\nSaved: mobility_dashboard.png")


if __name__ == "__main__":
    main()