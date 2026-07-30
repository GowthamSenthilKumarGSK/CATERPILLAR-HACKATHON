import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, timedelta

st.set_page_config(page_title="Demand Forecasting", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>Equipment Demand Forecasting</h1>",
    unsafe_allow_html=True,
)

DB_PATH = "equipment_rental.db"


@st.cache_data(ttl=60)
def load_rental_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT r.equipment_id, e.type, r.site_id, r.operator_id,
               r.check_in_date, r.actual_return_date, r.expected_return_date,
               r.rental_days, r.is_returned,
               r.engine_hours_per_day, r.idle_hours_per_day,
               e.daily_rental_rate
        FROM rentals r
        JOIN equipment e ON r.equipment_id = e.equipment_id
        WHERE r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
        ORDER BY r.check_in_date
    """, conn)
    conn.close()

    df["check_in_date"] = pd.to_datetime(df["check_in_date"])
    df["actual_return_date"] = pd.to_datetime(df["actual_return_date"])
    df["expected_return_date"] = pd.to_datetime(df["expected_return_date"])
    return df


df = load_rental_data()

if df.empty:
    st.warning("No rental data available for forecasting.")
    st.stop()

# --- Sidebar filters ---
st.sidebar.title("Forecast Settings")
eq_types = ["All"] + sorted(df["type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Equipment Type", eq_types)

sites = ["All"] + sorted(df["site_id"].dropna().unique().tolist())
selected_site = st.sidebar.selectbox("Site", sites)

forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

filtered = df.copy()
if selected_type != "All":
    filtered = filtered[filtered["type"] == selected_type]
if selected_site != "All":
    filtered = filtered[filtered["site_id"] == selected_site]

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# =====================================================================
# SECTION 1: HISTORICAL DEMAND TRENDS
# =====================================================================
st.subheader("Historical Rental Demand")

daily_demand = (
    filtered.set_index("check_in_date")
    .resample("W")["equipment_id"]
    .count()
    .rename("rentals")
    .reset_index()
)
daily_demand.columns = ["Week", "Rentals"]

st.line_chart(daily_demand, x="Week", y="Rentals", use_container_width=True)

# =====================================================================
# SECTION 2: DEMAND BY EQUIPMENT TYPE
# =====================================================================
st.subheader("Demand by Equipment Type")

type_demand = filtered.groupby("type").agg(
    total_rentals=("equipment_id", "count"),
    avg_rental_days=("rental_days", "mean"),
    total_revenue=("daily_rental_rate", lambda x: (x * filtered.loc[x.index, "rental_days"]).sum()),
).reset_index()
type_demand.columns = ["Type", "Total Rentals", "Avg Rental Days", "Est. Revenue ($)"]
type_demand["Avg Rental Days"] = type_demand["Avg Rental Days"].round(1)
type_demand["Est. Revenue ($)"] = type_demand["Est. Revenue ($)"].round(0).astype(int)

st.dataframe(type_demand, use_container_width=True, hide_index=True)

# =====================================================================
# SECTION 3: DEMAND BY SITE
# =====================================================================
st.subheader("Demand by Site")

site_demand = filtered.groupby("site_id").agg(
    total_rentals=("equipment_id", "count"),
    avg_rental_days=("rental_days", "mean"),
    unique_equipment=("equipment_id", "nunique"),
).reset_index()
site_demand.columns = ["Site", "Total Rentals", "Avg Rental Days", "Unique Equipment Used"]
site_demand["Avg Rental Days"] = site_demand["Avg Rental Days"].round(1)
site_demand = site_demand.sort_values("Total Rentals", ascending=False)

st.dataframe(site_demand, use_container_width=True, hide_index=True)

# =====================================================================
# SECTION 4: FORECAST — MOVING AVERAGE PROJECTION
# =====================================================================
st.markdown("---")
st.subheader("Demand Forecast")

completed = filtered[filtered["is_returned"] == 1].copy()

if completed.empty:
    st.info("Not enough completed rentals to generate a forecast.")
    st.stop()

monthly_demand = (
    completed.set_index("check_in_date")
    .resample("MS")["equipment_id"]
    .count()
    .rename("rentals")
)

if len(monthly_demand) < 2:
    st.info("Need at least 2 months of data for forecasting.")
    st.stop()

window = min(3, len(monthly_demand))
moving_avg = monthly_demand.rolling(window=window).mean().dropna()

last_date = monthly_demand.index.max()
avg_monthly_rate = moving_avg.iloc[-1]
avg_daily_rate = avg_monthly_rate / 30

forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days, freq="D")
forecast_weekly = (
    pd.Series(avg_daily_rate, index=forecast_dates)
    .resample("W")
    .sum()
    .reset_index()
)
forecast_weekly.columns = ["Week", "Predicted Rentals"]
forecast_weekly["Predicted Rentals"] = forecast_weekly["Predicted Rentals"].round(1)

col1, col2, col3 = st.columns(3)
col1.metric("Avg Monthly Demand", f"{avg_monthly_rate:.1f} rentals")
col2.metric(
    f"Predicted Next {forecast_days} Days",
    f"{avg_daily_rate * forecast_days:.0f} rentals",
)

avg_days = completed["rental_days"].mean()
avg_rate = completed["daily_rental_rate"].mean()
projected_revenue = avg_daily_rate * forecast_days * avg_days * avg_rate
col3.metric("Projected Revenue", f"${projected_revenue:,.0f}")

st.markdown("#### Forecast — Weekly Predicted Rentals")
st.bar_chart(forecast_weekly, x="Week", y="Predicted Rentals", use_container_width=True)

# =====================================================================
# SECTION 5: UTILIZATION & AVAILABILITY OUTLOOK
# =====================================================================
st.markdown("---")
st.subheader("Fleet Utilization & Availability Outlook")

conn = sqlite3.connect(DB_PATH)
equipment_df = pd.read_sql_query("SELECT equipment_id, type, status FROM equipment", conn)
conn.close()

if selected_type != "All":
    equipment_df = equipment_df[equipment_df["type"] == selected_type]

total_fleet = len(equipment_df)
available_now = len(equipment_df[equipment_df["status"] == "available"])
rented_now = len(equipment_df[equipment_df["status"].isin(["rented", "overdue"])])

active_rentals = filtered[(filtered["is_returned"] == 0) & (filtered["expected_return_date"].notna())].copy()
returning_7d = 0
returning_30d = 0
today = pd.Timestamp(date.today())
if not active_rentals.empty:
    returning_7d = len(active_rentals[active_rentals["expected_return_date"] <= today + timedelta(days=7)])
    returning_30d = len(active_rentals[active_rentals["expected_return_date"] <= today + timedelta(days=30)])

c1, c2, c3, c4 = st.columns(4)
for col, label, value, color in [
    (c1, "Total Fleet", total_fleet, "#3b82f6"),
    (c2, "Available Now", available_now, "#22c55e"),
    (c3, "Returning in 7 Days", returning_7d, "#f59e0b"),
    (c4, "Returning in 30 Days", returning_30d, "#a855f7"),
]:
    col.markdown(
        f"""
        <div style="
            background:{color}20; border-left:4px solid {color};
            padding:12px 16px; border-radius:8px; text-align:center;">
            <div style="font-size:28px; font-weight:700; color:{color};">{value}</div>
            <div style="font-size:13px; color:{color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

utilization_pct = (rented_now / total_fleet * 100) if total_fleet > 0 else 0
predicted_demand = avg_daily_rate * forecast_days
supply_gap = predicted_demand - (available_now + returning_30d)

st.markdown("---")
st.subheader("Supply vs. Demand Summary")

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Current Utilization", f"{utilization_pct:.0f}%")
sc2.metric(f"Predicted Demand ({forecast_days}d)", f"{predicted_demand:.0f} rentals")

if supply_gap > 0:
    sc3.metric("Supply Gap", f"{supply_gap:.0f} units short", delta=f"-{supply_gap:.0f}", delta_color="inverse")
    st.warning(
        f"Projected demand exceeds available supply by **{supply_gap:.0f}** units over the next "
        f"**{forecast_days} days**. Consider procuring additional equipment or adjusting rental schedules."
    )
else:
    sc3.metric("Supply Surplus", f"{abs(supply_gap):.0f} units", delta=f"+{abs(supply_gap):.0f}")
    st.success(
        f"Fleet capacity is sufficient to meet projected demand over the next **{forecast_days} days**."
    )

# =====================================================================
# SECTION 6: SEASONAL PATTERNS
# =====================================================================
st.markdown("---")
st.subheader("Seasonal Demand Patterns")

completed_with_month = completed.copy()
completed_with_month["month"] = completed_with_month["check_in_date"].dt.month_name()
completed_with_month["month_num"] = completed_with_month["check_in_date"].dt.month

monthly_pattern = (
    completed_with_month.groupby(["month_num", "month"])["equipment_id"]
    .count()
    .reset_index()
)
monthly_pattern.columns = ["month_num", "Month", "Rentals"]
monthly_pattern = monthly_pattern.sort_values("month_num")

st.bar_chart(monthly_pattern, x="Month", y="Rentals", use_container_width=True)

peak_month = monthly_pattern.loc[monthly_pattern["Rentals"].idxmax(), "Month"]
st.info(f"Peak demand month: **{peak_month}**")
