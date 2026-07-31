import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, datetime
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from db_helpers import get_connection, refresh_equipment_status
from forecast_LightGBM import (
    load_joined_rental_data, build_demand_panel,
    load_or_train_forecast_models,
)
from report_page import render_report_page

st.set_page_config(page_title="Equipment Rental Dashboard", layout="wide")

TODAY = date.today()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Check In / Out", "Usage Logging", "Alerts & Reminders", "Demand Forecasting", "Anomaly Detection", "Smart Scheduling", "Predictive Maintenance", "Ask Fleet AI","Report Export"], label_visibility="collapsed")


# === Shared helpers ===

def status_label(status):
    return {
        "available": "Available",
        "rented": "Active Rental",
        "overdue": "Overdue",
        "unknown": "Data Error",
        "flagged": "Suspicious",
    }.get(status, status)


def status_color(status):
    return {
        "available": "#22c55e",
        "rented": "#3b82f6",
        "overdue": "#ef4444",
        "unknown": "#f59e0b",
        "flagged": "#a855f7",
    }.get(status, "#6b7280")


def status_icon(status):
    return {
        "available": "✅",
        "rented": "🔵",
        "overdue": "🔴",
        "unknown": "⚠️",
        "flagged": "🟣",
    }.get(status, "⭕")


# =====================================================================
# PAGE 1: DASHBOARD
# =====================================================================
if page == "Dashboard":
    conn = sqlite3.connect("equipment_rental.db")
    df = pd.read_sql_query("""
        SELECT e.equipment_id, e.type, e.status, e.daily_rental_rate,
               r.operator_id, r.site_id, r.expected_return_date
        FROM equipment e
        LEFT JOIN rentals r ON e.equipment_id = r.equipment_id
            AND r.id = (SELECT MAX(r2.id) FROM rentals r2 WHERE r2.equipment_id = e.equipment_id)
    """, conn)
    conn.close()

    st.markdown(
        "<h1 style='text-align:center;'>Smart Rental Tracking System</h1>",
        unsafe_allow_html=True,
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        types = ["All"] + sorted(df["type"].dropna().unique().tolist())
        selected_type = st.selectbox("Filter by Equipment Type", types)
    with col_f2:
        sites = ["All"] + sorted(df["site_id"].dropna().unique().tolist())
        selected_site = st.selectbox("Filter by Site ID", sites)

    filtered = df.copy()
    if selected_type != "All":
        filtered = filtered[filtered["type"] == selected_type]
    if selected_site != "All":
        filtered = filtered[filtered["site_id"] == selected_site]

    # Status summary cards
    status_keys = ["available", "rented", "overdue", "unknown", "flagged"]
    cols = st.columns(len(status_keys))
    for col, s in zip(cols, status_keys):
        count = len(filtered[filtered["status"] == s])
        color = status_color(s)
        col.markdown(
            f"""
            <div style="
                background:{color}20; border-left:4px solid {color};
                padding:12px 16px; border-radius:8px; text-align:center;">
                <div style="font-size:28px; font-weight:700; color:{color};">{count}</div>
                <div style="font-size:13px; color:{color};">{status_icon(s)} {status_label(s)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Column headers
    h1, h2, h3, h4, h5 = st.columns([1.1, 0.9, 0.9, 0.8, 1.2])
    h1.markdown("**Equipment ID**")
    h2.markdown("**Type**")
    h3.markdown("**Operator**")
    h4.markdown("**Site**")
    h5.markdown("**Status**")
    st.markdown("<hr style='margin:2px 0; border:none; border-top:2px solid #9ca3af;'>", unsafe_allow_html=True)

    # Equipment rows
    for _, row in filtered.iterrows():
        s = row["status"]
        color = status_color(s)
        icon = status_icon(s)
        label = status_label(s)

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([1.1, 0.9, 0.9, 0.8, 1.2])
            c1.markdown(f"**{row['equipment_id']}**")
            c2.markdown(f"{row['type']}")
            c3.markdown(f"{row['operator_id'] if pd.notna(row['operator_id']) else '—'}")
            c4.markdown(f"{row['site_id'] if pd.notna(row['site_id']) else '—'}")
            c5.markdown(
                f"<span style='background:{color}20; color:{color}; "
                f"padding:4px 10px; border-radius:12px; font-size:13px; font-weight:600;'>"
                f"{icon} {label}</span>",
                unsafe_allow_html=True,
            )

            with st.expander(f"Next Actions — {row['equipment_id']}"):
                if s == "available":
                    st.markdown(
                        f"<div style='background:#22c55e15; border-left:4px solid #22c55e; padding:12px 16px; border-radius:6px;'>"
                        f"<b style='color:#22c55e;'>Next Actions</b><br>"
                        f"<span style='font-size:14px;'>"
                        f"1. Equipment returned on time. Available for new assignment.<br>"
                        f"2. Schedule maintenance inspection before next rental.<br>"
                        f"3. Update inventory — mark as available in the fleet pool.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif s == "rented":
                    exp = pd.to_datetime(row['expected_return_date']).date() if pd.notna(row['expected_return_date']) else None
                    days_left = (exp - TODAY).days if exp else "?"
                    op = row['operator_id'] if pd.notna(row['operator_id']) else '—'
                    st.markdown(
                        f"<div style='background:#3b82f615; border-left:4px solid #3b82f6; padding:12px 16px; border-radius:6px;'>"
                        f"<b style='color:#3b82f6;'>Next Actions</b><br>"
                        f"<span style='font-size:14px;'>"
                        f"1. Rental is active. <b>{days_left} days remaining</b> until expected return.<br>"
                        f"2. Send return reminder to operator <b>{op}</b> if within 3 days of due date.<br>"
                        f"3. Monitor engine & idle hours for usage compliance.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif s == "overdue":
                    exp = pd.to_datetime(row['expected_return_date']).date() if pd.notna(row['expected_return_date']) else None
                    overdue_days = (TODAY - exp).days if exp else "?"
                    op = row['operator_id'] if pd.notna(row['operator_id']) else '—'
                    st.markdown(
                        f"<div style='background:#ef444415; border-left:4px solid #ef4444; padding:12px 16px; border-radius:6px;'>"
                        f"<b style='color:#ef4444;'>Next Actions — URGENT</b><br>"
                        f"<span style='font-size:14px;'>"
                        f"1. Equipment is <b>{overdue_days} days past due</b>. Escalate to site supervisor immediately.<br>"
                        f"2. Contact operator <b>{op}</b> for return status.<br>"
                        f"3. Apply late rental penalty charges as per contract terms.<br>"
                        f"4. If no response within 48 hours, initiate equipment recovery process.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif s == "unknown":
                    st.markdown(
                        f"<div style='background:#f59e0b15; border-left:4px solid #f59e0b; padding:12px 16px; border-radius:6px;'>"
                        f"<b style='color:#f59e0b;'>Next Actions — DATA CLEANUP</b><br>"
                        f"<span style='font-size:14px;'>"
                        f"1. Check-in and check-out dates are both missing — record is incomplete.<br>"
                        f"2. Cross-verify with site logs and operator records to recover the dates.<br>"
                        f"3. Contact data entry team to correct this record.<br>"
                        f"4. Flag for audit — do not assign this equipment until record is resolved.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif s == "flagged":
                    st.markdown(
                        f"<div style='background:#a855f715; border-left:4px solid #a855f7; padding:12px 16px; border-radius:6px;'>"
                        f"<b style='color:#a855f7;'>Next Actions — INVESTIGATION REQUIRED</b><br>"
                        f"<span style='font-size:14px;'>"
                        f"1. No operator or site assigned — equipment location is unknown.<br>"
                        f"2. Verify physical location of the equipment through GPS or last known site.<br>"
                        f"3. Check if equipment was transferred without updating records.<br>"
                        f"4. Escalate to fleet manager — possible unauthorized use or misplacement.</span></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown(
            "<hr style='margin:2px 0; border:none; border-top:1px solid #e5e7eb;'>",
            unsafe_allow_html=True,
        )


# =====================================================================
# PAGE 2: CHECK IN / CHECK OUT
# =====================================================================
elif page == "Check In / Out":
    st.markdown(
        "<h1 style='text-align:center;'>Equipment Check-In / Check-Out</h1>",
        unsafe_allow_html=True,
    )

    action = st.radio("Select Action", ["Check In", "Check Out"], horizontal=True)

    conn = get_connection()

    # --- CHECK IN ---
    if action == "Check In":
        st.markdown("---")
        st.subheader("Check In Equipment")

        available_eq = pd.read_sql_query(
            "SELECT equipment_id, type FROM equipment WHERE status = 'available' ORDER BY equipment_id", conn
        )

        busy_operators = pd.read_sql_query(
            "SELECT DISTINCT operator_id FROM rentals WHERE is_returned = 0 AND operator_id IS NOT NULL", conn
        )
        busy_op_set = set(busy_operators["operator_id"].tolist())

        all_operators = pd.read_sql_query(
            "SELECT DISTINCT operator_id FROM rentals WHERE operator_id IS NOT NULL ORDER BY operator_id", conn
        )
        free_operators = [op for op in all_operators["operator_id"].tolist() if op not in busy_op_set]

        if available_eq.empty:
            st.warning("No equipment is currently available for check-in.")
        elif not free_operators:
            st.warning("No operators are currently free for assignment.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                selected_eq = st.selectbox(
                    "Equipment ID",
                    available_eq["equipment_id"].tolist(),
                    format_func=lambda x: f"{x} ({available_eq[available_eq['equipment_id'] == x]['type'].values[0]})",
                )
                operator_id = st.selectbox("Operator ID", free_operators)
                site_id = st.text_input("Site ID", placeholder="e.g. S001")

            with col2:
                check_in_date = st.date_input("Check-In Date", value=date.today())
                expected_return = st.date_input("Expected Return Date")
                rental_days_est = (expected_return - check_in_date).days
                st.metric("Estimated Rental Days", rental_days_est)

            if st.button("Submit Check-In", type="primary", use_container_width=True):
                if not site_id.strip():
                    st.error("Site ID is required.")
                elif expected_return <= check_in_date:
                    st.error("Expected Return Date must be after Check-In Date.")
                else:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO rentals (
                            equipment_id, operator_id, site_id, check_in_date,
                            expected_return_date, actual_return_date, rental_days,
                            is_returned, condition_notes
                        ) VALUES (?, ?, ?, ?, ?, NULL, 0, 0, NULL)
                    """, (
                        selected_eq,
                        operator_id,
                        site_id.strip(),
                        check_in_date.strftime("%Y-%m-%d"),
                        expected_return.strftime("%Y-%m-%d"),
                    ))
                    conn.commit()

                    new_status = refresh_equipment_status(conn, selected_eq)
                    st.success(
                        f"Check-in recorded for **{selected_eq}** at site **{site_id}** "
                        f"with operator **{operator_id}**. Status updated to **{new_status}**."
                    )

    # --- CHECK OUT ---
    else:
        st.markdown("---")
        st.subheader("Check Out Equipment")

        open_rentals = pd.read_sql_query("""
            SELECT r.id, r.equipment_id, e.type, r.operator_id, r.site_id,
                   r.check_in_date, r.expected_return_date
            FROM rentals r
            JOIN equipment e ON r.equipment_id = e.equipment_id
            WHERE r.is_returned = 0 AND r.actual_return_date IS NULL
                  AND r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
            ORDER BY r.id DESC
        """, conn)

        if open_rentals.empty:
            st.info("No open rentals found to check out.")
        else:
            open_rentals["label"] = (
                open_rentals["equipment_id"] + " | " +
                open_rentals["type"] + " | " +
                open_rentals["operator_id"].fillna("—") + " @ " +
                open_rentals["site_id"].fillna("—")
            )

            selected_label = st.selectbox("Select Open Rental", open_rentals["label"].tolist())
            selected_row = open_rentals[open_rentals["label"] == selected_label].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Equipment:** {selected_row['equipment_id']} ({selected_row['type']})")
                st.markdown(f"**Operator:** {selected_row['operator_id']}")
                st.markdown(f"**Site:** {selected_row['site_id']}")
                st.markdown(f"**Check-In Date:** {selected_row['check_in_date']}")
                st.markdown(f"**Expected Return:** {selected_row['expected_return_date']}")

            with col2:
                check_in_dt = datetime.strptime(selected_row["check_in_date"], "%Y-%m-%d").date()
                check_out_date = date.today()
                st.date_input("Check-Out Date", value=check_out_date, disabled=True)
                engine_hrs = st.number_input("Engine Hours / Day", min_value=0.0, step=0.5, format="%.1f")
                idle_hrs = st.number_input("Idle Hours / Day", min_value=0.0, step=0.5, format="%.1f")

                rental_days = (check_out_date - check_in_dt).days
                st.metric("Rental Days", rental_days)

            if st.button("Submit Check-Out", type="primary", use_container_width=True):
                if rental_days <= 0:
                    st.error("Check-Out Date must be after Check-In Date.")
                else:
                    cursor = conn.cursor()

                    cursor.execute("""
                        UPDATE rentals
                        SET actual_return_date = ?,
                            rental_days = ?,
                            is_returned = 1,
                            engine_hours_per_day = ?,
                            idle_hours_per_day = ?
                        WHERE id = ?
                    """, (
                        check_out_date.strftime("%Y-%m-%d"),
                        rental_days,
                        engine_hrs,
                        idle_hrs,
                        int(selected_row["id"]),
                    ))

                    # cursor.execute("""
                    #     UPDATE EquipmentRental
                    #     SET CheckOutDate = ?,
                    #         EngineHoursPerDay = ?,
                    #         IdleHoursPerDay = ?,
                    #         RentalDays = ?
                    #     WHERE EquipmentID = ?
                    # """, (
                    #     check_out_date.strftime("%Y-%m-%d"),
                    #     engine_hrs,
                    #     idle_hrs,
                    #     rental_days,
                    #     selected_row["equipment_id"],
                    # ))

                    conn.commit()

                    new_status = refresh_equipment_status(conn, selected_row["equipment_id"])
                    st.success(
                        f"Check-out recorded for **{selected_row['equipment_id']}**. "
                        f"Rental days: **{rental_days}**. "
                        f"Engine hrs/day: **{engine_hrs}**, Idle hrs/day: **{idle_hrs}**. "
                        f"Status updated to **{new_status}**."
                    )

    conn.close()


# =====================================================================
# PAGE 3: USAGE LOGGING
# =====================================================================
elif page == "Usage Logging":
    st.markdown(
        "<h1 style='text-align:center;'>Usage Logging & Summary</h1>",
        unsafe_allow_html=True,
    )

    conn = sqlite3.connect("equipment_rental.db")

    usage_df = pd.read_sql_query("""
        SELECT r.id, r.equipment_id, e.type, r.operator_id, r.site_id,
               r.check_in_date, r.actual_return_date, r.expected_return_date,
               r.rental_days, r.is_returned,
               r.engine_hours_per_day, r.idle_hours_per_day,
               e.daily_rental_rate
        FROM rentals r
        JOIN equipment e ON r.equipment_id = e.equipment_id
        ORDER BY r.id DESC
    """, conn)

    conn.close()

    if usage_df.empty:
        st.info("No rental records found.")
    else:
        # --- Filters ---
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            type_opts = ["All"] + sorted(usage_df["type"].dropna().unique().tolist())
            sel_type = st.selectbox("Equipment Type", type_opts)
        with col_f2:
            if sel_type != "All":
                eq_pool = usage_df[usage_df["type"] == sel_type]["equipment_id"].dropna().unique().tolist()
            else:
                eq_pool = usage_df["equipment_id"].dropna().unique().tolist()
            eq_opts = ["All"] + sorted(eq_pool)
            sel_eq = st.selectbox("Equipment", eq_opts)
        with col_f3:
            site_opts = ["All"] + sorted(usage_df["site_id"].dropna().unique().tolist())
            sel_site = st.selectbox("Site", site_opts)
        with col_f4:
            show_filter = st.selectbox("Show", ["All Rentals", "Active Only", "Returned Only"])

        filt = usage_df.copy()
        if sel_eq != "All":
            filt = filt[filt["equipment_id"] == sel_eq]
        if sel_type != "All":
            filt = filt[filt["type"] == sel_type]
        if sel_site != "All":
            filt = filt[filt["site_id"] == sel_site]
        if show_filter == "Active Only":
            filt = filt[filt["is_returned"] == 0]
        elif show_filter == "Returned Only":
            filt = filt[filt["is_returned"] == 1]

        returned = filt[filt["is_returned"] == 1]

        # --- Summary Cards ---
        total_rental_days = returned["rental_days"].sum()
        total_engine_hrs = (returned["engine_hours_per_day"] * returned["rental_days"]).sum()
        total_idle_hrs = (returned["idle_hours_per_day"] * returned["rental_days"]).sum()
        total_runtime_hrs = total_engine_hrs + total_idle_hrs
        active_count = len(filt[filt["is_returned"] == 0])

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, value, color in [
            (c1, "Total Rented Days", f"{int(total_rental_days)}", "#3b82f6"),
            (c2, "Total Runtime Hrs", f"{total_runtime_hrs:.1f}", "#22c55e"),
            (c3, "Total Engine Hrs", f"{total_engine_hrs:.1f}", "#f59e0b"),
            (c4, "Total Idle Hrs", f"{total_idle_hrs:.1f}", "#ef4444"),
            (c5, "Active Rentals", f"{active_count}", "#a855f7"),
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

        st.markdown("---")

        # --- Usage Per Site ---
        st.subheader("Usage Per Site")
        if not returned.empty:
            site_summary = returned.groupby("site_id").agg(
                rentals=("id", "count"),
                total_days=("rental_days", "sum"),
                avg_engine_hrs=("engine_hours_per_day", "mean"),
                avg_idle_hrs=("idle_hours_per_day", "mean"),
            ).reset_index()
            site_summary.columns = ["Site", "Rentals", "Total Days", "Avg Engine Hrs/Day", "Avg Idle Hrs/Day"]
            site_summary["Avg Engine Hrs/Day"] = site_summary["Avg Engine Hrs/Day"].round(1)
            site_summary["Avg Idle Hrs/Day"] = site_summary["Avg Idle Hrs/Day"].round(1)
            st.dataframe(site_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No completed rentals to summarize.")

        st.markdown("---")

        # --- Detailed Usage Log ---
        st.subheader("Detailed Usage Log")

        log_display = filt[["equipment_id", "type", "operator_id", "site_id",
                            "check_in_date", "actual_return_date", "rental_days",
                            "engine_hours_per_day", "idle_hours_per_day", "is_returned"]].copy()
        log_display["status"] = log_display["is_returned"].map({0: "🔵 Active", 1: "✅ Returned"})
        log_display["total_engine_hrs"] = (log_display["engine_hours_per_day"] * log_display["rental_days"]).round(1)
        log_display["total_idle_hrs"] = (log_display["idle_hours_per_day"] * log_display["rental_days"]).round(1)
        log_display = log_display.drop(columns=["is_returned"])
        log_display.columns = ["Equipment", "Type", "Operator", "Site",
                               "Check-In", "Return Date", "Days",
                               "Engine Hrs/Day", "Idle Hrs/Day", "Status",
                               "Total Engine Hrs", "Total Idle Hrs"]
        st.dataframe(log_display, use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- Downtime Analysis ---
        st.subheader("Downtime Analysis")
        if not returned.empty:
            returned_copy = returned.copy()
            returned_copy["idle_ratio"] = (
                returned_copy["idle_hours_per_day"] /
                (returned_copy["engine_hours_per_day"] + returned_copy["idle_hours_per_day"]).replace(0, float("nan"))
            ) * 100
            downtime = returned_copy.groupby("equipment_id").agg(
                total_days=("rental_days", "sum"),
                avg_idle_pct=("idle_ratio", "mean"),
            ).reset_index()
            downtime.columns = ["Equipment", "Total Rental Days", "Avg Idle %"]
            downtime["Avg Idle %"] = downtime["Avg Idle %"].round(1)
            downtime = downtime.sort_values("Avg Idle %", ascending=False)
            st.dataframe(downtime, use_container_width=True, hide_index=True)
        else:
            st.info("No completed rentals for downtime analysis.")

        st.markdown("---")

        # --- Fuel Consumption ---
        st.subheader("Fuel Consumption per Equipment")
        fuel_conn = sqlite3.connect("equipment_rental.db")
        fuel_df = pd.read_sql_query("""
            SELECT e.equipment_id, e.type, e.age,
                   er.EngineHoursPerDay, er.IdleHoursPerDay, er.RentalDays,
                   fc.fuel_less_than_5, fc.fuel_greater_than_or_equal_5
            FROM equipment e
            JOIN EquipmentRental er ON e.equipment_id = er.EquipmentID
            JOIN fuel_consumption fc ON fc.equipment_type = e.type
        """, fuel_conn)
        fuel_conn.close()

        if fuel_df.empty:
            st.info("No fuel consumption data available.")
        else:
            fuel_df["fuel_rate"] = fuel_df.apply(
                lambda r: r["fuel_less_than_5"] if r["age"] < 5 else r["fuel_greater_than_or_equal_5"], axis=1
            )
            fuel_df["engine_fuel"] = fuel_df["EngineHoursPerDay"] * fuel_df["fuel_rate"] * fuel_df["RentalDays"]
            fuel_df["idle_fuel"] = fuel_df["IdleHoursPerDay"] * (fuel_df["fuel_rate"] * 0.25) * fuel_df["RentalDays"]
            fuel_df["total_fuel"] = fuel_df["engine_fuel"] + fuel_df["idle_fuel"]

            display_fuel = fuel_df[["equipment_id", "type", "age", "engine_fuel", "idle_fuel", "total_fuel"]].copy()
            display_fuel.columns = ["Equipment", "Type", "Age", "Engine Fuel (L)", "Idle Fuel (L)", "Total Fuel (L)"]
            for c in ["Engine Fuel (L)", "Idle Fuel (L)", "Total Fuel (L)"]:
                display_fuel[c] = display_fuel[c].round(2)
            st.dataframe(display_fuel, use_container_width=True, hide_index=True)

            chart_data = display_fuel.set_index("Equipment")[["Engine Fuel (L)", "Idle Fuel (L)"]]
            st.bar_chart(chart_data, color=["#f59e0b", "#3b82f6"])


# =====================================================================
# PAGE 4: ALERTS & REMINDERS
# =====================================================================
elif page == "Alerts & Reminders":
    st.markdown(
        "<h1 style='text-align:center;'>Alerts & Reminders</h1>",
        unsafe_allow_html=True,
    )

    conn = sqlite3.connect("equipment_rental.db")

    open_rentals = pd.read_sql_query("""
        SELECT r.id, r.equipment_id, e.type, r.operator_id, r.site_id,
               r.check_in_date, r.expected_return_date
        FROM rentals r
        JOIN equipment e ON r.equipment_id = e.equipment_id
        WHERE r.is_returned = 0 AND r.actual_return_date IS NULL
              AND r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
              AND r.expected_return_date IS NOT NULL
        ORDER BY r.expected_return_date ASC
    """, conn)

    conn.close()

    if open_rentals.empty:
        st.info("No active rentals to monitor.")
    else:
        alerts = []
        for _, row in open_rentals.iterrows():
            exp = datetime.strptime(row["expected_return_date"], "%Y-%m-%d").date()
            days_left = (exp - TODAY).days
            eq_id = row["equipment_id"]
            eq_type = row["type"]
            op = row["operator_id"] if pd.notna(row["operator_id"]) else "—"
            site = row["site_id"] if pd.notna(row["site_id"]) else "—"

            if days_left > 5:
                level = "ok"
                icon = "✅"
                color = "#22c55e"
                tag = "On Track"
                msg = f"Return in {days_left} days. No action needed."
            elif 3 <= days_left <= 5:
                level = "gentle"
                icon = "💬"
                color = "#3b82f6"
                tag = "Gentle Reminder"
                msg = f"Returning in {days_left} days. Please plan for return of {eq_type} ({eq_id}) from site {site}."
            elif days_left == 2:
                level = "soft_warning"
                icon = "🔔"
                color = "#f59e0b"
                tag = "Soft Warning"
                msg = f"Only 2 days left! Operator {op}, please prepare to return {eq_type} ({eq_id}) from site {site}."
            elif days_left == 1:
                level = "warning"
                icon = "⚠️"
                color = "#f97316"
                tag = "Warning"
                msg = f"Return TOMORROW! Operator {op} must return {eq_type} ({eq_id}) from site {site} by end of day tomorrow."
            elif days_left == 0:
                level = "due_today"
                icon = "🚨"
                color = "#ef4444"
                tag = "Due Today"
                msg = f"TIME TO RETURN! {eq_type} ({eq_id}) must be returned from site {site} by operator {op} TODAY."
            else:
                overdue_days = abs(days_left)
                level = "overdue"
                icon = "🔴"
                color = "#dc2626"
                tag = f"Overdue by {overdue_days} day{'s' if overdue_days > 1 else ''}"
                msg = (
                    f"OVERDUE by {overdue_days} day{'s' if overdue_days > 1 else ''}! "
                    f"Return {eq_type} ({eq_id}) from site {site} immediately. "
                    f"Operator {op} — escalate to supervisor."
                )

            alerts.append({
                "equipment_id": eq_id, "type": eq_type, "operator": op,
                "site": site, "expected_return": row["expected_return_date"],
                "days_left": days_left, "level": level, "icon": icon,
                "color": color, "tag": tag, "msg": msg,
            })

        # --- Summary counts ---
        level_counts = {}
        for a in alerts:
            level_counts[a["level"]] = level_counts.get(a["level"], 0) + 1

        summary_items = [
            ("Overdue", level_counts.get("overdue", 0), "#dc2626"),
            ("Due Today", level_counts.get("due_today", 0), "#ef4444"),
            ("Warning", level_counts.get("warning", 0), "#f97316"),
            ("Soft Warning", level_counts.get("soft_warning", 0), "#f59e0b"),
            ("Reminder", level_counts.get("gentle", 0), "#3b82f6"),
            ("On Track", level_counts.get("ok", 0), "#22c55e"),
        ]

        cols = st.columns(len(summary_items))
        for col, (lbl, cnt, clr) in zip(cols, summary_items):
            col.markdown(
                f"""
                <div style="
                    background:{clr}20; border-left:4px solid {clr};
                    padding:12px 16px; border-radius:8px; text-align:center;">
                    <div style="font-size:28px; font-weight:700; color:{clr};">{cnt}</div>
                    <div style="font-size:13px; color:{clr};">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Filter by severity ---
        severity_options = ["All", "Overdue", "Due Today", "Warning", "Soft Warning", "Gentle Reminder", "On Track"]
        level_map = {
            "Overdue": "overdue", "Due Today": "due_today", "Warning": "warning",
            "Soft Warning": "soft_warning", "Gentle Reminder": "gentle", "On Track": "ok",
        }
        sel_severity = st.selectbox("Filter by Severity", severity_options)

        filtered_alerts = alerts
        if sel_severity != "All":
            filtered_alerts = [a for a in alerts if a["level"] == level_map[sel_severity]]

        # --- Alert cards ---
        # Sort: overdue first (most negative days_left), then ascending
        filtered_alerts.sort(key=lambda a: a["days_left"])

        for a in filtered_alerts:
            st.markdown(
                f"<div style='background:{a['color']}10; border-left:5px solid {a['color']}; "
                f"padding:14px 18px; border-radius:8px; margin-bottom:8px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<div>"
                f"<span style='font-size:18px;'>{a['icon']}</span> "
                f"<b style='font-size:15px;'>{a['equipment_id']}</b> "
                f"<span style='font-size:13px; color:#6b7280;'>({a['type']})</span>"
                f"</div>"
                f"<span style='background:{a['color']}25; color:{a['color']}; "
                f"padding:3px 10px; border-radius:10px; font-size:12px; font-weight:600;'>"
                f"{a['tag']}</span>"
                f"</div>"
                f"<div style='margin-top:6px; font-size:14px; color:#374151;'>{a['msg']}</div>"
                f"<div style='margin-top:4px; font-size:12px; color:#9ca3af;'>"
                f"Operator: {a['operator']} &nbsp;|&nbsp; Site: {a['site']} &nbsp;|&nbsp; "
                f"Expected Return: {a['expected_return']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# =====================================================================
# PAGE 5: DEMAND FORECASTING
# =====================================================================
# elif page == "Demand Forecasting":
#     st.markdown(
#         "<h1 style='text-align:center;'>Equipment Demand Forecasting</h1>",
#         unsafe_allow_html=True,
#     )

#     SYNTH_DB = "equipment_rental_synthetic.db"
#     MAIN_DB = "equipment_rental.db"

#     import os
#     fc_db_path = SYNTH_DB if os.path.exists(SYNTH_DB) else MAIN_DB
#     conn = sqlite3.connect(fc_db_path)

#     df_fc = pd.read_sql_query("""
#         SELECT r.equipment_id, e.type, r.site_id, r.operator_id,
#                r.check_in_date, r.actual_return_date, r.expected_return_date,
#                r.rental_days, r.is_returned,
#                r.engine_hours_per_day, r.idle_hours_per_day,
#                e.daily_rental_rate
#         FROM rentals r
#         JOIN equipment e ON r.equipment_id = e.equipment_id
#         WHERE r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
#         ORDER BY r.check_in_date
#     """, conn)

#     df_fc["check_in_date"] = pd.to_datetime(df_fc["check_in_date"])
#     df_fc["actual_return_date"] = pd.to_datetime(df_fc["actual_return_date"])
#     df_fc["expected_return_date"] = pd.to_datetime(df_fc["expected_return_date"])

#     if df_fc.empty:
#         st.warning("No rental data available for forecasting.")
#     else:
#         # Filters
#         fc1, fc2, fc3 = st.columns(3)
#         with fc1:
#             eq_types = ["All"] + sorted(df_fc["type"].dropna().unique().tolist())
#             sel_fc_type = st.selectbox("Equipment Type", eq_types, key="fc_type")
#         with fc2:
#             fc_sites = ["All"] + sorted(df_fc["site_id"].dropna().unique().tolist())
#             sel_fc_site = st.selectbox("Site", fc_sites, key="fc_site")
#         with fc3:
#             forecast_weeks = st.slider("Forecast Horizon (weeks)", 1, 12, 8)

#         fc_filtered = df_fc.copy()
#         if sel_fc_type != "All":
#             fc_filtered = fc_filtered[fc_filtered["type"] == sel_fc_type]
#         if sel_fc_site != "All":
#             fc_filtered = fc_filtered[fc_filtered["site_id"] == sel_fc_site]

#         if fc_filtered.empty:
#             st.warning("No data matches the selected filters.")
#         else:
#             # --- SECTION 1: Historical demand ---
#             st.subheader("Historical Rental Demand")
#             daily_demand = (
#                 fc_filtered.set_index("check_in_date")
#                 .resample("W")["equipment_id"]
#                 .count()
#                 .rename("rentals")
#                 .reset_index()
#             )
#             daily_demand.columns = ["Week", "Rentals"]
#             st.line_chart(daily_demand, x="Week", y="Rentals", use_container_width=True)

#             # --- SECTION 2: Demand by type ---
#             st.subheader("Demand by Equipment Type")
#             type_demand = fc_filtered.groupby("type").agg(
#                 total_rentals=("equipment_id", "count"),
#                 avg_rental_days=("rental_days", "mean"),
#                 total_revenue=("daily_rental_rate", lambda x: (x * fc_filtered.loc[x.index, "rental_days"]).sum()),
#             ).reset_index()
#             type_demand.columns = ["Type", "Total Rentals", "Avg Rental Days", "Est. Revenue ($)"]
#             type_demand["Avg Rental Days"] = type_demand["Avg Rental Days"].round(1)
#             type_demand["Est. Revenue ($)"] = type_demand["Est. Revenue ($)"].round(0).astype(int)
#             st.dataframe(type_demand, use_container_width=True, hide_index=True)

#             # --- SECTION 3: Demand by site ---
#             st.subheader("Demand by Site")
#             site_demand = fc_filtered.groupby("site_id").agg(
#                 total_rentals=("equipment_id", "count"),
#                 avg_rental_days=("rental_days", "mean"),
#                 unique_equipment=("equipment_id", "nunique"),
#             ).reset_index()
#             site_demand.columns = ["Site", "Total Rentals", "Avg Rental Days", "Unique Equipment Used"]
#             site_demand["Avg Rental Days"] = site_demand["Avg Rental Days"].round(1)
#             site_demand = site_demand.sort_values("Total Rentals", ascending=False)
#             st.dataframe(site_demand, use_container_width=True, hide_index=True)

#             # =============================================================
#             # SECTION 4: LightGBM ML FORECAST
#             # =============================================================
#             st.markdown("---")
#             st.subheader("LightGBM Demand Forecast")
#             st.caption("Global gradient-boosted model trained across all (site, type) series with walk-forward validation")

#             @st.cache_data(ttl=300, show_spinner="Training LightGBM model...")
#             def run_lgbm_forecast(_db_path, horizon):
#                 rentals = load_joined_rental_data(_db_path)
#                 panel = build_demand_panel(rentals, freq="W-MON")
#                 features_df = add_features(panel)
#                 available_feats = [c for c in FEATURE_COLS if c in features_df.columns]

#                 n_periods = features_df["period"].nunique()
#                 min_train = max(8, n_periods // 3)

#                 wf_result = walk_forward_validate(
#                     features_df, available_feats,
#                     n_folds=4, min_train_periods=min_train, horizon=1,
#                 )
#                 baseline_df = evaluate_baseline_walk_forward(panel, n_folds=4, horizon=1)
#                 final = fit_final_models_and_forecast(
#                     features_df, available_feats, horizon=horizon, freq="W-MON",
#                 )
#                 return wf_result, baseline_df, final, features_df

#             try:
#                 wf_result, baseline_df, final_fc, features_df = run_lgbm_forecast(fc_db_path, forecast_weeks)

#                 # --- Model performance cards ---
#                 baseline_mae = baseline_df["mae"].mean()
#                 lgbm_mae = wf_result.mean_mae
#                 improvement = (baseline_mae - lgbm_mae) / baseline_mae * 100 if baseline_mae > 0 else 0

#                 pm1, pm2, pm3, pm4 = st.columns(4)
#                 for col, lbl, val, clr in [
#                     (pm1, "LightGBM MAE", f"{lgbm_mae:.3f}", "#3b82f6"),
#                     (pm2, "LightGBM RMSE", f"{wf_result.mean_rmse:.3f}", "#f59e0b"),
#                     (pm3, "Baseline MAE", f"{baseline_mae:.3f}", "#6b7280"),
#                     (pm4, "Improvement", f"{improvement:.1f}%", "#22c55e" if improvement > 0 else "#ef4444"),
#                 ]:
#                     col.markdown(
#                         f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px 16px; border-radius:8px; text-align:center;'>"
#                         f"<div style='font-size:28px; font-weight:700; color:{clr};'>{val}</div>"
#                         f"<div style='font-size:13px; color:{clr};'>{lbl}</div></div>",
#                         unsafe_allow_html=True,
#                     )

#                 if improvement <= 0:
#                     st.warning("LightGBM does not beat the seasonal-naive baseline — consider adding more data or features.")
#                 else:
#                     st.success(f"LightGBM outperforms seasonal-naive baseline by **{improvement:.1f}%** (walk-forward out-of-sample).")

#                 # --- Walk-forward fold details ---
#                 st.markdown("---")
#                 st.subheader("Walk-Forward Validation (Expanding Window)")
#                 st.caption("Each fold trains on all data up to the cutoff, then predicts the next week — no data leakage.")
#                 fold_display = wf_result.fold_metrics.copy()
#                 fold_display.columns = ["Fold", "Train Periods", "Test Period Start", "Test Rows", "MAE", "RMSE"]
#                 fold_display["MAE"] = fold_display["MAE"].round(4)
#                 fold_display["RMSE"] = fold_display["RMSE"].round(4)
#                 st.dataframe(fold_display, use_container_width=True, hide_index=True)

#                 # --- Forecast chart: point + upper ---
#                 st.markdown("---")
#                 st.subheader(f"Forecast — Next {forecast_weeks} Weeks (Point + Upper 80th Pct)")

#                 merged_fc = final_fc.point_forecast.merge(
#                     final_fc.upper_forecast, on=["site_id", "equipment_type", "period"]
#                 )

#                 fc_display = merged_fc.copy()
#                 if sel_fc_type != "All":
#                     fc_display = fc_display[fc_display["equipment_type"] == sel_fc_type]
#                 if sel_fc_site != "All":
#                     fc_display = fc_display[fc_display["site_id"] == sel_fc_site]

#                 if fc_display.empty:
#                     st.info("No forecast data for the selected filters.")
#                 else:
#                     weekly_agg = fc_display.groupby("period").agg(
#                         point=("point_forecast", "sum"),
#                         upper=("upper_forecast", "sum"),
#                     ).reset_index()
#                     weekly_agg.columns = ["Week", "Predicted (Point)", "Upper (80th Pct)"]
#                     weekly_agg["Predicted (Point)"] = weekly_agg["Predicted (Point)"].round(1)
#                     weekly_agg["Upper (80th Pct)"] = weekly_agg["Upper (80th Pct)"].round(1)

#                     st.line_chart(weekly_agg.set_index("Week"), use_container_width=True)

#                     st.dataframe(weekly_agg, use_container_width=True, hide_index=True)

#                 # --- Per site x type forecast table ---
#                 st.markdown("---")
#                 st.subheader("Forecast by Site & Equipment Type")
#                 st.caption("Granular (site, type, week) predictions for pre-positioning decisions")

#                 detail_fc = merged_fc.copy()
#                 if sel_fc_type != "All":
#                     detail_fc = detail_fc[detail_fc["equipment_type"] == sel_fc_type]
#                 if sel_fc_site != "All":
#                     detail_fc = detail_fc[detail_fc["site_id"] == sel_fc_site]

#                 if not detail_fc.empty:
#                     pivot_point = detail_fc.groupby(["site_id", "equipment_type"]).agg(
#                         total_point=("point_forecast", "sum"),
#                         total_upper=("upper_forecast", "sum"),
#                         avg_weekly=("point_forecast", "mean"),
#                     ).reset_index()
#                     pivot_point.columns = ["Site", "Type", "Total Predicted", "Total Upper", "Avg Weekly"]
#                     pivot_point["Total Predicted"] = pivot_point["Total Predicted"].round(1)
#                     pivot_point["Total Upper"] = pivot_point["Total Upper"].round(1)
#                     pivot_point["Avg Weekly"] = pivot_point["Avg Weekly"].round(2)
#                     pivot_point = pivot_point.sort_values("Total Predicted", ascending=False)
#                     st.dataframe(pivot_point, use_container_width=True, hide_index=True)

#                 # --- Feature importance ---
#                 st.markdown("---")
#                 st.subheader("Feature Importance (LightGBM)")
#                 imp = final_fc.feature_importance.copy()
#                 imp.columns = ["Feature", "Importance"]
#                 st.bar_chart(imp.set_index("Feature"), use_container_width=True)

#             except (ValueError, Exception) as e:
#                 st.error(f"LightGBM forecasting failed: {e}")
#                 st.info("Falling back to moving average forecast below.")

#             # =============================================================
#             # SECTION 5: MOVING AVERAGE FORECAST (fallback / comparison)
#             # =============================================================
#             st.markdown("---")
#             st.subheader("Moving Average Forecast (Baseline Comparison)")

#             completed = fc_filtered[fc_filtered["is_returned"] == 1].copy()

#             if completed.empty:
#                 st.info("Not enough completed rentals to generate a moving average forecast.")
#             else:
#                 monthly_demand = (
#                     completed.set_index("check_in_date")
#                     .resample("MS")["equipment_id"]
#                     .count()
#                     .rename("rentals")
#                 )

#                 if len(monthly_demand) < 2:
#                     st.info("Need at least 2 months of data for moving average.")
#                 else:
#                     forecast_days = forecast_weeks * 7
#                     window = min(3, len(monthly_demand))
#                     moving_avg = monthly_demand.rolling(window=window).mean().dropna()

#                     last_date = monthly_demand.index.max()
#                     avg_monthly_rate = moving_avg.iloc[-1]
#                     avg_daily_rate = avg_monthly_rate / 30

#                     forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days, freq="D")
#                     forecast_weekly = (
#                         pd.Series(avg_daily_rate, index=forecast_dates)
#                         .resample("W")
#                         .sum()
#                         .reset_index()
#                     )
#                     forecast_weekly.columns = ["Week", "Predicted Rentals"]
#                     forecast_weekly["Predicted Rentals"] = forecast_weekly["Predicted Rentals"].round(1)

#                     mc1, mc2, mc3 = st.columns(3)
#                     mc1.metric("Avg Monthly Demand", f"{avg_monthly_rate:.1f} rentals")
#                     mc2.metric(f"Predicted Next {forecast_days} Days", f"{avg_daily_rate * forecast_days:.0f} rentals")

#                     avg_days = completed["rental_days"].mean()
#                     avg_rate = completed["daily_rental_rate"].mean()
#                     projected_revenue = avg_daily_rate * forecast_days * avg_days * avg_rate
#                     mc3.metric("Projected Revenue", f"${projected_revenue:,.0f}")

#                     st.bar_chart(forecast_weekly, x="Week", y="Predicted Rentals", use_container_width=True)

#             # =============================================================
#             # SECTION 6: FLEET UTILIZATION & AVAILABILITY
#             # =============================================================
#             st.markdown("---")
#             st.subheader("Fleet Utilization & Availability Outlook")

#             equipment_df = pd.read_sql_query("SELECT equipment_id, type, status FROM equipment", conn)
#             if sel_fc_type != "All":
#                 equipment_df = equipment_df[equipment_df["type"] == sel_fc_type]

#             total_fleet = len(equipment_df)
#             available_now = len(equipment_df[equipment_df["status"] == "available"])
#             rented_now = len(equipment_df[equipment_df["status"].isin(["rented", "overdue"])])

#             active_rentals = fc_filtered[(fc_filtered["is_returned"] == 0) & (fc_filtered["expected_return_date"].notna())].copy()
#             today_ts = pd.Timestamp(date.today())
#             returning_7d = len(active_rentals[active_rentals["expected_return_date"] <= today_ts + timedelta(days=7)]) if not active_rentals.empty else 0
#             returning_30d = len(active_rentals[active_rentals["expected_return_date"] <= today_ts + timedelta(days=30)]) if not active_rentals.empty else 0

#             uc1, uc2, uc3, uc4 = st.columns(4)
#             for col, lbl, val, clr in [
#                 (uc1, "Total Fleet", total_fleet, "#3b82f6"),
#                 (uc2, "Available Now", available_now, "#22c55e"),
#                 (uc3, "Returning in 7 Days", returning_7d, "#f59e0b"),
#                 (uc4, "Returning in 30 Days", returning_30d, "#a855f7"),
#             ]:
#                 col.markdown(
#                     f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px 16px; border-radius:8px; text-align:center;'>"
#                     f"<div style='font-size:28px; font-weight:700; color:{clr};'>{val}</div>"
#                     f"<div style='font-size:13px; color:{clr};'>{lbl}</div></div>",
#                     unsafe_allow_html=True,
#                 )

#             # Supply vs demand
#             utilization_pct = (rented_now / total_fleet * 100) if total_fleet > 0 else 0

#             st.markdown("---")
#             st.subheader("Supply vs. Demand Summary")

#             sd1, sd2 = st.columns(2)
#             sd1.metric("Current Utilization", f"{utilization_pct:.0f}%")
#             sd2.metric("Fleet Size", f"{total_fleet} units")

#             # =============================================================
#             # SECTION 7: SEASONAL PATTERNS
#             # =============================================================
#             st.markdown("---")
#             st.subheader("Seasonal Demand Patterns")

#             completed_all = fc_filtered[fc_filtered["is_returned"] == 1].copy()
#             if not completed_all.empty:
#                 completed_all["month"] = completed_all["check_in_date"].dt.month_name()
#                 completed_all["month_num"] = completed_all["check_in_date"].dt.month

#                 monthly_pattern = (
#                     completed_all.groupby(["month_num", "month"])["equipment_id"]
#                     .count()
#                     .reset_index()
#                 )
#                 monthly_pattern.columns = ["month_num", "Month", "Rentals"]
#                 monthly_pattern = monthly_pattern.sort_values("month_num")

#                 st.bar_chart(monthly_pattern, x="Month", y="Rentals", use_container_width=True)

#                 peak_month = monthly_pattern.loc[monthly_pattern["Rentals"].idxmax(), "Month"]
#                 st.info(f"Peak demand month: **{peak_month}**")

#     conn.close()

elif page == "Demand Forecasting":
    st.markdown(
        "<h1 style='text-align:center;'>Equipment Demand Forecasting</h1>",
        unsafe_allow_html=True,
    )

    SYNTH_DB = "equipment_rental_large.db"
    MAIN_DB = "equipment_rental.db"

    import os
    fc_db_path = SYNTH_DB if os.path.exists(SYNTH_DB) else MAIN_DB
    conn = sqlite3.connect(fc_db_path)

    df_fc = pd.read_sql_query("""
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

    df_fc["check_in_date"] = pd.to_datetime(df_fc["check_in_date"])
    df_fc["actual_return_date"] = pd.to_datetime(df_fc["actual_return_date"])
    df_fc["expected_return_date"] = pd.to_datetime(df_fc["expected_return_date"])

    if df_fc.empty:
        st.warning("No rental data available for forecasting.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            eq_types = ["All"] + sorted(df_fc["type"].dropna().unique().tolist())
            sel_fc_type = st.selectbox("Equipment Type", eq_types, key="fc_type")
        with fc2:
            fc_sites = ["All"] + sorted(df_fc["site_id"].dropna().unique().tolist())
            sel_fc_site = st.selectbox("Site", fc_sites, key="fc_site")
        with fc3:
            forecast_weeks = st.slider("Forecast Horizon (weeks)", 1, 12, 8)

        fc_filtered = df_fc.copy()
        if sel_fc_type != "All":
            fc_filtered = fc_filtered[fc_filtered["type"] == sel_fc_type]
        if sel_fc_site != "All":
            fc_filtered = fc_filtered[fc_filtered["site_id"] == sel_fc_site]

        if fc_filtered.empty:
            st.warning("No data matches the selected filters.")
        else:
            # --- SECTION 1: Historical working-hours demand ---
            st.subheader("Historical Working-Hours Demand")
            rentals = load_joined_rental_data(fc_db_path)
            history_panel = build_demand_panel(rentals, freq="W-MON")
            history_panel["week"] = pd.to_datetime(history_panel["week"])

            if sel_fc_type != "All":
                history_panel = history_panel[history_panel["equipment_type"] == sel_fc_type]
            if sel_fc_site != "All":
                history_panel = history_panel[history_panel["site_id"] == sel_fc_site]

            historical_weekly = (
                history_panel.groupby("week", as_index=False)["working_hours"].sum()
                .rename(columns={"working_hours": "Working Hours"})
            )
            historical_weekly.columns = ["Week", "Working Hours"]
            st.line_chart(historical_weekly.set_index("Week"), use_container_width=True)

            # --- SECTION 2: Demand by equipment type ---
            st.subheader("Demand by Equipment Type")
            type_demand = (
                history_panel.groupby("equipment_type", as_index=False)
                .agg(
                    total_working_hours=("working_hours", "sum"),
                    avg_weekly_working_hours=("working_hours", "mean"),
                )
            )
            type_demand.columns = ["Type", "Total Working Hours", "Avg Weekly Working Hours"]
            type_demand["Total Working Hours"] = type_demand["Total Working Hours"].round(1)
            type_demand["Avg Weekly Working Hours"] = type_demand["Avg Weekly Working Hours"].round(1)
            st.dataframe(type_demand, use_container_width=True, hide_index=True)

            # --- SECTION 3: Demand by site ---
            st.subheader("Demand by Site")
            site_demand = (
                history_panel.groupby("site_id", as_index=False)
                .agg(
                    total_working_hours=("working_hours", "sum"),
                    avg_weekly_working_hours=("working_hours", "mean"),
                    unique_equipment=("equipment_type", "nunique"),
                )
            )
            site_demand.columns = ["Site", "Total Working Hours", "Avg Weekly Working Hours", "Unique Equipment Types"]
            site_demand["Total Working Hours"] = site_demand["Total Working Hours"].round(1)
            site_demand["Avg Weekly Working Hours"] = site_demand["Avg Weekly Working Hours"].round(1)
            site_demand = site_demand.sort_values("Total Working Hours", ascending=False)
            st.dataframe(site_demand, use_container_width=True, hide_index=True)

            # =============================================================
            # SECTION 4: DEMAND FORECASTS
            # =============================================================
            st.markdown("---")
            st.subheader("Demand Forecasts (LGBM & ARIMA)")
            st.caption("Weekly working-hours demand forecast from the saved LightGBM and ARIMA models")

            @st.cache_data(ttl=300, show_spinner="Loading saved demand models...")
            def run_demand_forecast(_db_path, horizon):
                _, _, lgbm_forecast, arima_forecast = load_or_train_forecast_models(
                    db_path=_db_path, horizon=horizon, out_dir="saved_models"
                )
                return lgbm_forecast, arima_forecast

            try:
                lgbm_forecast, arima_forecast = run_demand_forecast(fc_db_path, forecast_weeks)

                def prepare_forecast_view(forecast_df):
                    view = forecast_df.copy()
                    view["week"] = pd.to_datetime(view["week"])
                    if sel_fc_type != "All":
                        view = view[view["equipment_type"] == sel_fc_type]
                    if sel_fc_site != "All":
                        view = view[view["site_id"] == sel_fc_site]
                    return view

                lgbm_view = prepare_forecast_view(lgbm_forecast)
                arima_view = prepare_forecast_view(arima_forecast)

                if lgbm_view.empty and arima_view.empty:
                    st.info("No forecast data for the selected filters.")
                else:
                    st.markdown("---")
                    st.subheader("Forecast by Site-Machine Pair")
                    pair_forecast = (
                        lgbm_view.rename(columns={"forecast": "forecast_hours"})
                        .copy()
                    )
                    pair_forecast["pair_label"] = (
                        pair_forecast["site_id"].astype(str) + " - " + pair_forecast["equipment_type"].astype(str)
                    )

                    site_names = sorted(pair_forecast["site_id"].astype(str).unique().tolist())
                    for site_name in site_names:
                        site_plot = pair_forecast[pair_forecast["site_id"].astype(str) == site_name]
                        if site_plot.empty:
                            continue

                        site_plot = (
                            site_plot.pivot_table(
                                index="week",
                                columns="pair_label",
                                values="forecast_hours",
                                aggfunc="sum",
                            )
                            .fillna(0)
                        )
                        if not site_plot.empty:
                            st.subheader(f"{site_name} Forecast")
                            st.line_chart(site_plot, use_container_width=True)

                    detail_fc = (
                        lgbm_view.rename(columns={"forecast": "lightgbm_forecast"})
                        .merge(
                            arima_view.rename(columns={"forecast": "arima_forecast"}),
                            on=["site_id", "equipment_type", "week"],
                            how="outer",
                        )
                        .fillna(0)
                    )
                    detail_fc["week"] = detail_fc["week"].dt.strftime("%Y-%m-%d")
                    pivot = (
                        detail_fc.groupby(["site_id", "equipment_type"], as_index=False)
                        .agg(
                            lightgbm_total=("lightgbm_forecast", "sum"),
                            arima_total=("arima_forecast", "sum"),
                        )
                    )
                    pivot.columns = ["Site", "Type", "LightGBM Forecast (hrs)", "ARIMA Forecast (hrs)"]
                    pivot["LightGBM Forecast (hrs)"] = pivot["LightGBM Forecast (hrs)"].round(1)
                    pivot["ARIMA Forecast (hrs)"] = pivot["ARIMA Forecast (hrs)"].round(1)
                    pivot = pivot.sort_values("LightGBM Forecast (hrs)", ascending=False)
                    st.subheader("Forecast by Site & Equipment Type")
                    st.dataframe(pivot, use_container_width=True, hide_index=True)

            except (ValueError, Exception) as e:
                st.error(f"Demand forecasting failed: {e}")

            # =============================================================
            # SECTION 5: FLEET UTILIZATION & AVAILABILITY
            # =============================================================
            st.markdown("---")
            st.subheader("Fleet Utilization & Availability Outlook")

            equipment_df = pd.read_sql_query("SELECT equipment_id, type, status FROM equipment", conn)
            if sel_fc_type != "All":
                equipment_df = equipment_df[equipment_df["type"] == sel_fc_type]

            total_fleet = len(equipment_df)
            available_now = len(equipment_df[equipment_df["status"] == "available"])
            rented_now = len(equipment_df[equipment_df["status"].isin(["rented", "overdue"])])

            active_rentals = fc_filtered[(fc_filtered["is_returned"] == 0) & (fc_filtered["expected_return_date"].notna())].copy()
            today_ts = pd.Timestamp(date.today())
            returning_7d = len(active_rentals[active_rentals["expected_return_date"] <= today_ts + timedelta(days=7)]) if not active_rentals.empty else 0
            returning_30d = len(active_rentals[active_rentals["expected_return_date"] <= today_ts + timedelta(days=30)]) if not active_rentals.empty else 0

            uc1, uc2, uc3, uc4 = st.columns(4)
            for col, lbl, val, clr in [
                (uc1, "Total Fleet", total_fleet, "#3b82f6"),
                (uc2, "Available Now", available_now, "#22c55e"),
                (uc3, "Returning in 7 Days", returning_7d, "#f59e0b"),
                (uc4, "Returning in 30 Days", returning_30d, "#a855f7"),
            ]:
                col.markdown(
                    f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px 16px; border-radius:8px; text-align:center;'>"
                    f"<div style='font-size:28px; font-weight:700; color:{clr};'>{val}</div>"
                    f"<div style='font-size:13px; color:{clr};'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

            # Supply vs demand
            utilization_pct = (rented_now / total_fleet * 100) if total_fleet > 0 else 0

            st.markdown("---")
            st.subheader("Supply vs. Demand Summary")

            sd1, sd2 = st.columns(2)
            sd1.metric("Current Utilization", f"{utilization_pct:.0f}%")
            sd2.metric("Fleet Size", f"{total_fleet} units")

            # =============================================================
            # SECTION 7: SEASONAL PATTERNS
            # =============================================================
            st.markdown("---")
            st.subheader("Seasonal Demand Patterns")

            completed_all = fc_filtered[fc_filtered["is_returned"] == 1].copy()
            if not completed_all.empty:
                completed_all["month"] = completed_all["check_in_date"].dt.month_name()
                completed_all["month_num"] = completed_all["check_in_date"].dt.month

                monthly_pattern = (
                    completed_all.groupby(["month_num", "month"])["equipment_id"]
                    .count()
                    .reset_index()
                )
                monthly_pattern.columns = ["month_num", "Month", "Rentals"]
                monthly_pattern = monthly_pattern.sort_values("month_num")

                st.bar_chart(monthly_pattern, x="Month", y="Rentals", use_container_width=True)

                peak_month = monthly_pattern.loc[monthly_pattern["Rentals"].idxmax(), "Month"]
                st.info(f"Peak demand month: **{peak_month}**")

    conn.close()


# =====================================================================
# PAGE 6: ANOMALY DETECTION
# =====================================================================
elif page == "Anomaly Detection":
    st.markdown(
        "<h1 style='text-align:center;'>Anomaly Detection</h1>",
        unsafe_allow_html=True,
    )

    conn = sqlite3.connect("equipment_rental.db")
    df_anom = pd.read_sql_query("SELECT * FROM EquipmentRental", conn)
    conn.close()

    df_anom.replace(['NULL', 'None', '', ' '], np.nan, inplace=True)

    date_cols = ['CheckInDate', 'CheckOutDate', 'ExpectedReturnDate']
    for col in date_cols:
        if col in df_anom.columns:
            df_anom[col] = pd.to_datetime(df_anom[col], errors='coerce')

    numeric_cols = ['EngineHoursPerDay', 'IdleHoursPerDay', 'RentalDays']
    for col in numeric_cols:
        if col in df_anom.columns:
            df_anom[col] = pd.to_numeric(df_anom[col], errors='coerce').fillna(0)

    # --- Rule-based anomaly flags ---
    df_anom['Flag_Unassigned_Asset'] = df_anom['SiteID'].isna() | df_anom['LastOperatorID'].isna()
    df_anom['Flag_High_Idle_Waste'] = (df_anom['IdleHoursPerDay'] > df_anom['EngineHoursPerDay']) & (df_anom['IdleHoursPerDay'] > 5)
    df_anom['Flag_Zero_Utilization'] = (df_anom['EngineHoursPerDay'] == 0) & (df_anom['RentalDays'] > 0)
    df_anom['Flag_Date_Error'] = df_anom['CheckOutDate'] < df_anom['CheckInDate']

    # --- ML: Isolation Forest ---
    ml_features = ['EngineHoursPerDay', 'IdleHoursPerDay', 'RentalDays']
    X = df_anom[ml_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    df_anom['Flag_ML_Anomaly'] = iso_forest.fit_predict(X_scaled) == -1

    # Consolidate
    flag_columns = [col for col in df_anom.columns if col.startswith('Flag_')]
    df_anom['Requires_Action'] = df_anom[flag_columns].any(axis=1)

    # Build alert tags
    def build_alert_tags(row):
        tags = []
        if row.get('Flag_Unassigned_Asset'):
            tags.append({"severity": "high", "issue": "Unassigned Equipment - Security Risk"})
        if row.get('Flag_High_Idle_Waste'):
            tags.append({"severity": "medium", "issue": f"Long Idle Hours ({row.get('IdleHoursPerDay', 0)} hrs/day)"})
        if row.get('Flag_Zero_Utilization'):
            tags.append({"severity": "high", "issue": "Zero Utilization - Asset Abandoned"})
        if row.get('Flag_Date_Error'):
            tags.append({"severity": "critical", "issue": "Chronological Data Corruption"})
        if row.get('Flag_ML_Anomaly'):
            tags.append({"severity": "info", "issue": "AI Flag: Abnormal Usage Pattern Detected"})
        return tags

    df_anom['Alert_Details'] = df_anom.apply(build_alert_tags, axis=1)
    alerts_df = df_anom[df_anom['Requires_Action']].copy()

    total = len(df_anom)
    anomaly_count = len(alerts_df)
    clean_count = total - anomaly_count

    # --- Summary cards ---
    severity_colors = {"critical": "#dc2626", "high": "#ef4444", "medium": "#f59e0b", "info": "#3b82f6"}
    sev_counts = {"critical": 0, "high": 0, "medium": 0, "info": 0}
    for _, row in alerts_df.iterrows():
        for tag in row['Alert_Details']:
            sev = tag['severity']
            if sev in sev_counts:
                sev_counts[sev] += 1

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(
        f"<div style='background:#22c55e20; border-left:4px solid #22c55e; padding:12px 16px; border-radius:8px; text-align:center;'>"
        f"<div style='font-size:28px; font-weight:700; color:#22c55e;'>{total}</div>"
        f"<div style='font-size:13px; color:#22c55e;'>Total Scanned</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div style='background:#ef444420; border-left:4px solid #ef4444; padding:12px 16px; border-radius:8px; text-align:center;'>"
        f"<div style='font-size:28px; font-weight:700; color:#ef4444;'>{anomaly_count}</div>"
        f"<div style='font-size:13px; color:#ef4444;'>Anomalies</div></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div style='background:#22c55e20; border-left:4px solid #22c55e; padding:12px 16px; border-radius:8px; text-align:center;'>"
        f"<div style='font-size:28px; font-weight:700; color:#22c55e;'>{clean_count}</div>"
        f"<div style='font-size:13px; color:#22c55e;'>Clean</div></div>",
        unsafe_allow_html=True,
    )
    for col, (sev, cnt) in zip([c4, c5, c6], [("critical", sev_counts["critical"]), ("high", sev_counts["high"]), ("medium", sev_counts["medium"])]):
        clr = severity_colors[sev]
        col.markdown(
            f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px 16px; border-radius:8px; text-align:center;'>"
            f"<div style='font-size:28px; font-weight:700; color:{clr};'>{cnt}</div>"
            f"<div style='font-size:13px; color:{clr};'>{sev.title()}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Filter ---
    filter_opts = ["All Anomalies", "Unassigned Asset", "High Idle Waste", "Zero Utilization", "Date Error", "ML Anomaly"]
    sel_filter = st.selectbox("Filter by Anomaly Type", filter_opts)

    display_df = alerts_df.copy()
    if sel_filter == "Unassigned Asset":
        display_df = display_df[display_df["Flag_Unassigned_Asset"]]
    elif sel_filter == "High Idle Waste":
        display_df = display_df[display_df["Flag_High_Idle_Waste"]]
    elif sel_filter == "Zero Utilization":
        display_df = display_df[display_df["Flag_Zero_Utilization"]]
    elif sel_filter == "Date Error":
        display_df = display_df[display_df["Flag_Date_Error"]]
    elif sel_filter == "ML Anomaly":
        display_df = display_df[display_df["Flag_ML_Anomaly"]]

    if display_df.empty:
        st.info("No anomalies found for this filter.")
    else:
        for _, row in display_df.iterrows():
            tags = row['Alert_Details']
            max_sev = "info"
            for t in tags:
                if t["severity"] == "critical":
                    max_sev = "critical"
                    break
                elif t["severity"] == "high" and max_sev != "critical":
                    max_sev = "high"
                elif t["severity"] == "medium" and max_sev not in ("critical", "high"):
                    max_sev = "medium"
            color = severity_colors.get(max_sev, "#6b7280")

            eq_id = row['EquipmentID']
            eq_type = row['Type']
            site = row['SiteID'] if pd.notna(row['SiteID']) else '—'
            op = row['LastOperatorID'] if pd.notna(row['LastOperatorID']) else '—'

            tag_html = ""
            for t in tags:
                t_clr = severity_colors.get(t["severity"], "#6b7280")
                tag_html += (
                    f"<span style='background:{t_clr}20; color:{t_clr}; "
                    f"padding:2px 8px; border-radius:8px; font-size:11px; font-weight:600; margin-right:6px;'>"
                    f"{t['severity'].upper()}</span>"
                )

            issue_list = "<br>".join([f"• {t['issue']}" for t in tags])

            st.markdown(
                f"<div style='background:{color}08; border-left:5px solid {color}; "
                f"padding:14px 18px; border-radius:8px; margin-bottom:10px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<div>"
                f"<b style='font-size:16px;'>{eq_id}</b> "
                f"<span style='font-size:13px; color:#6b7280;'>({eq_type})</span>"
                f"</div>"
                f"<div>{tag_html}</div>"
                f"</div>"
                f"<div style='margin-top:8px; font-size:14px; color:#374151;'>{issue_list}</div>"
                f"<div style='margin-top:6px; font-size:12px; color:#9ca3af;'>"
                f"Operator: {op} &nbsp;|&nbsp; Site: {site} &nbsp;|&nbsp; "
                f"Engine: {row['EngineHoursPerDay']} hrs/day &nbsp;|&nbsp; "
                f"Idle: {row['IdleHoursPerDay']} hrs/day &nbsp;|&nbsp; "
                f"Rental Days: {int(row['RentalDays'])}"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# =====================================================================
# PAGE 7: SMART SCHEDULING
# =====================================================================
elif page == "Smart Scheduling":
    st.markdown(
        "<h1 style='text-align:center;'>Smart Scheduling</h1>",
        unsafe_allow_html=True,
    )
    st.caption("AI-recommended operator-equipment assignments based on historical performance and site experience")

    import os
    SYNTH_DB = "equipment_rental_synthetic.db"
    MAIN_DB = "equipment_rental.db"
    sched_db = SYNTH_DB if os.path.exists(SYNTH_DB) else MAIN_DB
    conn = sqlite3.connect(sched_db)

    history_df = pd.read_sql_query("""
        SELECT r.operator_id, r.site_id, r.equipment_id, e.type AS equipment_type,
               r.rental_days, r.is_returned,
               r.engine_hours_per_day, r.idle_hours_per_day,
               r.check_in_date, r.actual_return_date, r.expected_return_date
        FROM rentals r
        JOIN equipment e ON r.equipment_id = e.equipment_id
        WHERE r.operator_id IS NOT NULL
          AND r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
        ORDER BY r.check_in_date
    """, conn)

    available_eq = pd.read_sql_query(
        "SELECT equipment_id, type, status FROM equipment WHERE status = 'available' ORDER BY equipment_id", conn
    )
    busy_operators = pd.read_sql_query(
        "SELECT DISTINCT operator_id FROM rentals WHERE is_returned = 0 AND operator_id IS NOT NULL", conn
    )
    conn.close()

    if history_df.empty:
        st.warning("No rental history available for scheduling recommendations.")
    else:
        history_df["check_in_date"] = pd.to_datetime(history_df["check_in_date"])
        history_df["actual_return_date"] = pd.to_datetime(history_df["actual_return_date"])
        history_df["expected_return_date"] = pd.to_datetime(history_df["expected_return_date"])

        completed = history_df[history_df["is_returned"] == 1].copy()
        completed["utilization"] = completed["engine_hours_per_day"] / (
            completed["engine_hours_per_day"] + completed["idle_hours_per_day"]
        ).replace(0, float("nan"))

        completed["was_on_time"] = (
            completed["actual_return_date"] <= completed["expected_return_date"]
        ).astype(int)

        busy_set = set(busy_operators["operator_id"].tolist()) if not busy_operators.empty else set()
        all_operators = sorted(history_df["operator_id"].unique().tolist())
        free_operators = [op for op in all_operators if op not in busy_set]

        # --- Operator experience by site ---
        site_exp = completed.groupby(["operator_id", "site_id"]).agg(
            rentals=("equipment_id", "count"),
            total_days=("rental_days", "sum"),
            avg_utilization=("utilization", "mean"),
            on_time_pct=("was_on_time", "mean"),
        ).reset_index()
        site_exp["avg_utilization"] = (site_exp["avg_utilization"] * 100).round(1)
        site_exp["on_time_pct"] = (site_exp["on_time_pct"] * 100).round(1)

        # --- Operator experience by equipment type ---
        type_exp = completed.groupby(["operator_id", "equipment_type"]).agg(
            rentals=("equipment_id", "count"),
            total_days=("rental_days", "sum"),
            avg_utilization=("utilization", "mean"),
            on_time_pct=("was_on_time", "mean"),
        ).reset_index()
        type_exp["avg_utilization"] = (type_exp["avg_utilization"] * 100).round(1)
        type_exp["on_time_pct"] = (type_exp["on_time_pct"] * 100).round(1)

        # --- Combined experience (site + type) ---
        combo_exp = completed.groupby(["operator_id", "site_id", "equipment_type"]).agg(
            rentals=("equipment_id", "count"),
            total_days=("rental_days", "sum"),
            avg_utilization=("utilization", "mean"),
            on_time_pct=("was_on_time", "mean"),
        ).reset_index()
        combo_exp["avg_utilization"] = (combo_exp["avg_utilization"] * 100).round(1)
        combo_exp["on_time_pct"] = (combo_exp["on_time_pct"] * 100).round(1)

        # --- Scoring function ---
        def compute_score(row):
            rental_score = min(row["rentals"] / 5, 1.0) * 30
            days_score = min(row["total_days"] / 100, 1.0) * 20
            util_score = (row["avg_utilization"] / 100) * 25
            ontime_score = (row["on_time_pct"] / 100) * 25
            return round(rental_score + days_score + util_score + ontime_score, 1)

        # =============================================================
        # SECTION 1: AI RECOMMENDATION ENGINE
        # =============================================================
        st.subheader("Get AI Recommendation")

        rec1, rec2 = st.columns(2)
        with rec1:
            all_sites = sorted(history_df["site_id"].dropna().unique().tolist())
            target_site = st.selectbox("Select Target Site", all_sites, key="sched_site")
        with rec2:
            all_types = sorted(history_df["equipment_type"].dropna().unique().tolist())
            target_type = st.selectbox("Select Equipment Type", all_types, key="sched_type")

        only_free = st.checkbox("Show only available operators", value=True)

        candidates = combo_exp[
            (combo_exp["site_id"] == target_site) & (combo_exp["equipment_type"] == target_type)
        ].copy()

        site_only = site_exp[site_exp["site_id"] == target_site].copy()
        type_only = type_exp[type_exp["equipment_type"] == target_type].copy()

        all_scored = []

        scored_operators = set()
        for _, row in candidates.iterrows():
            scored_operators.add(row["operator_id"])
            all_scored.append({
                "operator_id": row["operator_id"],
                "match_type": "Exact (Site + Type)",
                "rentals": row["rentals"],
                "total_days": row["total_days"],
                "avg_utilization": row["avg_utilization"],
                "on_time_pct": row["on_time_pct"],
                "score": compute_score(row),
            })

        for _, row in site_only.iterrows():
            if row["operator_id"] not in scored_operators:
                scored_operators.add(row["operator_id"])
                all_scored.append({
                    "operator_id": row["operator_id"],
                    "match_type": "Site Experience",
                    "rentals": row["rentals"],
                    "total_days": row["total_days"],
                    "avg_utilization": row["avg_utilization"],
                    "on_time_pct": row["on_time_pct"],
                    "score": compute_score(row) * 0.7,
                })

        for _, row in type_only.iterrows():
            if row["operator_id"] not in scored_operators:
                scored_operators.add(row["operator_id"])
                all_scored.append({
                    "operator_id": row["operator_id"],
                    "match_type": "Type Experience",
                    "rentals": row["rentals"],
                    "total_days": row["total_days"],
                    "avg_utilization": row["avg_utilization"],
                    "on_time_pct": row["on_time_pct"],
                    "score": compute_score(row) * 0.6,
                })

        if all_scored:
            scored_df = pd.DataFrame(all_scored).sort_values("score", ascending=False)
            if only_free:
                scored_df = scored_df[~scored_df["operator_id"].isin(busy_set)]

            if scored_df.empty:
                st.warning("No available operators found with experience for this combination.")
            else:
                top = scored_df.iloc[0]
                st.markdown(
                    f"<div style='background:#22c55e15; border-left:5px solid #22c55e; "
                    f"padding:16px 20px; border-radius:8px; margin-bottom:16px;'>"
                    f"<div style='font-size:13px; color:#22c55e; font-weight:600; text-transform:uppercase;'>"
                    f"AI Recommended Operator</div>"
                    f"<div style='font-size:24px; font-weight:700; margin-top:4px;'>{top['operator_id']}</div>"
                    f"<div style='margin-top:8px; font-size:14px; color:#374151;'>"
                    f"<b>Match:</b> {top['match_type']} &nbsp;|&nbsp; "
                    f"<b>Score:</b> {top['score']:.1f}/100 &nbsp;|&nbsp; "
                    f"<b>Rentals:</b> {int(top['rentals'])} &nbsp;|&nbsp; "
                    f"<b>Total Days:</b> {int(top['total_days'])} &nbsp;|&nbsp; "
                    f"<b>Utilization:</b> {top['avg_utilization']:.1f}% &nbsp;|&nbsp; "
                    f"<b>On-Time:</b> {top['on_time_pct']:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("#### All Candidates (Ranked)")
                display_scored = scored_df.copy()
                display_scored["score"] = display_scored["score"].round(1)
                display_scored["is_free"] = ~display_scored["operator_id"].isin(busy_set)
                display_scored["is_free"] = display_scored["is_free"].map({True: "Available", False: "Busy"})
                display_scored.columns = [
                    "Operator", "Match Type", "Rentals", "Total Days",
                    "Utilization %", "On-Time %", "Score", "Status"
                ]
                st.dataframe(display_scored, use_container_width=True, hide_index=True)
        else:
            st.info(f"No operator has prior experience at **{target_site}** with **{target_type}**. Any available operator can be assigned.")
            if free_operators:
                st.markdown(f"**Available operators:** {', '.join(free_operators[:10])}")

        # =============================================================
        # SECTION 2: TOP OPERATORS OVERVIEW
        # =============================================================
        st.markdown("---")
        st.subheader("Top Operators — Overall Performance")

        overall = completed.groupby("operator_id").agg(
            total_rentals=("equipment_id", "count"),
            total_days=("rental_days", "sum"),
            unique_sites=("site_id", "nunique"),
            unique_types=("equipment_type", "nunique"),
            avg_utilization=("utilization", "mean"),
            on_time_pct=("was_on_time", "mean"),
        ).reset_index()
        overall["avg_utilization"] = (overall["avg_utilization"] * 100).round(1)
        overall["on_time_pct"] = (overall["on_time_pct"] * 100).round(1)
        overall["is_free"] = ~overall["operator_id"].isin(busy_set)
        overall["is_free"] = overall["is_free"].map({True: "Available", False: "Busy"})
        overall = overall.sort_values("total_rentals", ascending=False)
        overall.columns = [
            "Operator", "Total Rentals", "Total Days", "Sites Worked",
            "Types Operated", "Utilization %", "On-Time %", "Status"
        ]

        top_cards = overall.head(5)
        cols = st.columns(5)
        for col, (_, row) in zip(cols, top_cards.iterrows()):
            clr = "#22c55e" if row["Status"] == "Available" else "#6b7280"
            col.markdown(
                f"<div style='background:{clr}15; border-left:4px solid {clr}; padding:12px 14px; border-radius:8px; text-align:center;'>"
                f"<div style='font-size:20px; font-weight:700; color:{clr};'>{row['Operator']}</div>"
                f"<div style='font-size:12px; color:#6b7280; margin-top:4px;'>"
                f"{int(row['Total Rentals'])} rentals &nbsp;|&nbsp; {int(row['Total Days'])} days<br>"
                f"{int(row['Sites Worked'])} sites &nbsp;|&nbsp; {int(row['Types Operated'])} types<br>"
                f"Util: {row['Utilization %']}% &nbsp;|&nbsp; On-Time: {row['On-Time %']}%"
                f"</div>"
                f"<div style='margin-top:6px;'>"
                f"<span style='background:{clr}25; color:{clr}; padding:2px 8px; border-radius:8px; font-size:11px; font-weight:600;'>"
                f"{row['Status']}</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.dataframe(overall, use_container_width=True, hide_index=True)

# =====================================================================
# PAGE 8: PREDICTIVE MAINTENANCE
# =====================================================================
elif page == "Predictive Maintenance":
    st.markdown(
        "<h1 style='text-align:center;'>Predictive Maintenance</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Predicts when equipment will need servicing based on cumulative engine hours, idle wear, and equipment age")

    import os
    SYNTH_DB = "equipment_rental_synthetic.db"
    MAIN_DB = "equipment_rental.db"
    maint_db = SYNTH_DB if os.path.exists(SYNTH_DB) else MAIN_DB
    conn = sqlite3.connect(maint_db)

    maint_df = pd.read_sql_query("""
        SELECT r.equipment_id, e.type, e.age, e.status,
               r.rental_days, r.is_returned,
               r.engine_hours_per_day, r.idle_hours_per_day,
               r.check_in_date, r.actual_return_date
        FROM rentals r
        JOIN equipment e ON r.equipment_id = e.equipment_id
        WHERE r.operator_id IS NOT NULL
          AND r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
        ORDER BY r.check_in_date
    """, conn)
    conn.close()

    if maint_df.empty:
        st.warning("No rental data available for maintenance predictions.")
    else:
        maint_df["check_in_date"] = pd.to_datetime(maint_df["check_in_date"])
        maint_df["actual_return_date"] = pd.to_datetime(maint_df["actual_return_date"])
        completed_m = maint_df[maint_df["is_returned"] == 1].copy()

        # --- MAINTENANCE SCHEDULE PER EQUIPMENT TYPE ---
        # Based on real heavy equipment OEM service intervals
        # Each tier: (cumulative_engine_hours_threshold, service_name, components, est_cost, downtime_days)
        MAINT_SCHEDULE = {
            "Excavator": [
                (250,  "Basic Service",        ["Engine oil & filter", "Air filter", "Coolant level", "Grease points"], 800, 1),
                (500,  "Intermediate Service",  ["Hydraulic oil & filter", "Fuel filter", "Fan belts", "Battery check"], 2200, 2),
                (1000, "Major Service",         ["Hydraulic pump inspection", "Undercarriage wear check", "Turbocharger", "Injector calibration"], 5500, 3),
                (2000, "Engine Overhaul",       ["Piston rings & liners", "Main bearings", "Valve train", "Cooling system rebuild"], 15000, 7),
                (4000, "Complete Rebuild",      ["Full engine rebuild", "Hydraulic system rebuild", "Track/undercarriage replace", "Electrical system overhaul"], 35000, 14),
            ],
            "Crane": [
                (250,  "Basic Service",        ["Engine oil & filter", "Wire rope inspection", "Coolant level", "Grease points"], 900, 1),
                (500,  "Intermediate Service",  ["Hydraulic oil & filter", "Brake pads", "Sheave inspection", "Load test"], 2800, 2),
                (1000, "Major Service",         ["Boom structural inspection", "Slew ring bearing", "Outrigger cylinders", "Safety device calibration"], 7000, 4),
                (2000, "Engine Overhaul",       ["Piston rings & liners", "Transmission rebuild", "Winch drum inspection", "Cooling system rebuild"], 18000, 7),
                (4000, "Complete Rebuild",      ["Full engine rebuild", "Boom refurbishment", "Hydraulic system rebuild", "Electrical system overhaul"], 42000, 14),
            ],
            "Bulldozer": [
                (250,  "Basic Service",        ["Engine oil & filter", "Air filter", "Track tension adjust", "Grease points"], 750, 1),
                (500,  "Intermediate Service",  ["Hydraulic oil & filter", "Fuel filter", "Blade cutting edge check", "Fan belts"], 2000, 2),
                (1000, "Major Service",         ["Final drive inspection", "Undercarriage measurement", "Turbocharger", "Steering clutch adjust"], 5000, 3),
                (2000, "Engine Overhaul",       ["Piston rings & liners", "Main bearings", "Track chain replace", "Transmission rebuild"], 14000, 7),
                (4000, "Complete Rebuild",      ["Full engine rebuild", "Undercarriage rebuild", "Hydraulic system rebuild", "Blade & ripper overhaul"], 32000, 14),
            ],
            "Grader": [
                (250,  "Basic Service",        ["Engine oil & filter", "Air filter", "Tire pressure check", "Grease points"], 700, 1),
                (500,  "Intermediate Service",  ["Hydraulic oil & filter", "Circle & moldboard wear", "Brake adjustment", "Fuel filter"], 1800, 2),
                (1000, "Major Service",         ["Tandem drive inspection", "Articulation pins & bearings", "Turbocharger", "Scarifier teeth"], 4500, 3),
                (2000, "Engine Overhaul",       ["Piston rings & liners", "Main bearings", "Transmission rebuild", "Circle gear replace"], 12000, 7),
                (4000, "Complete Rebuild",      ["Full engine rebuild", "All-wheel-drive rebuild", "Hydraulic system rebuild", "Frame straighten & weld"], 28000, 14),
            ],
        }

        # --- IDLE WEAR FACTOR ---
        # Idling causes real damage: carbon buildup in cylinders, DPF clogging,
        # fuel injector coking, wet stacking, coolant degradation, battery sulfation.
        # Industry standard: 1 idle hour = 0.3 engine-equivalent wear hours.
        IDLE_WEAR_FACTOR = 0.3

        # --- AGE DEGRADATION ---
        # Older machines need more frequent servicing: wear accelerates with age.
        # Reduce effective service interval by 3% per year of age.
        AGE_PENALTY_PER_YEAR = 0.03

        # --- COMPONENT WEAR FROM IDLE ---
        # Specific components degraded by excessive idling
        IDLE_DAMAGE_COMPONENTS = [
            {"component": "Diesel Particulate Filter (DPF)", "idle_hr_trigger": 300,
             "issue": "Soot accumulation from incomplete combustion during idling", "action": "Forced regen or manual cleaning"},
            {"component": "Fuel Injectors", "idle_hr_trigger": 500,
             "issue": "Carbon deposits and coking from low-temp combustion", "action": "Ultrasonic cleaning or replacement"},
            {"component": "Exhaust Gas Recirculation (EGR)", "idle_hr_trigger": 400,
             "issue": "Carbon buildup restricting exhaust flow", "action": "EGR valve cleaning or replacement"},
            {"component": "Coolant System", "idle_hr_trigger": 600,
             "issue": "Coolant degradation from prolonged low-temp operation", "action": "Coolant flush and refill"},
            {"component": "Battery & Alternator", "idle_hr_trigger": 700,
             "issue": "Undercharging during idle leads to sulfation", "action": "Battery load test and replacement if needed"},
            {"component": "Turbocharger Seals", "idle_hr_trigger": 800,
             "issue": "Oil leakage past seals due to low exhaust pressure during idle", "action": "Turbo seal inspection and replacement"},
        ]

        # --- COMPUTE PER-EQUIPMENT MAINTENANCE STATUS ---
        equip_list = []
        for eq_id in completed_m["equipment_id"].unique():
            eq_data = completed_m[completed_m["equipment_id"] == eq_id].sort_values("check_in_date")
            eq_type = eq_data["type"].iloc[0]
            eq_age = eq_data["age"].iloc[0]

            total_engine_hrs = (eq_data["engine_hours_per_day"] * eq_data["rental_days"]).sum()
            total_idle_hrs = (eq_data["idle_hours_per_day"] * eq_data["rental_days"]).sum()
            total_days = eq_data["rental_days"].sum()
            num_rentals = len(eq_data)

            effective_hours = total_engine_hrs + (total_idle_hrs * IDLE_WEAR_FACTOR)

            age_factor = 1 - (eq_age * AGE_PENALTY_PER_YEAR)
            age_factor = max(age_factor, 0.5)

            schedule = MAINT_SCHEDULE.get(eq_type, MAINT_SCHEDULE["Excavator"])
            adjusted_schedule = [(thresh * age_factor, name, comps, cost, days) for thresh, name, comps, cost, days in schedule]

            upcoming_services = []
            for thresh, name, comps, cost, dtime in adjusted_schedule:
                cycles_done = int(effective_hours // thresh)
                next_due_at = (cycles_done + 1) * thresh
                hrs_to_next = next_due_at - effective_hours
                upcoming_services.append({
                    "threshold": next_due_at, "name": name,
                    "components": comps, "cost": cost, "downtime": dtime,
                    "hrs_to_next": hrs_to_next, "cycle": cycles_done + 1,
                    "interval": thresh,
                })
            upcoming_services.sort(key=lambda s: s["hrs_to_next"])
            next_service = upcoming_services[0]

            hrs_remaining = max(next_service["hrs_to_next"], 0)

            avg_daily_engine = total_engine_hrs / total_days if total_days > 0 else 0
            avg_daily_idle = total_idle_hrs / total_days if total_days > 0 else 0
            avg_daily_effective = avg_daily_engine + (avg_daily_idle * IDLE_WEAR_FACTOR)
            days_until_service = int(hrs_remaining / avg_daily_effective) if avg_daily_effective > 0 else 999

            if days_until_service <= 0:
                risk = "Critical"
                risk_color = "#dc2626"
            elif days_until_service <= 14:
                risk = "High"
                risk_color = "#ef4444"
            elif days_until_service <= 30:
                risk = "Warning"
                risk_color = "#f59e0b"
            elif days_until_service <= 60:
                risk = "Monitor"
                risk_color = "#3b82f6"
            else:
                risk = "Good"
                risk_color = "#22c55e"

            idle_ratio = total_idle_hrs / (total_engine_hrs + total_idle_hrs) * 100 if (total_engine_hrs + total_idle_hrs) > 0 else 0

            idle_flags = []
            for comp in IDLE_DAMAGE_COMPONENTS:
                if total_idle_hrs >= comp["idle_hr_trigger"]:
                    idle_flags.append(comp)

            equip_list.append({
                "equipment_id": eq_id,
                "type": eq_type,
                "age": eq_age,
                "total_engine_hrs": round(total_engine_hrs, 1),
                "total_idle_hrs": round(total_idle_hrs, 1),
                "effective_hrs": round(effective_hours, 1),
                "idle_ratio": round(idle_ratio, 1),
                "avg_daily_engine": round(avg_daily_engine, 1),
                "avg_daily_idle": round(avg_daily_idle, 1),
                "next_service": next_service["name"],
                "next_threshold": round(next_service["threshold"], 0),
                "service_interval": round(next_service["interval"], 0),
                "upcoming_services": upcoming_services,
                "hrs_remaining": round(hrs_remaining, 1),
                "days_until_service": days_until_service,
                "est_cost": next_service["cost"],
                "downtime_days": next_service["downtime"],
                "components": next_service["components"],
                "risk": risk,
                "risk_color": risk_color,
                "idle_flags": idle_flags,
                "total_days": total_days,
                "num_rentals": num_rentals,
            })

        equip_maint = pd.DataFrame(equip_list).sort_values("days_until_service")

        # =============================================================
        # SECTION 1: FLEET HEALTH SUMMARY
        # =============================================================
        risk_counts = equip_maint["risk"].value_counts().to_dict()
        total_cost = equip_maint[equip_maint["days_until_service"] <= 30]["est_cost"].sum()

        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
        for col, (lbl, clr) in zip(
            [rc1, rc2, rc3, rc4, rc5],
            [("Critical", "#dc2626"), ("High", "#ef4444"), ("Warning", "#f59e0b"), ("Monitor", "#3b82f6"), ("Good", "#22c55e")],
        ):
            cnt = risk_counts.get(lbl, 0)
            col.markdown(
                f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px 16px; border-radius:8px; text-align:center;'>"
                f"<div style='font-size:28px; font-weight:700; color:{clr};'>{cnt}</div>"
                f"<div style='font-size:13px; color:{clr};'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )
        rc6.markdown(
            f"<div style='background:#f59e0b20; border-left:4px solid #f59e0b; padding:12px 16px; border-radius:8px; text-align:center;'>"
            f"<div style='font-size:28px; font-weight:700; color:#f59e0b;'>${total_cost:,.0f}</div>"
            f"<div style='font-size:13px; color:#f59e0b;'>30-Day Maint. Budget</div></div>",
            unsafe_allow_html=True,
        )

        # =============================================================
        # SECTION 2: MAINTENANCE ALERTS
        # =============================================================
        st.markdown("---")
        st.subheader("Maintenance Alerts")

        maint_filter = st.selectbox("Filter by Risk Level", ["All", "Critical", "High", "Warning", "Monitor", "Good"], key="maint_risk")
        type_filter_m = st.selectbox("Filter by Equipment Type", ["All"] + sorted(equip_maint["type"].unique().tolist()), key="maint_type")

        display_maint = equip_maint.copy()
        if maint_filter != "All":
            display_maint = display_maint[display_maint["risk"] == maint_filter]
        if type_filter_m != "All":
            display_maint = display_maint[display_maint["type"] == type_filter_m]

        for _, eq in display_maint.iterrows():
            clr = eq["risk_color"]

            comp_list = " &nbsp;|&nbsp; ".join(eq["components"][:4])

            tier_order = ["Basic Service", "Intermediate Service", "Major Service", "Engine Overhaul", "Complete Rebuild"]
            tier_colors = {"Basic Service": "#22c55e", "Intermediate Service": "#3b82f6",
                           "Major Service": "#f59e0b", "Engine Overhaul": "#ef4444", "Complete Rebuild": "#dc2626"}
            svc_by_name = {s["name"]: s for s in eq["upcoming_services"]}
            svc_rows = ""
            for tier_name in tier_order:
                svc = svc_by_name.get(tier_name)
                if svc is None:
                    continue
                svc_clr = tier_colors.get(tier_name, "#6b7280")
                svc_pct = max(0, min(100, 100 - (svc["hrs_to_next"] / svc["interval"] * 100))) if svc["interval"] > 0 else 0
                svc_rows += (
                    f"<div style='display:flex; align-items:center; gap:8px; margin-top:4px; font-size:13px;'>"
                    f"<span style='min-width:155px; color:{svc_clr}; font-weight:600;'>{svc['name']}</span>"
                    f"<div style='flex:1; background:#e5e7eb; border-radius:4px; height:6px; overflow:hidden;'>"
                    f"<div style='background:{svc_clr}; width:{svc_pct:.0f}%; height:100%;'></div></div>"
                    f"<span style='min-width:90px; text-align:right; color:#6b7280;'>{svc['hrs_to_next']:.0f} hrs left</span>"
                    f"<span style='min-width:70px; text-align:right; color:#9ca3af;'>${svc['cost']:,}</span>"
                    f"<span style='min-width:55px; text-align:right; color:#9ca3af;'>{svc['downtime']}d</span>"
                    f"</div>"
                )

            st.markdown(
                f"<div style='background:{clr}08; border-left:5px solid {clr}; padding:16px 20px; border-radius:8px; margin-bottom:10px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<div>"
                f"<b style='font-size:17px;'>{eq['equipment_id']}</b> "
                f"<span style='font-size:13px; color:#6b7280;'>({eq['type']} | Age: {eq['age']} yrs)</span>"
                f"</div>"
                f"<span style='background:{clr}25; color:{clr}; padding:3px 12px; border-radius:10px; font-size:12px; font-weight:700;'>"
                f"{eq['risk']}</span>"
                f"</div>"
                f"<div style='margin-top:10px; display:flex; gap:24px; font-size:14px; color:#374151;'>"
                f"<div><b>Next Service:</b> {eq['next_service']}</div>"
                f"<div><b>Days Left:</b> {eq['days_until_service']}</div>"
                f"<div><b>Est. Cost:</b> ${eq['est_cost']:,}</div>"
                f"<div><b>Downtime:</b> {eq['downtime_days']} day{'s' if eq['downtime_days'] > 1 else ''}</div>"
                f"</div>"
                f"<div style='margin-top:10px; font-size:12px; color:#6b7280;'><b>All Service Tiers:</b></div>"
                f"<div style='display:flex; align-items:center; gap:8px; margin-top:4px; font-size:11px; color:#9ca3af;'>"
                f"<span style='min-width:155px;'>Service</span>"
                f"<span style='flex:1; text-align:center;'>Wear Progress</span>"
                f"<span style='min-width:90px; text-align:right;'>Hrs Left</span>"
                f"<span style='min-width:70px; text-align:right;'>Cost</span>"
                f"<span style='min-width:55px; text-align:right;'>Downtime</span>"
                f"</div>"
                f"{svc_rows}"
                f"<div style='display:flex; justify-content:space-between; font-size:11px; color:#9ca3af; margin-top:6px;'>"
                f"<span>Effective hours: {eq['effective_hrs']}</span>"
                f"<span>Components: {comp_list}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # =============================================================
        # SECTION 3: IDLE EMISSION DAMAGE REPORT
        # =============================================================
        st.markdown("---")
        st.subheader("Idle Wear & Emission Damage")
        st.caption("Components degraded by excessive idling — carbon buildup, DPF clogging, coolant issues")

        idle_report = []
        for _, eq in equip_maint.iterrows():
            for flag in eq["idle_flags"]:
                idle_report.append({
                    "Equipment": eq["equipment_id"],
                    "Type": eq["type"],
                    "Total Idle Hrs": eq["total_idle_hrs"],
                    "Idle Ratio": f"{eq['idle_ratio']}%",
                    "Component": flag["component"],
                    "Issue": flag["issue"],
                    "Action Required": flag["action"],
                })

        if idle_report:
            idle_rpt_df = pd.DataFrame(idle_report)

            comp_counts = idle_rpt_df["Component"].value_counts()
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Equipment with Idle Damage", f"{idle_rpt_df['Equipment'].nunique()}/{len(equip_maint)}")
            ic2.metric("Total Component Alerts", len(idle_report))
            most_common = comp_counts.index[0] if len(comp_counts) > 0 else "N/A"
            ic3.metric("Most Affected Component", most_common)

            st.dataframe(idle_rpt_df, use_container_width=True, hide_index=True)
        else:
            st.success("No equipment has accumulated enough idle hours to trigger component damage alerts.")

        # =============================================================
        # SECTION 4: MAINTENANCE TIMELINE
        # =============================================================
        st.markdown("---")
        st.subheader("Maintenance Timeline")
        st.caption("When each equipment unit is predicted to need its next service")

        timeline = equip_maint[["equipment_id", "type", "age", "next_service", "days_until_service",
                                "hrs_remaining", "effective_hrs", "est_cost", "risk"]].copy()
        timeline.columns = ["Equipment", "Type", "Age", "Next Service", "Days Until Service",
                            "Hrs Remaining", "Effective Hrs", "Est. Cost ($)", "Risk"]
        timeline = timeline.sort_values("Days Until Service")
        st.dataframe(timeline, use_container_width=True, hide_index=True)

        # =============================================================
        # SECTION 5: MAINTENANCE COST PROJECTION
        # =============================================================
        st.markdown("---")
        st.subheader("Maintenance Cost Projection")

        cost_7d = equip_maint[equip_maint["days_until_service"] <= 7]["est_cost"].sum()
        cost_14d = equip_maint[equip_maint["days_until_service"] <= 14]["est_cost"].sum()
        cost_30d = equip_maint[equip_maint["days_until_service"] <= 30]["est_cost"].sum()
        cost_60d = equip_maint[equip_maint["days_until_service"] <= 60]["est_cost"].sum()
        cost_90d = equip_maint[equip_maint["days_until_service"] <= 90]["est_cost"].sum()

        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        for col, lbl, val, clr in [
            (cc1, "Next 7 Days", cost_7d, "#dc2626"),
            (cc2, "Next 14 Days", cost_14d, "#ef4444"),
            (cc3, "Next 30 Days", cost_30d, "#f59e0b"),
            (cc4, "Next 60 Days", cost_60d, "#3b82f6"),
            (cc5, "Next 90 Days", cost_90d, "#22c55e"),
        ]:
            col.markdown(
                f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px 16px; border-radius:8px; text-align:center;'>"
                f"<div style='font-size:24px; font-weight:700; color:{clr};'>${val:,.0f}</div>"
                f"<div style='font-size:13px; color:{clr};'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

        cost_by_type = equip_maint.groupby("type")["est_cost"].sum().reset_index()
        cost_by_type.columns = ["Equipment Type", "Total Upcoming Cost ($)"]
        cost_by_type = cost_by_type.sort_values("Total Upcoming Cost ($)", ascending=False)
        st.bar_chart(cost_by_type.set_index("Equipment Type"), use_container_width=True)

        # =============================================================
        # SECTION 6: MAINTENANCE SCHEDULE REFERENCE
        # =============================================================
        st.markdown("---")
        st.subheader("Maintenance Schedule Reference")
        st.caption("Standard OEM service intervals per equipment type (adjusted by age)")

        ref_type = st.selectbox("Select Equipment Type", list(MAINT_SCHEDULE.keys()), key="maint_ref")
        ref_schedule = MAINT_SCHEDULE[ref_type]

        for thresh, name, comps, cost, dtime in ref_schedule:
            st.markdown(
                f"<div style='background:#f3f4f615; border-left:4px solid #3b82f6; padding:12px 16px; border-radius:8px; margin-bottom:8px;'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<b>{name}</b>"
                f"<span style='color:#6b7280;'>Every {int(thresh)} engine hrs &nbsp;|&nbsp; ~${cost:,} &nbsp;|&nbsp; {dtime} day{'s' if dtime > 1 else ''} downtime</span>"
                f"</div>"
                f"<div style='margin-top:6px; font-size:13px; color:#374151;'>"
                + " &nbsp;&bull;&nbsp; ".join(comps)
                + f"</div></div>",
                unsafe_allow_html=True,
            )


# =====================================================================
# PAGE 9: ASK FLEET AI
# =====================================================================
elif page == "Ask Fleet AI":
    st.markdown(
        "<h1 style='text-align:center;'>Ask Fleet AI</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Ask questions in plain English — the AI queries your fleet data and returns instant answers")

    import os, re
    SYNTH_DB = "equipment_rental_synthetic.db"
    MAIN_DB = "equipment_rental.db"
    ai_db = SYNTH_DB if os.path.exists(SYNTH_DB) else MAIN_DB
    conn_ai = sqlite3.connect(ai_db)

    @st.cache_data(ttl=120)
    def load_fleet_data(_db_path):
        conn = sqlite3.connect(_db_path)
        rentals = pd.read_sql_query("""
            SELECT r.*, e.type AS equipment_type, e.status, e.daily_rental_rate, e.age
            FROM rentals r
            JOIN equipment e ON r.equipment_id = e.equipment_id
            WHERE r.check_in_date IS NOT NULL AND r.check_in_date != '1900-01-01'
        """, conn)
        equipment = pd.read_sql_query("SELECT * FROM equipment", conn)
        conn.close()
        rentals["check_in_date"] = pd.to_datetime(rentals["check_in_date"])
        rentals["actual_return_date"] = pd.to_datetime(rentals["actual_return_date"])
        rentals["expected_return_date"] = pd.to_datetime(rentals["expected_return_date"])
        return rentals, equipment

    rentals_ai, equipment_ai = load_fleet_data(ai_db)
    conn_ai.close()

    today_ai = pd.Timestamp(date.today())

    def parse_time_range(query):
        q = query.lower()
        if "last month" in q:
            start = (today_ai - pd.DateOffset(months=1)).replace(day=1)
            end = today_ai.replace(day=1) - timedelta(days=1)
            label = "Last Month"
        elif "last 3 months" in q or "last three months" in q or "past 3 months" in q:
            start = today_ai - pd.DateOffset(months=3)
            end = today_ai
            label = "Last 3 Months"
        elif "last 6 months" in q or "last six months" in q or "past 6 months" in q:
            start = today_ai - pd.DateOffset(months=6)
            end = today_ai
            label = "Last 6 Months"
        elif "this year" in q or "current year" in q:
            start = today_ai.replace(month=1, day=1)
            end = today_ai
            label = "This Year"
        elif "last year" in q or "previous year" in q:
            start = (today_ai - pd.DateOffset(years=1)).replace(month=1, day=1)
            end = (today_ai - pd.DateOffset(years=1)).replace(month=12, day=31)
            label = "Last Year"
        elif "this month" in q:
            start = today_ai.replace(day=1)
            end = today_ai
            label = "This Month"
        elif "last week" in q:
            start = today_ai - timedelta(days=7)
            end = today_ai
            label = "Last Week"
        else:
            start = rentals_ai["check_in_date"].min()
            end = today_ai
            label = "All Time"
        return start, end, label

    def extract_equipment_type(query):
        q = query.lower()
        for t in ["excavator", "crane", "bulldozer", "grader"]:
            if t in q:
                return t.title()
        return None

    def extract_site(query):
        match = re.search(r's\d{3}', query, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def extract_operator(query):
        match = re.search(r'op\d{3}', query, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def classify_intent(query):
        q = query.lower()
        if any(w in q for w in ["underutiliz", "idle", "low utiliz", "wast", "not used", "unused", "sitting"]):
            return "underutilized"
        if any(w in q for w in ["overutiliz", "high utiliz", "most used", "heavily used", "busiest", "hardest working"]):
            return "overutilized"
        if any(w in q for w in ["overdue", "late", "past due", "not returned", "delayed"]):
            return "overdue"
        if any(w in q for w in ["available", "free", "not rented", "idle equipment", "not in use"]):
            return "available"
        if any(w in q for w in ["revenue", "income", "earning", "money", "profit", "cost"]):
            return "revenue"
        if any(w in q for w in ["maintenance", "service", "repair", "breakdown", "fix"]):
            return "maintenance"
        if any(w in q for w in ["operator", "who operated", "best operator", "top operator", "experienced"]):
            return "operator"
        if any(w in q for w in ["site", "location", "which site", "busiest site", "most active site"]):
            return "site"
        if any(w in q for w in ["how many", "count", "total", "number of"]):
            return "count"
        if any(w in q for w in ["trend", "pattern", "over time", "history", "monthly", "weekly"]):
            return "trend"
        if any(w in q for w in ["forecast", "predict", "next month", "future", "expect"]):
            return "forecast"
        if any(w in q for w in ["compare", "vs", "versus", "difference between"]):
            return "compare"
        if any(w in q for w in ["longest", "shortest", "most days", "fewest", "maximum", "minimum"]):
            return "extreme"
        if any(w in q for w in ["status", "summary", "overview", "fleet health", "dashboard"]):
            return "summary"
        return "general"

    def render_answer_card(title, value, color="#3b82f6"):
        st.markdown(
            f"<div style='background:{color}15; border-left:5px solid {color}; "
            f"padding:14px 18px; border-radius:8px; margin-bottom:10px;'>"
            f"<div style='font-size:13px; color:{color}; font-weight:600; text-transform:uppercase;'>{title}</div>"
            f"<div style='font-size:22px; font-weight:700; margin-top:4px;'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    def process_query(query):
        intent = classify_intent(query)
        eq_type = extract_equipment_type(query)
        site = extract_site(query)
        operator = extract_operator(query)
        start, end, time_label = parse_time_range(query)

        data = rentals_ai.copy()
        time_filtered = data[(data["check_in_date"] >= start) & (data["check_in_date"] <= end)]
        if eq_type:
            time_filtered = time_filtered[time_filtered["equipment_type"] == eq_type]
        if site:
            time_filtered = time_filtered[time_filtered["site_id"] == site]
        if operator:
            time_filtered = time_filtered[time_filtered["operator_id"] == operator]

        completed = time_filtered[time_filtered["is_returned"] == 1].copy()
        if not completed.empty:
            completed["utilization"] = completed["engine_hours_per_day"] / (
                completed["engine_hours_per_day"] + completed["idle_hours_per_day"]
            ).replace(0, float("nan"))

        type_label = eq_type if eq_type else "All Equipment"
        scope = f"{type_label} | {time_label}"
        if site:
            scope += f" | Site: {site}"
        if operator:
            scope += f" | Operator: {operator}"

        st.markdown(f"**Scope:** {scope}")
        st.markdown(f"**Records found:** {len(time_filtered)} rentals ({len(completed)} completed)")
        st.markdown("---")

        if intent == "underutilized":
            if completed.empty:
                st.info("No completed rentals found for this filter.")
                return
            equip_util = completed.groupby(["equipment_id", "equipment_type"]).agg(
                avg_util=("utilization", "mean"),
                avg_engine=("engine_hours_per_day", "mean"),
                avg_idle=("idle_hours_per_day", "mean"),
                total_days=("rental_days", "sum"),
                rentals=("equipment_id", "count"),
            ).reset_index()
            equip_util["avg_util"] = (equip_util["avg_util"] * 100).round(1)
            equip_util["avg_engine"] = equip_util["avg_engine"].round(1)
            equip_util["avg_idle"] = equip_util["avg_idle"].round(1)
            under = equip_util[equip_util["avg_util"] < 60].sort_values("avg_util")

            render_answer_card("Underutilized Equipment (<60% utilization)", f"{len(under)} out of {len(equip_util)} units", "#f59e0b")

            if not under.empty:
                under_display = under.copy()
                under_display.columns = ["Equipment", "Type", "Utilization %", "Avg Engine Hrs/Day", "Avg Idle Hrs/Day", "Total Days", "Rentals"]
                st.dataframe(under_display, use_container_width=True, hide_index=True)
                st.bar_chart(under.set_index("equipment_id")["avg_util"], use_container_width=True)
            else:
                st.success("All equipment is well-utilized (>60%).")

        elif intent == "overutilized":
            if completed.empty:
                st.info("No completed rentals found.")
                return
            equip_util = completed.groupby(["equipment_id", "equipment_type"]).agg(
                avg_util=("utilization", "mean"),
                avg_engine=("engine_hours_per_day", "mean"),
                total_days=("rental_days", "sum"),
                rentals=("equipment_id", "count"),
            ).reset_index()
            equip_util["avg_util"] = (equip_util["avg_util"] * 100).round(1)
            equip_util["avg_engine"] = equip_util["avg_engine"].round(1)
            top = equip_util.sort_values("avg_util", ascending=False).head(10)

            render_answer_card("Most Utilized Equipment (Top 10)", f"Highest: {top.iloc[0]['avg_util']}%", "#22c55e")

            top_display = top.copy()
            top_display.columns = ["Equipment", "Type", "Utilization %", "Avg Engine Hrs/Day", "Total Days", "Rentals"]
            st.dataframe(top_display, use_container_width=True, hide_index=True)

        elif intent == "overdue":
            active = time_filtered[(time_filtered["is_returned"] == 0) & (time_filtered["expected_return_date"].notna())]
            overdue = active[active["expected_return_date"] < today_ai].copy()
            if not overdue.empty:
                overdue["days_overdue"] = (today_ai - overdue["expected_return_date"]).dt.days
            render_answer_card("Overdue Equipment", f"{len(overdue)} units past due date", "#ef4444")

            if not overdue.empty:
                ov_display = overdue[["equipment_id", "equipment_type", "operator_id", "site_id", "expected_return_date", "days_overdue"]].copy()
                ov_display.columns = ["Equipment", "Type", "Operator", "Site", "Expected Return", "Days Overdue"]
                ov_display = ov_display.sort_values("Days Overdue", ascending=False)
                st.dataframe(ov_display, use_container_width=True, hide_index=True)
            else:
                st.success("No overdue equipment found.")

        elif intent == "available":
            avail = equipment_ai[equipment_ai["status"] == "available"]
            if eq_type:
                avail = avail[avail["type"] == eq_type]
            render_answer_card("Available Equipment", f"{len(avail)} units ready for rental", "#22c55e")

            if not avail.empty:
                avail_display = avail[["equipment_id", "type", "daily_rental_rate", "age"]].copy()
                avail_display.columns = ["Equipment", "Type", "Daily Rate ($)", "Age (yrs)"]
                st.dataframe(avail_display, use_container_width=True, hide_index=True)

                type_counts = avail["type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                st.bar_chart(type_counts.set_index("Type"), use_container_width=True)

        elif intent == "revenue":
            if completed.empty:
                st.info("No completed rentals found.")
                return
            completed_rev = completed.copy()
            completed_rev["revenue"] = completed_rev["rental_days"] * completed_rev["daily_rental_rate"]
            total_rev = completed_rev["revenue"].sum()

            render_answer_card("Total Revenue", f"${total_rev:,.0f}", "#22c55e")

            rev_by_type = completed_rev.groupby("equipment_type")["revenue"].sum().reset_index()
            rev_by_type.columns = ["Type", "Revenue ($)"]
            rev_by_type["Revenue ($)"] = rev_by_type["Revenue ($)"].round(0).astype(int)
            rev_by_type = rev_by_type.sort_values("Revenue ($)", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Revenue by Equipment Type**")
                st.dataframe(rev_by_type, use_container_width=True, hide_index=True)
            with c2:
                st.bar_chart(rev_by_type.set_index("Type"), use_container_width=True)

            if not site:
                rev_by_site = completed_rev.groupby("site_id")["revenue"].sum().reset_index()
                rev_by_site.columns = ["Site", "Revenue ($)"]
                rev_by_site = rev_by_site.sort_values("Revenue ($)", ascending=False).head(10)
                st.markdown("**Top 10 Sites by Revenue**")
                st.bar_chart(rev_by_site.set_index("Site"), use_container_width=True)

        elif intent == "operator":
            if completed.empty:
                st.info("No completed rentals found.")
                return
            completed_op = completed.copy()
            completed_op["was_on_time"] = (completed_op["actual_return_date"] <= completed_op["expected_return_date"]).astype(int)

            op_stats = completed_op.groupby("operator_id").agg(
                total_rentals=("equipment_id", "count"),
                total_days=("rental_days", "sum"),
                avg_util=("utilization", "mean"),
                on_time=("was_on_time", "mean"),
                sites=("site_id", "nunique"),
                types=("equipment_type", "nunique"),
            ).reset_index()
            op_stats["avg_util"] = (op_stats["avg_util"] * 100).round(1)
            op_stats["on_time"] = (op_stats["on_time"] * 100).round(1)
            op_stats = op_stats.sort_values("total_rentals", ascending=False)

            best = op_stats.iloc[0]
            render_answer_card("Top Operator by Experience",
                f"{best['operator_id']} — {int(best['total_rentals'])} rentals, {best['avg_util']}% utilization, {best['on_time']}% on-time", "#3b82f6")

            op_stats.columns = ["Operator", "Rentals", "Total Days", "Utilization %", "On-Time %", "Sites", "Types"]
            st.dataframe(op_stats.head(15), use_container_width=True, hide_index=True)

        elif intent == "site":
            if time_filtered.empty:
                st.info("No rental data found.")
                return
            site_stats = time_filtered.groupby("site_id").agg(
                total_rentals=("equipment_id", "count"),
                unique_eq=("equipment_id", "nunique"),
                total_days=("rental_days", "sum"),
                unique_types=("equipment_type", "nunique"),
            ).reset_index()
            site_stats = site_stats.sort_values("total_rentals", ascending=False)

            busiest = site_stats.iloc[0]
            render_answer_card("Busiest Site",
                f"{busiest['site_id']} — {int(busiest['total_rentals'])} rentals, {int(busiest['unique_eq'])} equipment units", "#a855f7")

            site_stats.columns = ["Site", "Rentals", "Unique Equipment", "Total Days", "Equipment Types"]
            st.dataframe(site_stats, use_container_width=True, hide_index=True)
            st.bar_chart(site_stats.set_index("Site")["Rentals"], use_container_width=True)

        elif intent == "count":
            q = query.lower()
            if "active" in q or "rented" in q or "in use" in q:
                active = equipment_ai[equipment_ai["status"].isin(["rented", "overdue"])]
                if eq_type:
                    active = active[active["type"] == eq_type]
                render_answer_card("Currently Active / Rented", f"{len(active)} units", "#3b82f6")
                if not active.empty:
                    st.dataframe(active[["equipment_id", "type", "status"]], use_container_width=True, hide_index=True)
            else:
                render_answer_card("Total Rentals in Period", f"{len(time_filtered)} rental records", "#3b82f6")
                by_type = time_filtered.groupby("equipment_type")["equipment_id"].count().reset_index()
                by_type.columns = ["Type", "Count"]
                st.bar_chart(by_type.set_index("Type"), use_container_width=True)

        elif intent == "trend":
            if time_filtered.empty:
                st.info("No data found.")
                return
            weekly = time_filtered.set_index("check_in_date").resample("W")["equipment_id"].count().reset_index()
            weekly.columns = ["Week", "Rentals"]

            render_answer_card("Rental Trend", f"{len(weekly)} weeks of data | Avg: {weekly['Rentals'].mean():.1f}/week", "#3b82f6")
            st.line_chart(weekly, x="Week", y="Rentals", use_container_width=True)

            if eq_type is None:
                monthly_type = time_filtered.copy()
                monthly_type["month"] = monthly_type["check_in_date"].dt.to_period("M").dt.to_timestamp()
                pivot = monthly_type.groupby(["month", "equipment_type"])["equipment_id"].count().reset_index()
                pivot.columns = ["Month", "Type", "Rentals"]
                pivot_wide = pivot.pivot(index="Month", columns="Type", values="Rentals").fillna(0)
                st.markdown("**Trend by Equipment Type**")
                st.line_chart(pivot_wide, use_container_width=True)

        elif intent == "forecast":
            if completed.empty:
                st.info("Not enough data.")
                return
            monthly = completed.set_index("check_in_date").resample("MS")["equipment_id"].count()
            if len(monthly) >= 2:
                window = min(3, len(monthly))
                ma = monthly.rolling(window).mean().dropna()
                avg_monthly = ma.iloc[-1]
                avg_daily = avg_monthly / 30

                render_answer_card("Forecast (Moving Average)",
                    f"~{avg_monthly:.0f} rentals/month | ~{avg_daily * 30:.0f} next 30 days", "#a855f7")

                fc_dates = pd.date_range(monthly.index.max() + timedelta(days=1), periods=90, freq="D")
                fc_weekly = pd.Series(avg_daily, index=fc_dates).resample("W").sum().reset_index()
                fc_weekly.columns = ["Week", "Predicted"]
                fc_weekly["Predicted"] = fc_weekly["Predicted"].round(1)
                st.bar_chart(fc_weekly, x="Week", y="Predicted", use_container_width=True)

        elif intent == "maintenance":
            equip_data = equipment_ai.copy()
            if eq_type:
                equip_data = equip_data[equip_data["type"] == eq_type]
            completed_m = rentals_ai[rentals_ai["is_returned"] == 1].copy()

            maint_list = []
            for eq_id in equip_data["equipment_id"]:
                eq_rentals = completed_m[completed_m["equipment_id"] == eq_id]
                if eq_rentals.empty:
                    continue
                total_eng = (eq_rentals["engine_hours_per_day"] * eq_rentals["rental_days"]).sum()
                total_idle = (eq_rentals["idle_hours_per_day"] * eq_rentals["rental_days"]).sum()
                eff = total_eng + total_idle * 0.3
                eq_age = equip_data[equip_data["equipment_id"] == eq_id]["age"].iloc[0]
                age_factor = max(1 - eq_age * 0.03, 0.5)
                rebuild_thresh = 4000 * age_factor
                hrs_left = max(rebuild_thresh - eff, 0)
                total_days_r = eq_rentals["rental_days"].sum()
                daily_rate = (total_eng / total_days_r + total_idle / total_days_r * 0.3) if total_days_r > 0 else 1
                days_left = int(hrs_left / daily_rate) if daily_rate > 0 else 999
                maint_list.append({
                    "Equipment": eq_id,
                    "Type": equip_data[equip_data["equipment_id"] == eq_id]["type"].iloc[0],
                    "Age": eq_age,
                    "Effective Hrs": round(eff, 0),
                    "Next Service At": round(rebuild_thresh, 0),
                    "Days Until Service": days_left,
                })

            maint_df = pd.DataFrame(maint_list).sort_values("Days Until Service")
            urgent = maint_df[maint_df["Days Until Service"] <= 14]

            render_answer_card("Equipment Needing Service Soon",
                f"{len(urgent)} units need service within 14 days", "#ef4444")
            st.dataframe(maint_df, use_container_width=True, hide_index=True)

        elif intent == "compare":
            if completed.empty:
                st.info("No completed rentals found.")
                return
            comp = completed.groupby("equipment_type").agg(
                rentals=("equipment_id", "count"),
                avg_days=("rental_days", "mean"),
                avg_engine=("engine_hours_per_day", "mean"),
                avg_idle=("idle_hours_per_day", "mean"),
                avg_util=("utilization", "mean"),
                revenue=("daily_rental_rate", lambda x: (x * completed.loc[x.index, "rental_days"]).sum()),
            ).reset_index()
            comp["avg_days"] = comp["avg_days"].round(1)
            comp["avg_engine"] = comp["avg_engine"].round(1)
            comp["avg_idle"] = comp["avg_idle"].round(1)
            comp["avg_util"] = (comp["avg_util"] * 100).round(1)
            comp["revenue"] = comp["revenue"].round(0).astype(int)
            comp.columns = ["Type", "Rentals", "Avg Days", "Avg Engine Hrs", "Avg Idle Hrs", "Utilization %", "Revenue ($)"]

            render_answer_card("Equipment Type Comparison", f"{len(comp)} types compared", "#3b82f6")
            st.dataframe(comp, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Utilization by Type**")
                st.bar_chart(comp.set_index("Type")["Utilization %"], use_container_width=True)
            with c2:
                st.markdown("**Revenue by Type**")
                st.bar_chart(comp.set_index("Type")["Revenue ($)"], use_container_width=True)

        elif intent == "extreme":
            if completed.empty:
                st.info("No completed rentals found.")
                return
            q = query.lower()
            if "longest" in q or "most days" in q or "maximum" in q:
                top_rental = completed.sort_values("rental_days", ascending=False).head(10)
                render_answer_card("Longest Rentals", f"Max: {int(top_rental.iloc[0]['rental_days'])} days", "#f59e0b")
                display = top_rental[["equipment_id", "equipment_type", "operator_id", "site_id", "rental_days", "check_in_date"]].copy()
                display.columns = ["Equipment", "Type", "Operator", "Site", "Rental Days", "Check-In"]
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                short_rental = completed[completed["rental_days"] > 0].sort_values("rental_days").head(10)
                render_answer_card("Shortest Rentals", f"Min: {int(short_rental.iloc[0]['rental_days'])} days", "#3b82f6")
                display = short_rental[["equipment_id", "equipment_type", "operator_id", "site_id", "rental_days", "check_in_date"]].copy()
                display.columns = ["Equipment", "Type", "Operator", "Site", "Rental Days", "Check-In"]
                st.dataframe(display, use_container_width=True, hide_index=True)

        elif intent == "summary":
            total_eq = len(equipment_ai)
            avail = len(equipment_ai[equipment_ai["status"] == "available"])
            rented = len(equipment_ai[equipment_ai["status"].isin(["rented", "overdue"])])
            overdue_cnt = len(equipment_ai[equipment_ai["status"] == "overdue"])
            total_rentals = len(time_filtered)

            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, lbl, val, clr in [
                (sc1, "Fleet Size", total_eq, "#3b82f6"),
                (sc2, "Available", avail, "#22c55e"),
                (sc3, "Rented", rented, "#f59e0b"),
                (sc4, "Overdue", overdue_cnt, "#ef4444"),
            ]:
                col.markdown(
                    f"<div style='background:{clr}20; border-left:4px solid {clr}; padding:12px; border-radius:8px; text-align:center;'>"
                    f"<div style='font-size:28px; font-weight:700; color:{clr};'>{val}</div>"
                    f"<div style='font-size:13px; color:{clr};'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            type_summary = equipment_ai.groupby("type")["status"].value_counts().unstack(fill_value=0).reset_index()
            st.dataframe(type_summary, use_container_width=True, hide_index=True)

        else:
            st.markdown("**Here's a general overview based on your query:**")
            render_answer_card("Rentals in Scope", f"{len(time_filtered)} records", "#3b82f6")

            if not time_filtered.empty:
                by_type = time_filtered.groupby("equipment_type")["equipment_id"].count().reset_index()
                by_type.columns = ["Type", "Rentals"]
                st.bar_chart(by_type.set_index("Type"), use_container_width=True)

                if not completed.empty:
                    avg_days = completed["rental_days"].mean()
                    avg_eng = completed["engine_hours_per_day"].mean()
                    avg_idle = completed["idle_hours_per_day"].mean()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Rental Days", f"{avg_days:.1f}")
                    m2.metric("Avg Engine Hrs/Day", f"{avg_eng:.1f}")
                    m3.metric("Avg Idle Hrs/Day", f"{avg_idle:.1f}")

            st.info("Try asking something more specific — see the suggested questions below.")

    # --- SUGGESTED QUESTIONS ---
    st.markdown("#### Ask a Question")

    if "fleet_ai_picked" not in st.session_state:
        st.session_state.fleet_ai_picked = ""

    suggested = [
        "Which excavators were underutilized last month?",
        "Show me all overdue equipment",
        "Who is the best crane operator?",
        "What's the total revenue this year?",
        "Which site has the most rentals?",
        "Compare all equipment types",
        "Show rental trends for last 6 months",
        "How many bulldozers are available?",
        "Which equipment needs maintenance soon?",
        "What are the longest rentals this year?",
        "Give me a fleet summary",
        "Forecast demand for next month",
    ]

    st.markdown("**Quick queries** — click any to run instantly:")
    cols = st.columns(3)
    for i, q in enumerate(suggested):
        if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.fleet_ai_picked = q

    st.markdown("---")
    user_query = st.text_input(
        "Or type your own question...",
        placeholder="e.g., Which excavators were underutilized last month?",
        key="fleet_ai_query",
    )

    active_query = user_query.strip() if user_query.strip() else st.session_state.fleet_ai_picked

    if active_query:
        st.markdown("---")
        st.markdown(f"**Question:** *{active_query}*")
        st.markdown("")
        process_query(active_query)


elif page == "Report Export":
    render_report_page(TODAY)
