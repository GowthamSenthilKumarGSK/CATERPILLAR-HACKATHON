import streamlit as st
import sqlite3
import pandas as pd
from datetime import date, datetime
from db_helpers import get_connection, refresh_equipment_status

st.set_page_config(page_title="Equipment Rental Dashboard", layout="wide")

TODAY = date.today()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Check In / Out", "Usage Logging", "Alerts & Reminders"], label_visibility="collapsed")


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
        "flagged": "#a855"
        "f7",
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
