import streamlit as st
import sqlite3
import pandas as pd
from datetime import date

st.set_page_config(page_title="Equipment Rental Dashboard", layout="wide")

TODAY = date.today()


def get_data():
    conn = sqlite3.connect("equipment_rental.db")
    df = pd.read_sql_query("""
        SELECT e.equipment_id, e.type, e.status, e.daily_rental_rate,
               r.operator_id, r.site_id, r.expected_return_date
        FROM equipment e
        LEFT JOIN rentals r ON e.equipment_id = r.equipment_id
            AND r.id = (SELECT MAX(r2.id) FROM rentals r2 WHERE r2.equipment_id = e.equipment_id)
    """, conn)
    conn.close()
    return df


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


df = get_data()

# --- Header ---
st.markdown(
    "<h1 style='text-align:center;'>Equipment Rental Dashboard</h1>",
    unsafe_allow_html=True,
)

# --- Filters ---
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

# --- Status summary cards ---
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

# --- Column headers ---
h1, h2, h3, h4, h5 = st.columns([1.1, 0.9, 0.9, 0.8, 1.2])
h1.markdown("**Equipment ID**")
h2.markdown("**Type**")
h3.markdown("**Operator**")
h4.markdown("**Site**")
h5.markdown("**Status**")
st.markdown("<hr style='margin:2px 0; border:none; border-top:2px solid #9ca3af;'>", unsafe_allow_html=True)

# --- Equipment rows ---
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
