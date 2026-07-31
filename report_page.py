import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st
from fpdf import FPDF


class CaterpillarPDF(FPDF):

  def header(self):
    self.set_font("Arial", "B", 15)
    self.cell(0, 8, "CATERPILLAR SMART RENTAL SYSTEM", ln=True, align="C")
    self.set_font("Arial", "B", 11)
    self.cell(
        0, 6, "FLEET HEALTH & ANOMALY EXECUTIVE REPORT", ln=True, align="C"
    )
    self.set_font("Arial", "I", 9)
    self.cell(
        0,
        5,
        f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True,
        align="C",
    )
    self.ln(5)

  def footer(self):
    self.set_y(-15)
    self.set_font("Arial", "I", 8)
    self.cell(
        0,
        10,
        f"Page {self.page_no()} | Confidential - For Caterpillar Operations"
        " Only",
        align="C",
    )


def render_report_page(today):
  """Renders the executive report generation page."""
  st.markdown(
      "<h1 style='text-align:center;'>Executive Report Export</h1>",
      unsafe_allow_html=True,
  )
  st.caption(
      "Generate an enterprise-grade PDF summary for fleet managers and"
      " Caterpillar leadership."
  )

  col_info, col_action = st.columns([2, 1])

  with col_info:
    st.markdown("""
        ### What's included in this report?
        * **Fleet Overview:** Total active, available, overdue, and unassigned machines.
        * **Operational Metrics:** Rental days log and idle vs. engine hour breakdown.
        * **Critical Anomaly Summary:** High idle fuel waste, data corruption errors, and flagged equipment.
        * **Timestamp & Audit Trail:** Official snapshot generated directly from real-time database state.
        """)

  # --- FETCH DATA ---
  conn = sqlite3.connect("equipment_rental.db")

  eq_summary = pd.read_sql_query(
      "SELECT status, COUNT(*) as count FROM equipment GROUP BY status", conn
  )
  status_counts = dict(zip(eq_summary["status"], eq_summary["count"]))

  overdue_df = pd.read_sql_query(
      """
        SELECT r.equipment_id, e.type, r.operator_id, r.site_id, r.expected_return_date
        FROM rentals r JOIN equipment e ON r.equipment_id = e.equipment_id
        WHERE r.is_returned = 0 AND r.expected_return_date < ?
    """,
      conn,
      params=(today.strftime("%Y-%m-%d"),),
  )

  rental_anom = pd.read_sql_query("""SELECT r.equipment_id, e.type, r.operator_id, r.site_id, r.expected_return_date,r.engine_hours_per_day,r.idle_hours_per_day
          FROM rentals r JOIN equipment e ON r.equipment_id = e.equipment_id""", conn)
  conn.close()

  total_units = sum(status_counts.values())
  avail_units = status_counts.get("available", 0)
  rented_units = status_counts.get("rented", 0)
  overdue_units = status_counts.get("overdue", 0)
  flagged_units = status_counts.get("flagged", 0)

  # --- EXPORT ACTION ---
  with col_action:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "📄 Build PDF Report", type="primary", use_container_width=True
    ):
      with st.spinner("Compiling database records into PDF..."):
        pdf = CaterpillarPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 1. KPI Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "1. Executive KPI Overview", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"- Total Tracked Assets: {total_units}", ln=True)
        pdf.cell(0, 6, f"- Currently Available: {avail_units}", ln=True)
        pdf.cell(0, 6, f"- Active Rentals: {rented_units}", ln=True)
        pdf.cell(0, 6, f"- Overdue Returns: {overdue_units}", ln=True)
        pdf.cell(
            0, 6, f"- Flagged / Suspicious Assets: {flagged_units}", ln=True
        )
        pdf.ln(4)

        # 2. Overdue Table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "2. Critical Action Required: Overdue Assets", ln=True)
        pdf.set_font("Arial", "", 10)

        if overdue_df.empty:
          pdf.cell(0, 6, "No overdue equipment detected at this time.", ln=True)
        else:
          pdf.set_font("Arial", "B", 9)
          pdf.cell(30, 6, "Asset ID", 1)
          pdf.cell(35, 6, "Type", 1)
          pdf.cell(30, 6, "Operator", 1)
          pdf.cell(30, 6, "Site", 1)
          pdf.cell(35, 6, "Expected Return", 1)
          pdf.ln()

          pdf.set_font("Arial", "", 9)
          for _, row in overdue_df.iterrows():
            pdf.cell(30, 6, str(row["equipment_id"]), 1)
            pdf.cell(35, 6, str(row["type"]), 1)
            pdf.cell(30, 6, str(row["operator_id"] or "N/A"), 1) 
            pdf.cell(30, 6, str(row["site_id"] or "N/A"), 1)     
            pdf.cell(35, 6, str(row["expected_return_date"]), 1)
            pdf.ln()

        pdf.ln(6)

        # 3. Anomaly Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "3. Anomaly & Risk Analysis", ln=True)
        pdf.set_font("Arial", "", 10)

        high_idle = len(
            rental_anom[
                rental_anom["idle_hours_per_day"]
                > rental_anom["engine_hours_per_day"]
            ]
        )
        unassigned = len(
            rental_anom[
                rental_anom["site_id"].isna()
                | rental_anom["operator_id"].isna()
            ]
        )

        pdf.cell(
            0,
            6,
            f"- High Idle Waste Instances (OPEX Fuel Risk): {high_idle}"
            " machines",
            ln=True,
        )
        pdf.cell(
            0,
            6,
            "- Unassigned / Missing Operator Records (Ghost Assets):"
            f" {unassigned} machines",
            ln=True,
        )
        pdf.ln(8)

        # 4. Sign-off
        pdf.set_font("Arial", "I", 9)
        pdf.cell(
            0,
            6,
            "Report automatically compiled by Caterpillar Smart Rental Tracking"
            " System.",
            ln=True,
        )

        pdf_output = pdf.output(dest="S")
        if isinstance(pdf_output, str):
          pdf_bytes = pdf_output.encode("latin-1")
        elif isinstance(pdf_output, bytearray):
          pdf_bytes = bytes(pdf_output)
        else:
          pdf_bytes = pdf_output

        st.success("PDF Report generated successfully!")
        st.download_button(
            label="💾 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Caterpillar_Fleet_Report_{today.strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )