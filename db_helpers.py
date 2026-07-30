import sqlite3
from datetime import date, datetime

DB_PATH = "equipment_rental.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def derive_status(check_in, check_out, expected_return, operator, site):
    today = date.today()

    if check_out is not None:
        return "available"

    if check_in is None and check_out is None:
        return "unknown"

    if operator is None and site is None:
        return "flagged"

    if expected_return is not None:
        exp_date = datetime.strptime(expected_return, "%Y-%m-%d").date()
        if exp_date < today:
            return "overdue"

    return "rented"


def refresh_equipment_status(conn, equipment_id):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT check_in_date, actual_return_date, expected_return_date,
               operator_id, site_id
        FROM rentals
        WHERE equipment_id = ?
        ORDER BY id DESC LIMIT 1
    """, (equipment_id,))
    row = cursor.fetchone()

    if row is None:
        status = "available"
    else:
        check_in, check_out, exp_return, operator, site = row
        status = derive_status(check_in, check_out, exp_return, operator, site)

    cursor.execute(
        "UPDATE equipment SET status = ? WHERE equipment_id = ?",
        (status, equipment_id),
    )
    conn.commit()
    return status
