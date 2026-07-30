"""
Load synthetic_rental_data.csv into equipment_rental_synthetic.db
with the same schema (equipment + rentals) that the dashboard and
LightGBM forecasting pipeline expect.
"""
import sqlite3
import pandas as pd
from db_helpers import derive_status
from datetime import date

CSV_PATH = "synthetic_rental_data.csv"
DB_PATH = "equipment_rental_synthetic.db"

RENTAL_RATES = {"Excavator": 550, "Crane": 750, "Bulldozer": 600, "Grader": 500}

df = pd.read_csv(CSV_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS rentals;
DROP TABLE IF EXISTS equipment;
DROP TABLE IF EXISTS EquipmentRental;
DROP TABLE IF EXISTS fuel_consumption;

CREATE TABLE EquipmentRental (
    EquipmentID TEXT PRIMARY KEY,
    Type TEXT,
    SiteID TEXT,
    CheckInDate DATE,
    CheckOutDate DATE,
    ExpectedReturnDate DATE,
    EngineHoursPerDay REAL,
    IdleHoursPerDay REAL,
    RentalDays INTEGER,
    LastOperatorID TEXT
);

CREATE TABLE equipment (
    equipment_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT DEFAULT 'available',
    daily_rental_rate REAL DEFAULT 500,
    age INTEGER
);

CREATE TABLE rentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT,
    operator_id TEXT,
    site_id TEXT,
    check_in_date DATE,
    expected_return_date DATE,
    actual_return_date DATE,
    rental_days INTEGER DEFAULT 0,
    is_returned INTEGER DEFAULT 0,
    condition_notes TEXT,
    engine_hours_per_day REAL DEFAULT 0,
    idle_hours_per_day REAL DEFAULT 0,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
);

CREATE TABLE fuel_consumption (
    equipment_type TEXT PRIMARY KEY,
    fuel_less_than_5 REAL,
    fuel_greater_than_or_equal_5 REAL
);

INSERT INTO fuel_consumption VALUES ('Bulldozer', 10.0, 14.0);
INSERT INTO fuel_consumption VALUES ('Crane', 7.0, 10.0);
INSERT INTO fuel_consumption VALUES ('Excavator', 15.0, 21.0);
INSERT INTO fuel_consumption VALUES ('Grader', 11.0, 15.0);
""")

import random
random.seed(42)

unique_eq = df.drop_duplicates(subset="EquipmentID")
for _, row in unique_eq.iterrows():
    eq_id = row["EquipmentID"]
    eq_type = row["Type"]
    rate = RENTAL_RATES.get(eq_type, 500)
    age = random.randint(1, 12)

    last_rental = df[df["EquipmentID"] == eq_id].iloc[-1]
    status = derive_status(
        last_rental["CheckInDate"],
        last_rental["CheckOutDate"] if pd.notna(last_rental["CheckOutDate"]) else None,
        last_rental["ExpectedReturnDate"],
        last_rental["LastOperatorID"] if pd.notna(last_rental["LastOperatorID"]) else None,
        last_rental["SiteID"] if pd.notna(last_rental["SiteID"]) else None,
    )

    cursor.execute(
        "INSERT INTO equipment VALUES (?, ?, ?, ?, ?)",
        (eq_id, eq_type, status, rate, age),
    )

today = date.today()
for _, row in df.iterrows():
    has_checkout = pd.notna(row["CheckOutDate"])
    is_returned = 1 if has_checkout else 0
    actual_return = row["CheckOutDate"] if has_checkout else None

    cursor.execute("""
        INSERT INTO rentals (
            equipment_id, operator_id, site_id, check_in_date,
            expected_return_date, actual_return_date, rental_days,
            is_returned, engine_hours_per_day, idle_hours_per_day
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["EquipmentID"],
        row["LastOperatorID"] if pd.notna(row["LastOperatorID"]) else None,
        row["SiteID"] if pd.notna(row["SiteID"]) else None,
        row["CheckInDate"],
        row["ExpectedReturnDate"],
        actual_return,
        int(row["RentalDays"]),
        is_returned,
        row["EngineHoursPerDay"],
        row["IdleHoursPerDay"],
    ))

last_rows = df.sort_values("CheckInDate").drop_duplicates(subset="EquipmentID", keep="last")
for _, row in last_rows.iterrows():
    cursor.execute("""
        INSERT OR REPLACE INTO EquipmentRental VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["EquipmentID"], row["Type"],
        row["SiteID"] if pd.notna(row["SiteID"]) else None,
        row["CheckInDate"],
        row["CheckOutDate"] if pd.notna(row["CheckOutDate"]) else None,
        row["ExpectedReturnDate"],
        row["EngineHoursPerDay"], row["IdleHoursPerDay"],
        int(row["RentalDays"]),
        row["LastOperatorID"] if pd.notna(row["LastOperatorID"]) else None,
    ))

conn.commit()

eq_count = cursor.execute("SELECT COUNT(*) FROM equipment").fetchone()[0]
rental_count = cursor.execute("SELECT COUNT(*) FROM rentals").fetchone()[0]
er_count = cursor.execute("SELECT COUNT(*) FROM EquipmentRental").fetchone()[0]
print(f"Loaded into {DB_PATH}:")
print(f"  equipment:        {eq_count} rows")
print(f"  rentals:          {rental_count} rows")
print(f"  EquipmentRental:  {er_count} rows")
print(f"  fuel_consumption: 4 rows")

conn.close()
