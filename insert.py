import sqlite3
from datetime import date, datetime

conn = sqlite3.connect("equipment_rental.db")
cursor = conn.cursor()

# Drop derived tables, keep EquipmentRental as source of truth
cursor.execute("DROP TABLE IF EXISTS rentals")
cursor.execute("DROP TABLE IF EXISTS equipment")
cursor.execute("DROP TABLE IF EXISTS EquipmentRental")

# --- Original EquipmentRental table (source of truth) ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS EquipmentRental (
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
)
""")

equipment_data = [
    ('EQX1001', 'Excavator', 'S003', '2025-04-01', '2025-04-16', '2025-04-16', 1.5, 10, 15, 'OP101'),
    ('EQX1002', 'Crane', None, '2025-03-10', '2025-03-30', '2025-03-30', 0, 11, 20, None),
    ('EQX1003', 'Bulldozer', 'S002', '2025-02-15', '2025-03-11', '2025-03-12', 7.5, 0.5, 25, 'OP203'),
    ('EQX1004', 'Excavator', 'S004', '2025-05-05', '2025-05-15', '2025-05-15', 2, 9, 10, 'OP106'),
    ('EQX1005', 'Bulldozer', 'S006', '2025-01-01', '2025-01-31', '2025-01-31', 8, 0, 30, 'OP301'),
    ('EQX1006', 'Grader', 'S001', '2025-04-05', '2025-04-23', '2025-04-25', 3, 6, 18, 'OP114'),
    ('EQX1007', 'Excavator', None, '2025-03-20', '2025-04-01', '2025-04-01', 0, 12, 12, None),
    # Case 1: Still rented
    ('EQX1008', 'Excavator', 'S005', '2025-07-20', None, '2025-08-10', 6, 2, 21, 'OP108'),
    ('EQX1009', 'Crane', 'S003', '2025-07-25', None, '2025-08-15', 5, 3, 21, 'OP210'),
    # Case 2: Data error
    ('EQX1010', 'Bulldozer', 'S002', None, None, None, 0, 0, None, 'OP305'),
    ('EQX1011', 'Grader', None, None, None, None, 0, 0, None, None),
    # Case 3: Overdue
    ('EQX1012', 'Excavator', 'S004', '2025-05-10', None, '2025-05-20', 4, 5, 10, 'OP112'),
    ('EQX1013', 'Crane', 'S001', '2025-06-01', None, '2025-06-15', 3, 7, 14, 'OP215'),
    # Case 4: Suspicious
    ('EQX1014', 'Bulldozer', None, '2025-05-01', None, '2025-05-20', 0, 0, 19, None),
    ('EQX1015', 'Excavator', None, '2025-04-15', None, '2025-05-01', 0, 0, 16, None),
]

cursor.executemany("""
INSERT OR REPLACE INTO EquipmentRental (
    EquipmentID, Type, SiteID, CheckInDate, CheckOutDate, ExpectedReturnDate,
    EngineHoursPerDay, IdleHoursPerDay, RentalDays, LastOperatorID
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", equipment_data)

# --- Derive status from EquipmentRental ---
TODAY = date.today()


def derive_status(row):
    eq_id, typ, site, check_in, check_out, exp_return, op = row

    # Case: Returned
    if check_out is not None:
        return "available"

    # Case 2: Data error — both dates missing
    if check_in is None and check_out is None:
        return "unknown"

    # CheckIn exists, CheckOut is NULL from here

    # Case 4: Suspicious — no operator AND no site
    if op is None and site is None:
        return "flagged"

    # Case 3: Overdue — expected return has passed
    if exp_return is not None:
        exp_date = datetime.strptime(exp_return, "%Y-%m-%d").date()
        if exp_date < TODAY:
            return "overdue"

    # Case 1: Active rental
    return "rented"


# --- Create equipment table from unique entries in EquipmentRental ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT DEFAULT 'available',
    daily_rental_rate REAL DEFAULT 500
)
""")

rate_map = {"Excavator": 550, "Crane": 750, "Bulldozer": 600, "Grader": 500}

cursor.execute("""
    SELECT EquipmentID, Type, SiteID, CheckInDate, CheckOutDate,
           ExpectedReturnDate, LastOperatorID
    FROM EquipmentRental
""")
rows = cursor.fetchall()

for row in rows:
    eq_id, typ = row[0], row[1]
    status = derive_status(row)
    rate = rate_map.get(typ, 500)
    cursor.execute("""
        INSERT OR REPLACE INTO equipment (equipment_id, type, status, daily_rental_rate)
        VALUES (?, ?, ?, ?)
    """, (eq_id, typ, status, rate))

# --- Create rentals table from EquipmentRental ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS rentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT NOT NULL,
    operator_id TEXT,
    site_id TEXT,
    check_in_date TEXT,
    expected_return_date TEXT,
    actual_return_date TEXT,
    rental_days INTEGER DEFAULT 0,
    is_returned BOOLEAN DEFAULT 0,
    condition_notes TEXT,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
)
""")

cursor.execute("SELECT * FROM EquipmentRental")
rental_rows = cursor.fetchall()

for r in rental_rows:
    eq_id, typ, site, check_in, check_out, exp_return, eng, idle, days, op = r
    is_returned = 1 if check_out is not None else 0
    cursor.execute("""
        INSERT INTO rentals (
            equipment_id, operator_id, site_id, check_in_date,
            expected_return_date, actual_return_date, rental_days,
            is_returned, condition_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (eq_id, op, site, check_in, exp_return, check_out, days, is_returned, None))

conn.commit()

# --- Display ---
print("=" * 80)
print("SOURCE: EquipmentRental")
print("=" * 80)
cursor.execute("SELECT * FROM EquipmentRental")
print(f"{'ID':<10} {'Type':<12} {'Site':<6} {'CheckIn':<12} {'CheckOut':<12} {'ExpReturn':<12} {'Eng':<6} {'Idle':<6} {'Days':<6} {'Operator'}")
print("-" * 95)
for row in cursor.fetchall():
    print(f"{str(row[0]):<10} {str(row[1]):<12} {str(row[2]):<6} {str(row[3]):<12} {str(row[4]):<12} {str(row[5]):<12} {str(row[6]):<6} {str(row[7]):<6} {str(row[8]):<6} {str(row[9])}")

print()
print("=" * 60)
print("DERIVED: equipment (unique, with status)")
print("=" * 60)
cursor.execute("SELECT * FROM equipment")
print(f"{'ID':<10} {'Type':<12} {'Status':<12} {'Rate'}")
print("-" * 45)
for row in cursor.fetchall():
    print(f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]}")

print()
print("=" * 90)
print("DERIVED: rentals")
print("=" * 90)
cursor.execute("SELECT * FROM rentals")
print(f"{'#':<4} {'Equip':<10} {'Operator':<10} {'Site':<6} {'CheckIn':<12} {'ExpReturn':<12} {'ActReturn':<12} {'Days':<6} {'Ret?'}")
print("-" * 85)
for row in cursor.fetchall():
    print(f"{row[0]:<4} {str(row[1]):<10} {str(row[2]):<10} {str(row[3]):<6} {str(row[4]):<12} {str(row[5]):<12} {str(row[6]):<12} {str(row[7]):<6} {str(row[8])}")

conn.close()
