import sqlite3

conn = sqlite3.connect("equipment_rental.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS rentals")
cursor.execute("DROP TABLE IF EXISTS equipment")
cursor.execute("DROP TABLE IF EXISTS EquipmentRental")

# Master data — what the machine IS
cursor.execute("""
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT DEFAULT 'available',
    daily_rental_rate REAL DEFAULT 500
)
""")

# Rental transactions — who has it, when
cursor.execute("""
CREATE TABLE IF NOT EXISTS rentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT NOT NULL,
    operator_id TEXT,
    site_id TEXT,
    check_in_date TEXT NOT NULL,
    expected_return_date TEXT,
    actual_return_date TEXT,
    rental_days INTEGER DEFAULT 0,
    is_returned BOOLEAN DEFAULT 0,
    condition_notes TEXT,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
)
""")

# --- Equipment master data ---
equipment_data = [
    ('EQX1001', 'Excavator', 'available', 550),
    ('EQX1002', 'Crane', 'available', 750),
    ('EQX1003', 'Bulldozer', 'available', 600),
    ('EQX1004', 'Excavator', 'available', 550),
    ('EQX1005', 'Bulldozer', 'available', 600),
    ('EQX1006', 'Grader', 'available', 500),
    ('EQX1007', 'Excavator', 'available', 550),
    ('EQX1008', 'Excavator', 'rented', 550),
    ('EQX1009', 'Crane', 'rented', 750),
    ('EQX1010', 'Bulldozer', 'unknown', 600),
    ('EQX1011', 'Grader', 'unknown', 500),
    ('EQX1012', 'Excavator', 'overdue', 550),
    ('EQX1013', 'Crane', 'overdue', 750),
    ('EQX1014', 'Bulldozer', 'flagged', 600),
    ('EQX1015', 'Excavator', 'flagged', 550),
]

cursor.executemany("""
INSERT OR REPLACE INTO equipment (equipment_id, type, status, daily_rental_rate)
VALUES (?, ?, ?, ?)
""", equipment_data)

# --- Rental transactions (split from the old single table) ---
# (equipment_id, operator_id, site_id, check_in_date, expected_return_date,
#  actual_return_date, rental_days, is_returned, condition_notes)
rental_data = [
    # Returned equipment
    ('EQX1001', 'OP101', 'S003', '2025-04-01', '2025-04-16', '2025-04-16', 15, 1, 'Good condition'),
    ('EQX1002', None, None, '2025-03-10', '2025-03-30', '2025-03-30', 20, 1, None),
    ('EQX1003', 'OP203', 'S002', '2025-02-15', '2025-03-12', '2025-03-11', 25, 1, 'Minor wear on tracks'),
    ('EQX1004', 'OP106', 'S004', '2025-05-05', '2025-05-15', '2025-05-15', 10, 1, 'Good condition'),
    ('EQX1005', 'OP301', 'S006', '2025-01-01', '2025-01-31', '2025-01-31', 30, 1, 'Blade needs sharpening'),
    ('EQX1006', 'OP114', 'S001', '2025-04-05', '2025-04-25', '2025-04-23', 18, 1, 'Returned early'),
    ('EQX1007', None, None, '2025-03-20', '2025-04-01', '2025-04-01', 12, 1, None),

    # Case 1: Active rental — still with customer
    ('EQX1008', 'OP108', 'S005', '2025-07-20', '2025-08-10', None, 21, 0, None),
    ('EQX1009', 'OP210', 'S003', '2025-07-25', '2025-08-15', None, 21, 0, None),

    # Case 2: Data error — missing check-in (use placeholder)
    ('EQX1010', 'OP305', 'S002', '1900-01-01', None, None, 0, 0, 'DATA ERROR: check-in date missing'),
    ('EQX1011', None, None, '1900-01-01', None, None, 0, 0, 'DATA ERROR: check-in and operator missing'),

    # Case 3: Overdue — expected return already passed
    ('EQX1012', 'OP112', 'S004', '2025-05-10', '2025-05-20', None, 10, 0, None),
    ('EQX1013', 'OP215', 'S001', '2025-06-01', '2025-06-15', None, 14, 0, None),

    # Case 4: Suspicious — no operator, no site
    ('EQX1014', None, None, '2025-05-01', '2025-05-20', None, 19, 0, 'No operator or site on record'),
    ('EQX1015', None, None, '2025-04-15', '2025-05-01', None, 16, 0, 'No operator or site on record'),
]

cursor.executemany("""
INSERT INTO rentals (
    equipment_id, operator_id, site_id, check_in_date,
    expected_return_date, actual_return_date, rental_days,
    is_returned, condition_notes
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", rental_data)

conn.commit()

# --- Display ---
print("=" * 60)
print("EQUIPMENT TABLE")
print("=" * 60)
cursor.execute("SELECT * FROM equipment")
print(f"{'ID':<10} {'Type':<12} {'Status':<12} {'Rate/Day'}")
print("-" * 45)
for row in cursor.fetchall():
    print(f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]}")

print()
print("=" * 90)
print("RENTALS TABLE")
print("=" * 90)
cursor.execute("SELECT * FROM rentals")
print(f"{'#':<4} {'Equip':<10} {'Operator':<10} {'Site':<6} {'CheckIn':<12} {'ExpReturn':<12} {'ActReturn':<12} {'Days':<6} {'Ret?':<5} {'Notes'}")
print("-" * 100)
for row in cursor.fetchall():
    print(f"{row[0]:<4} {str(row[1]):<10} {str(row[2]):<10} {str(row[3]):<6} {str(row[4]):<12} {str(row[5]):<12} {str(row[6]):<12} {str(row[7]):<6} {str(row[8]):<5} {str(row[9])}")

conn.close()
