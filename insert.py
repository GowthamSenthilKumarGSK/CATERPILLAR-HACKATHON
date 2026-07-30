import sqlite3

conn = sqlite3.connect("equipment_rental.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS EquipmentRental")

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

# Original data (unchanged, ExpectedReturnDate set to match CheckOutDate since they were returned on time)
equipment_data = [
    ('EQX1001', 'Excavator', 'S003', '2025-04-01', '2025-04-16', '2025-04-16', 1.5, 10, 15, 'OP101'),
    ('EQX1002', 'Crane', None, '2025-03-10', '2025-03-30', '2025-03-30', 0, 11, 20, None),
    ('EQX1003', 'Bulldozer', 'S002', '2025-02-15', '2025-03-11', '2025-03-12', 7.5, 0.5, 25, 'OP203'),
    ('EQX1004', 'Excavator', 'S004', '2025-05-05', '2025-05-15', '2025-05-15', 2, 9, 10, 'OP106'),
    ('EQX1005', 'Bulldozer', 'S006', '2025-01-01', '2025-01-31', '2025-01-31', 8, 0, 30, 'OP301'),
    ('EQX1006', 'Grader', 'S001', '2025-04-05', '2025-04-23', '2025-04-25', 3, 6, 18, 'OP114'),
    ('EQX1007', 'Excavator', None, '2025-03-20', '2025-04-01', '2025-04-01', 0, 12, 12, None),
]

# Case 1: Still rented (valid) - CheckOut is NULL, recently checked in
new_data = [
    ('EQX1008', 'Excavator', 'S005', '2025-07-20', None, '2025-08-10', 6, 2, 21, 'OP108'),
    ('EQX1009', 'Crane', 'S003', '2025-07-25', None, '2025-08-15', 5, 3, 21, 'OP210'),

    # Case 2: Data error - both CheckIn and CheckOut are NULL
    ('EQX1010', 'Bulldozer', 'S002', None, None, None, 0, 0, None, 'OP305'),
    ('EQX1011', 'Grader', None, None, None, None, 0, 0, None, None),

    # Case 3: Overdue rental - CheckIn exists, CheckOut NULL, ExpectedReturnDate has passed
    ('EQX1012', 'Excavator', 'S004', '2025-05-10', None, '2025-05-20', 4, 5, 10, 'OP112'),
    ('EQX1013', 'Crane', 'S001', '2025-06-01', None, '2025-06-15', 3, 7, 14, 'OP215'),

    # Case 4: Suspicious record - CheckIn exists, CheckOut NULL, Operator NULL, Site NULL
    ('EQX1014', 'Bulldozer', None, '2025-05-01', None, '2025-05-20', 0, 0, 19, None),
    ('EQX1015', 'Excavator', None, '2025-04-15', None, '2025-05-01', 0, 0, 16, None),
]

all_data = equipment_data + new_data

cursor.executemany("""
INSERT OR REPLACE INTO EquipmentRental (
    EquipmentID, Type, SiteID,
    CheckInDate, CheckOutDate, ExpectedReturnDate,
    EngineHoursPerDay, IdleHoursPerDay, RentalDays, LastOperatorID
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", all_data)

conn.commit()

cursor.execute("SELECT * FROM EquipmentRental")
rows = cursor.fetchall()

print("EquipmentRental Table:")
print(f"{'ID':<10} {'Type':<12} {'Site':<6} {'CheckIn':<12} {'CheckOut':<12} {'ExpReturn':<12} {'Eng/Day':<8} {'Idle/Day':<9} {'Days':<6} {'Operator'}")
print("-" * 105)
for row in rows:
    print(f"{str(row[0]):<10} {str(row[1]):<12} {str(row[2]):<6} {str(row[3]):<12} {str(row[4]):<12} {str(row[5]):<12} {str(row[6]):<8} {str(row[7]):<9} {str(row[8]):<6} {str(row[9])}")

conn.close()
