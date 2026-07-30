import sqlite3

conn = sqlite3.connect("equipment_rental.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS EquipmentRental (
    EquipmentID TEXT PRIMARY KEY,
    Type TEXT,
    SiteID TEXT,
    CheckInDate DATE,
    CheckOutDate DATE,
    EngineHoursPerDay REAL,
    IdleHoursPerDay REAL,
    RentalDays INTEGER,
    LastOperatorID TEXT
)
""")

# Sample data
equipment_data = [
    ('EQX1001', 'Excavator', 'S003', '2025-04-01', '2025-04-16', 1.5, 10, 15, 'OP101'),
    ('EQX1002', 'Crane', None, '2025-03-10', '2025-03-30', 0, 11, 20, None),
    ('EQX1003', 'Bulldozer', 'S002', '2025-02-15', '2025-03-11', 7.5, 0.5, 25, 'OP203'),
    ('EQX1004', 'Excavator', 'S004', '2025-05-05', '2025-05-15', 2, 9, 10, 'OP106'),
    ('EQX1005', 'Bulldozer', 'S006', '2025-01-01', '2025-01-31', 8, 0, 30, 'OP301'),
    ('EQX1006', 'Grader', 'S001', '2025-04-05', '2025-04-23', 3, 6, 18, 'OP114'),
    ('EQX1007', 'Excavator', None, '2025-03-20', '2025-04-01', 0, 12, 12, None)
]

# Insert the data
cursor.executemany("""
INSERT OR REPLACE INTO EquipmentRental (
    EquipmentID,
    Type,
    SiteID,
    CheckInDate,
    CheckOutDate,
    EngineHoursPerDay,
    IdleHoursPerDay,
    RentalDays,
    LastOperatorID
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", equipment_data)

# Commit the changes
conn.commit()

# Retrieve and display all records
cursor.execute("SELECT * FROM EquipmentRental")
rows = cursor.fetchall()

print("EquipmentRental Table:")
for row in rows:
    print(row)

# Close the connection
conn.close()