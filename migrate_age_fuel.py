import sqlite3
import random

conn = sqlite3.connect("equipment_rental.db")
cursor = conn.cursor()

# # Add age column to equipment table
# try:
#     cursor.execute("ALTER TABLE equipment ADD COLUMN age INTEGER")
# except sqlite3.OperationalError:
#     pass  # column already exists

# # Assign random ages <= 10 to all equipment
# cursor.execute("SELECT equipment_id FROM equipment")
# for (eq_id,) in cursor.fetchall():
#     age = random.randint(1, 10)
#     cursor.execute("UPDATE equipment SET age = ? WHERE equipment_id = ?", (age, eq_id))

# Create fuel_consumption table
cursor.execute("DROP TABLE IF EXISTS fuel_consumption")
cursor.execute("""
CREATE TABLE fuel_consumption (
    equipment_type TEXT PRIMARY KEY,
    fuel_less_than_5 REAL,
    fuel_greater_than_or_equal_5 REAL
)
""")

fuel_data = [
    ('Bulldozer', 10, 14),
    ('Crane', 7, 10),
    ('Excavator', 15, 21),
    ('Grader', 11, 15),
]

cursor.executemany("""
    INSERT INTO fuel_consumption (equipment_type, fuel_less_than_5, fuel_greater_than_or_equal_5)
    VALUES (?, ?, ?)
""", fuel_data)

conn.commit()

# Verify
print("Equipment with ages:")
for r in cursor.execute("SELECT equipment_id, type, age FROM equipment"):
    print(r)

print("\nFuel consumption by type:")
for r in cursor.execute("SELECT * FROM fuel_consumption"):
    print(r)

conn.close()
