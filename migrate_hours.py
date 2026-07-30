import sqlite3

conn = sqlite3.connect("equipment_rental.db")
cursor = conn.cursor()

# Add columns to rentals table
cursor.execute("ALTER TABLE rentals ADD COLUMN engine_hours_per_day REAL DEFAULT 0")
cursor.execute("ALTER TABLE rentals ADD COLUMN idle_hours_per_day REAL DEFAULT 0")

# Copy values from EquipmentRental into matching rentals rows
cursor.execute("""
    UPDATE rentals
    SET engine_hours_per_day = (
            SELECT er.EngineHoursPerDay FROM EquipmentRental er
            WHERE er.EquipmentID = rentals.equipment_id
        ),
        idle_hours_per_day = (
            SELECT er.IdleHoursPerDay FROM EquipmentRental er
            WHERE er.EquipmentID = rentals.equipment_id
        )
""")

conn.commit()

# Verify
cursor.execute("SELECT equipment_id, engine_hours_per_day, idle_hours_per_day FROM rentals")
print(f"{'Equipment':<12} {'Engine Hrs':<12} {'Idle Hrs'}")
print("-" * 35)
for row in cursor.fetchall():
    print(f"{row[0]:<12} {row[1]:<12} {row[2]}")

conn.close()
