import random
import csv
from datetime import date, timedelta

random.seed(42)

EQUIPMENT_TYPES = {
    "Excavator": {"rate": 550, "engine_hrs": (4, 10), "idle_hrs": (0.5, 3), "rental_days": (5, 40)},
    "Crane":     {"rate": 750, "engine_hrs": (3, 8),  "idle_hrs": (0.5, 4), "rental_days": (7, 45)},
    "Bulldozer": {"rate": 600, "engine_hrs": (5, 11), "idle_hrs": (0.5, 2.5), "rental_days": (5, 35)},
    "Grader":    {"rate": 500, "engine_hrs": (3, 9),  "idle_hrs": (0.5, 3), "rental_days": (4, 30)},
}

SITES = [f"S{str(i).zfill(3)}" for i in range(1, 16)]
OPERATORS = [f"OP{i}" for i in range(101, 131)]

NUM_EQUIPMENT = 50
START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 7, 15)

# Seasonal weight: higher demand in Mar-Jun and Sep-Nov
MONTH_WEIGHTS = {
    1: 0.6, 2: 0.7, 3: 1.2, 4: 1.4, 5: 1.3, 6: 1.1,
    7: 0.7, 8: 0.8, 9: 1.1, 10: 1.3, 11: 1.0, 12: 0.5,
}

equipment_pool = []
type_names = list(EQUIPMENT_TYPES.keys())
for i in range(1, NUM_EQUIPMENT + 1):
    eq_id = f"EQX{1000 + i}"
    eq_type = type_names[(i - 1) % len(type_names)]
    age = random.randint(1, 12)
    equipment_pool.append({"id": eq_id, "type": eq_type, "age": age})

rows = []
today = date.today()

for eq in equipment_pool:
    cfg = EQUIPMENT_TYPES[eq["type"]]
    cursor = START_DATE + timedelta(days=random.randint(0, 30))

    while cursor < END_DATE:
        month_weight = MONTH_WEIGHTS.get(cursor.month, 1.0)
        if random.random() > month_weight * 0.7:
            cursor += timedelta(days=random.randint(10, 40))
            continue

        check_in = cursor
        min_days, max_days = cfg["rental_days"]
        rental_days = random.randint(min_days, max_days)
        expected_return = check_in + timedelta(days=rental_days)

        is_completed = expected_return < today
        overdue_chance = random.random()

        if is_completed:
            if overdue_chance < 0.12:
                actual_return_offset = random.randint(1, 10)
                check_out = expected_return + timedelta(days=actual_return_offset)
                rental_days = (check_out - check_in).days
            else:
                check_out = expected_return - timedelta(days=random.randint(0, 2))
                if check_out <= check_in:
                    check_out = expected_return
                rental_days = (check_out - check_in).days
        else:
            check_out = None

        site = random.choice(SITES)
        operator = random.choice(OPERATORS)

        eng_lo, eng_hi = cfg["engine_hrs"]
        idle_lo, idle_hi = cfg["idle_hrs"]
        engine_hrs = round(random.uniform(eng_lo, eng_hi), 1)
        idle_hrs = round(random.uniform(idle_lo, idle_hi), 1)

        # Inject anomalies (~5% of rows)
        anomaly_roll = random.random()
        if anomaly_roll < 0.015:
            operator = None
            site = None
        elif anomaly_roll < 0.03:
            engine_hrs = 0.0
        elif anomaly_roll < 0.045:
            idle_hrs = round(random.uniform(8, 14), 1)
            engine_hrs = round(random.uniform(0.5, 2), 1)
        elif anomaly_roll < 0.055:
            check_in, expected_return = expected_return, check_in
            if check_out and check_out < check_in:
                check_out = check_in + timedelta(days=rental_days)

        rows.append({
            "EquipmentID": eq["id"],
            "Type": eq["type"],
            "SiteID": site,
            "CheckInDate": check_in.strftime("%Y-%m-%d"),
            "CheckOutDate": check_out.strftime("%Y-%m-%d") if check_out else None,
            "ExpectedReturnDate": expected_return.strftime("%Y-%m-%d"),
            "EngineHoursPerDay": engine_hrs,
            "IdleHoursPerDay": idle_hrs,
            "RentalDays": rental_days,
            "LastOperatorID": operator,
        })

        gap = random.randint(3, 25)
        if check_out:
            cursor = check_out + timedelta(days=gap)
        else:
            cursor = expected_return + timedelta(days=gap)

OUTPUT_FILE = "synthetic_rental_data.csv"
COLUMNS = [
    "EquipmentID", "Type", "SiteID", "CheckInDate", "CheckOutDate",
    "ExpectedReturnDate", "EngineHoursPerDay", "IdleHoursPerDay",
    "RentalDays", "LastOperatorID",
]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

types_count = {}
for r in rows:
    types_count[r["Type"]] = types_count.get(r["Type"], 0) + 1

print(f"Generated {len(rows)} rental records -> {OUTPUT_FILE}")
print(f"Equipment IDs: EQX1001 - EQX{1000 + NUM_EQUIPMENT}")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"By type: {types_count}")
anomaly_count = sum(1 for r in rows if not r["LastOperatorID"] or r["EngineHoursPerDay"] == 0)
print(f"Injected anomalies: ~{anomaly_count}")
