import sqlite3
import pandas as pd

conn = sqlite3.connect("equipment_rental.db")

df = pd.read_sql_query("SELECT * FROM rentals", conn)

print(df.to_string(index=False))

conn.close()