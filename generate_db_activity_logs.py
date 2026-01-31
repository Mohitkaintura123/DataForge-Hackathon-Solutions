import pandas as pd
import random
import os
from datetime import datetime, timedelta

random.seed(42)

users = [
    ("UserA", "Analyst"),
    ("UserB", "Engineer"),
    ("UserC", "Admin")
]

query_types = ["SELECT", "UPDATE", "DELETE"]
tables = ["customers", "transactions", "orders"]

rows = []
start_time = datetime(2024, 1, 1, 9, 0)

for _ in range(200):
    user, role = random.choice(users)
    query = random.choice(query_types)
    table = random.choice(tables)

    base = {"Analyst": 400, "Engineer": 900, "Admin": 1500}[role]
    rows_scanned = abs(int(random.gauss(base, base * 0.25)))
    duration_ms = rows_scanned * random.randint(1, 3)

    timestamp = start_time + timedelta(minutes=random.randint(0, 10000))

    rows.append([
        timestamp.strftime("%Y-%m-%d %H:%M"),
        user,
        role,
        query,
        table,
        rows_scanned,
        duration_ms
    ])


rows.extend([
    ["2024-01-15 02:30", "UserA", "Analyst", "SELECT", "transactions", 12000, 9800],
    ["2024-01-20 01:10", "UserB", "Engineer", "DELETE", "customers", 8000, 6500]
])

df = pd.DataFrame(rows, columns=[
    "timestamp",
    "user",
    "role",
    "query_type",
    "table",
    "rows_scanned",
    "duration_ms"
])

os.makedirs("output", exist_ok=True)

df.to_csv("output/db_activity_logs.csv", index=False)
print("PS-2 synthetic database logs generated.")
