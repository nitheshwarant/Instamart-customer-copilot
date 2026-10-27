
"""
Part A — Data Cleaning
Clean customers, products and orders.
Save cleaned CSV files and create the SQLite database.
"""

from pathlib import Path
import sqlite3
import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
DB = Path("data/instamart.db")

CLEAN.mkdir(parents=True, exist_ok=True)

# STEP 1 — Load raw data
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")


# STEP 2 — Clean customers

# Standardize city names
customers["city"] = (
    customers["city"].str.strip().str.title()
)

# Standardize gender
customers["gender"] = (
    customers["gender"]
    .fillna("unknown")
    .str.strip()
    .str.lower()
    .replace({"m": "male", "f": "female"})
)

# Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Set invalid ages to missing
invalid_ages = int((
    customers["age"].notna()
    & ~customers["age"].between(10, 100)
).sum())

customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

missing_ages = int(customers["age"].isna().sum())

# Fill using median age within each city
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)


# STEP 3 — Clean orders

# Convert timestamp to datetime
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

# Remove exact duplicates
exact_duplicates = int(orders.duplicated().sum())
orders = orders.drop_duplicates().copy()

# Keep latest record for each order ID
orders = orders.sort_values(
    ["order_id", "order_ts"], kind="mergesort"
)

older_records = len(orders) - orders["order_id"].nunique()

orders = orders.drop_duplicates(
    subset="order_id", keep="last"
).copy()


# STEP 4 — Repair order amounts

price_by_id = products.set_index(
    "product_id"
)["price_inr"]

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

invalid_amounts = int(
    orders["order_amount"].isna().sum()
)

calculated_amount = (
    orders["product_id"].map(price_by_id)
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = (
    orders["order_amount"].fillna(calculated_amount)
)


# STEP 5 — Repair delivery times

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

negative_delivery = int(
    (orders["delivery_minutes"] < 0).sum()
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Associate each order with its product category
category_by_id = products.set_index(
    "product_id"
)["category"]

orders["product_category"] = (
    orders["product_id"].map(category_by_id)
)

completed = orders["status"].eq("completed")

missing_completed_before = int(
    orders.loc[completed, "delivery_minutes"].isna().sum()
)

# Median computed from valid completed orders
category_median = (
    orders.loc[completed]
    .groupby("product_category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(category_median)
)

# Cancelled orders must have no delivery time
orders.loc[
    orders["status"].eq("cancelled"),
    "delivery_minutes"
] = float("nan")

orders = orders.drop(columns="product_category")


# STEP 6 — Save cleaned CSV files

customers["signup_date"] = (
    customers["signup_date"].dt.strftime("%Y-%m-%d")
)

orders["order_ts"] = (
    orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
)

for filename, df in [
    ("customers", customers),
    ("products", products),
    ("orders", orders)
]:
    df.to_csv(
        CLEAN / f"{filename}.csv", index=False
    )


# STEP 7 — Store in SQLite

with sqlite3.connect(DB) as conn:
    customers.to_sql(
        "customers", conn,
        if_exists="replace", index=False
    )
    products.to_sql(
        "products", conn,
        if_exists="replace", index=False
    )
    orders.to_sql(
        "orders", conn,
        if_exists="replace", index=False
    )


# STEP 8 — Print results

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

print("CUSTOMERS:", len(customers))
print("CITIES:", customers["city"].nunique())
print("INVALID AGES REPLACED:", invalid_ages)
print("TOTAL AGES FILLED:", missing_ages)
print("MISSING AGES AFTER:", customers["age"].isna().sum())
print("GENDERS:", customers["gender"].value_counts().to_dict())
print("PRODUCTS:", len(products))
print("EXACT DUPLICATES REMOVED:", exact_duplicates)
print("OLDER ORDER RECORDS REMOVED:", older_records)
print("INVALID AMOUNTS REBUILT:", invalid_amounts)
print("NEGATIVE DELIVERY TIMES CLEARED:", negative_delivery)
print("COMPLETED DELIVERY TIMES FILLED:", missing_completed_before)

print("CLEAN ORDERS:", len(orders))
print("COMPLETED ORDERS:", int(completed.sum()))
print("CANCELLED ORDERS:", int(cancelled.sum()))
print("PENDING ORDERS:", int(orders["status"].eq("pending").sum()))

print(
    "MISSING COMPLETED DELIVERY:",
    int(orders.loc[completed, "delivery_minutes"].isna().sum())
)
print(
    "MISSING CANCELLED DELIVERY:",
    int(orders.loc[cancelled, "delivery_minutes"].isna().sum())
)
print(
    "MISSING AMOUNTS:",
    int(orders["order_amount"].isna().sum())
)

revenue = orders.loc[completed, "order_amount"].sum()

print("COMPLETED REVENUE (INR):", f"{revenue:,.2f}")
print("Saved cleaned CSVs and data/instamart.db")
