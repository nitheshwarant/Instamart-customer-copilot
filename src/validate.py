
"""
Part A — Data Validation
Validates cleaned customer, product and order data.
"""

import sqlite3
import math
import pandas as pd

# ---------------------------------------
# STEP 1: Load cleaned datasets
# ---------------------------------------

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# ---------------------------------------
# STEP 2: Validate customers
# ---------------------------------------

# Total customers
assert len(customers) == 1200

# Exactly six cities
assert customers["city"].nunique() == 6

# Expected city distribution
expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184
}

assert customers["city"].value_counts().to_dict() == expected_cities

# No missing ages
assert customers["age"].isna().sum() == 0

# All ages within the expected range
assert customers["age"].between(18, 60).all()

# Expected gender counts
expected_genders = {
    "female": 610,
    "male": 560,
    "unknown": 30
}

assert customers["gender"].value_counts().to_dict() == expected_genders

# ---------------------------------------
# STEP 3: Validate products
# ---------------------------------------

assert len(products) == 40

assert products["product_id"].is_unique

# ---------------------------------------
# STEP 4: Validate orders
# ---------------------------------------

# Exactly one record per order ID
assert len(orders) == 12309
assert orders["order_id"].is_unique

# Status counts
completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

assert completed.sum() == 11110
assert cancelled.sum() == 1199

# No pending records
assert (orders["status"] == "pending").sum() == 0

# All order amounts are valid
assert orders["order_amount"].notna().all()

# No missing delivery time for completed orders
assert orders.loc[
    completed, "delivery_minutes"
].notna().all()

# Cancelled orders must have no delivery time
assert orders.loc[
    cancelled, "delivery_minutes"
].isna().sum() == 1199

# No negative delivery times
assert (
    orders["delivery_minutes"].dropna() >= 0
).all()

# Completed order revenue checkpoint
revenue = orders.loc[
    completed, "order_amount"
].sum()

assert math.isclose(
    revenue,
    2998668.80,
    rel_tol=0,
    abs_tol=0.01
)

# ---------------------------------------
# STEP 5: Validate SQLite database
# ---------------------------------------

conn = sqlite3.connect("data/instamart.db")

try:
    # Verify required tables exist
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    assert {"customers", "products", "orders"}.issubset(tables)

    # Verify database record counts
    assert conn.execute(
        "SELECT COUNT(*) FROM customers"
    ).fetchone()[0] == 1200

    assert conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0] == 40

    assert conn.execute(
        "SELECT COUNT(*) FROM orders"
    ).fetchone()[0] == 12309

    # Verify database revenue
    db_revenue = conn.execute(
        """
        SELECT SUM(order_amount)
        FROM orders
        WHERE status = 'completed'
        """
    ).fetchone()[0]

    assert math.isclose(
        db_revenue,
        2998668.80,
        rel_tol=0,
        abs_tol=0.01
    )

finally:
    conn.close()

# ---------------------------------------
# STEP 6: Final result
# ---------------------------------------

print("ALL CHECKPOINTS PASSED")
