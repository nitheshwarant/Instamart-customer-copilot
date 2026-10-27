
"""
Part A - Data Auditing
Instamart Customer Intelligence & Support Copilot

This script checks:
1. Dataset shapes
2. Column data types
3. Missing values and percentages
4. Duplicate orders
5. Invalid order amounts and delivery times
6. Invalid customer ages
7. Inconsistent city and gender values
"""

import pandas as pd


# --------------------------------------------------
# STEP 1: LOAD RAW DATA
# --------------------------------------------------

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")


# --------------------------------------------------
# STEP 2: GENERAL DATA AUDIT
# --------------------------------------------------

def audit_table(name, df):

    print(f"\n========== {name} ==========")

    # Total number of rows and columns
    print("Shape:", df.shape)

    # Check data types of all columns
    print("\nData Types:")
    print(df.dtypes)

    # Count missing values in each column
    missing_count = df.isna().sum()

    # Calculate missing percentage
    missing_percentage = (
        df.isna().mean() * 100
    ).round(2)

    # Create audit summary
    report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage
    })

    print("\nMissing Value Report:")
    print(report)


# Run audit for all three datasets
audit_table("CUSTOMERS", customers)
audit_table("PRODUCTS", products)
audit_table("ORDERS", orders)


# --------------------------------------------------
# STEP 3: ORDER DATA QUALITY CHECKS
# --------------------------------------------------

print("\n========== ORDER QUALITY ==========")

# Count all rows
print("Raw order rows:", len(orders))

# Find exact duplicate rows
print(
    "Exact duplicates:",
    orders.duplicated().sum()
)

# Count pending orders
print(
    "Pending orders:",
    (orders["status"] == "pending").sum()
)

# Count unique order IDs
print(
    "Unique order IDs:",
    orders["order_id"].nunique()
)

# Convert order amounts to numeric
# Invalid strings become NaN
amounts = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

print(
    "Non-numeric amounts:",
    amounts.isna().sum()
)

# Convert delivery time to numeric
delivery = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Count negative delivery durations
print(
    "Negative delivery times:",
    (delivery < 0).sum()
)

# Count missing delivery durations
print(
    "Blank delivery times:",
    delivery.isna().sum()
)


# --------------------------------------------------
# STEP 4: CUSTOMER DATA QUALITY CHECKS
# --------------------------------------------------

print("\n========== CUSTOMER QUALITY ==========")

# Convert age to numeric
ages = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

# Missing ages
print(
    "Blank ages:",
    ages.isna().sum()
)

# Invalid ages outside 10 to 100
invalid_ages = (
    ages.notna() & ~ages.between(10, 100)
)

print(
    "Invalid ages:",
    invalid_ages.sum()
)

# Different raw city spellings
print(
    "Raw city spellings:",
    customers["city"].nunique()
)

# Different gender spellings, excluding blanks
print(
    "Raw gender spellings:",
    customers["gender"].dropna().nunique()
)

# Missing gender values
print(
    "Blank genders:",
    customers["gender"].isna().sum()
)


# --------------------------------------------------
# STEP 5: REFERENCE CHECKPOINTS
# --------------------------------------------------

print("\n========== CHECKPOINT VALIDATION ==========")

checks = {
    "Raw orders": (len(orders), 13084),
    "Exact duplicates": (
        int(orders.duplicated().sum()), 119
    ),
    "Pending orders": (
        int((orders["status"] == "pending").sum()), 656
    ),
    "Unique order IDs": (
        orders["order_id"].nunique(), 12309
    ),
    "Invalid amounts": (
        int(amounts.isna().sum()), 120
    ),
    "Negative delivery": (
        int((delivery < 0).sum()), 106
    ),
    "Blank delivery": (
        int(delivery.isna().sum()), 2541
    ),
    "Blank ages": (
        int(ages.isna().sum()), 61
    ),
    "Invalid ages": (
        int(invalid_ages.sum()), 6
    ),
    "Raw city spellings": (
        customers["city"].nunique(), 24
    ),
    "Raw gender spellings": (
        customers["gender"].dropna().nunique(), 10
    ),
    "Blank genders": (
        int(customers["gender"].isna().sum()), 30
    )
}

all_passed = True

for name, (actual, expected) in checks.items():
    passed = actual == expected

    if not passed:
        all_passed = False

    status = "PASS" if passed else "FAIL"

    print(
        f"{name}: {actual} "
        f"(expected {expected}) - {status}"
    )

if all_passed:
    print("\nRAW DATA AUDIT PASSED")
else:
    print("\nRAW DATA AUDIT FAILED")
