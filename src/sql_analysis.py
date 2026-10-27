"""Part A: Run all seven SQL queries and cross-check completed-order revenue with pandas."""

from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")
EXPECTED_REVENUE = 2998668.80


def run_queries(conn):
    # Each of the seven SQL statements ends with a semicolon.
    statements = [q.strip() for q in SQL_PATH.read_text(encoding="utf-8").split(";") if q.strip()]
    assert len(statements) == 7, f"Expected 7 queries; found {len(statements)}"
    results = {}
    for i, sql in enumerate(statements, start=1):
        df = pd.read_sql_query(sql, conn)
        results[i] = df
        print(f"\n===== Q{i} — {len(df)} rows =====")
        if i == 2:
            print(df.to_string(index=False))
        elif i == 6:
            print(df[df["city_rank"] == 1].to_string(index=False))
        elif i == 7:
            print(df.tail(5).to_string(index=False))
        else:
            print(df.head(10).to_string(index=False))

    q1, q2, q3, q4, q5, q6, q7 = (results[k] for k in range(1, 8))
    assert len(q1) == 30
    assert len(q2) == 30
    assert abs(q2["revenue"].sum() - EXPECTED_REVENUE) < 0.01
    largest_cell = q2.loc[q2["revenue"].idxmax()]
    assert (largest_cell["city"], largest_cell["category"]) == ("Bengaluru", "Electronics")
    assert abs(largest_cell["revenue"] - 233909.45) < 0.01
    assert len(q3) == 11
    assert len(q4) == 25
    assert len(q5) == 550
    assert abs(q5["avg_customer_spend"].iloc[0] - 2552.06) < 0.01

    expected_top = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }
    for city, (cust, spend) in expected_top.items():
        top = q6[(q6["city"] == city) & (q6["city_rank"] == 1)]
        assert len(top) == 1
        assert top.iloc[0]["customer_id"] == cust
        assert abs(top.iloc[0]["total_spend"] - spend) < 0.01
    different_rank_rows = int((q6["city_rank"] != q6["row_number"]).sum())
    assert different_rank_rows == 8, f"Unexpected rank difference count: {different_rank_rows}"
    assert len(q7) == 183
    assert abs(q7["cumulative_revenue"].iloc[-1] - EXPECTED_REVENUE) < 0.01

    print("\n===== SQL checkpoint summary =====")
    print(f"Q1 high-spending customers: {len(q1)}")
    print(f"Q2 city-category revenue rows: {len(q2)}; total INR {q2['revenue'].sum():,.2f}")
    print(f"Q2 largest cell: {largest_cell['city']} / {largest_cell['category']}: INR {largest_cell['revenue']:,.2f}")
    print(f"Q3 products above average catalogue price: {len(q3)}")
    print(f"Q4 customers with no completed order: {len(q4)}")
    print(f"Q5 above-average completed spenders: {len(q5)}, average INR {q5['avg_customer_spend'].iloc[0]:,.2f}")
    print(f"Q6 rows with RANK() different from ROW_NUMBER(): {different_rank_rows}")
    print(f"Q7 days: {len(q7)}, final cumulative revenue INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")
    print("ALL SEVEN SQL CHECKPOINTS PASSED")
    return results


def pandas_analysis(conn, sql_results):
    # Load tables from SQLite, keeping only completed orders for monetary analysis.
    orders = pd.read_sql_query("SELECT * FROM orders WHERE status = 'completed'", conn)
    customers = pd.read_sql_query("SELECT customer_id, city FROM customers", conn)
    products = pd.read_sql_query("SELECT product_id, category FROM products", conn)

    joined = orders.merge(customers, on="customer_id").merge(products, on="product_id")
    pivot = pd.pivot_table(
        joined,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    )
    print("\n===== PANDAS PIVOT: Revenue by city and category =====")
    print(pivot.round(2).to_string())
    pivot_total = float(pivot.loc["All", "All"])
    sql_total = float(sql_results[2]["revenue"].sum())
    assert abs(pivot_total - sql_total) < 0.01
    print(f"Pandas and SQL totals match: INR {pivot_total:,.2f}")

    by_city = joined.groupby("city").agg(
        revenue_sum=("order_amount", "sum"),
        mean_order_amount=("order_amount", "mean"),
        order_count=("order_amount", "count"),
    ).sort_index()
    print("\n===== COMPLETED ORDERS BY CITY =====")
    print(by_city.round(2).to_string())

    amount = orders["order_amount"]
    q1, median, q3 = amount.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    low_outliers = int((amount < lower_fence).sum())
    high_outliers = int((amount > upper_fence).sum())
    print("\n===== EXPLORATORY STATISTICS =====")
    print(f"Mean: {amount.mean():.2f}; median: {median:.2f}; skewness: {amount.skew():.3f}")
    print(f"Q1: {q1:.2f}; Q3: {q3:.2f}; IQR: {iqr:.2f}")
    print(f"Lower fence: {lower_fence:.2f}; upper fence: {upper_fence:.2f}")
    print(f"Outliers below lower fence: {low_outliers}; above upper fence: {high_outliers}")

    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = orders[corr_cols].corr()
    print("\n===== CORRELATION MATRIX =====")
    print(corr.round(3).to_string())
    pairs = [
        (corr.loc[a, b], a, b)
        for i, a in enumerate(corr_cols)
        for b in corr_cols[i + 1:]
    ]
    strongest = max(pairs, key=lambda pair: abs(pair[0]))
    print(f"Strongest absolute correlation: {strongest[1]} with {strongest[2]} = {strongest[0]:.3f}")
    print("These weak correlations do not establish causation.")


if __name__ == "__main__":
    if not DB_PATH.is_file():
        raise FileNotFoundError("Database missing. Run python src/clean_data.py first.")
    if not SQL_PATH.is_file():
        raise FileNotFoundError("Missing sql/queries.sql")
    with sqlite3.connect(DB_PATH) as connection:
        sql_results = run_queries(connection)
        pandas_analysis(connection, sql_results)
