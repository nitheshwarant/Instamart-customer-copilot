
"""
Part B - Customer Segmentation
Task 1: RFM Analysis

R = Recency
F = Frequency
M = Monetary
"""

import pandas as pd

# ---------------------------------------
# STEP 1: Load cleaned order data
# ---------------------------------------

orders = pd.read_csv(
    "data/clean/orders.csv",
    parse_dates=["order_ts"]
)

print("Total cleaned orders:", len(orders))


# ---------------------------------------
# STEP 2: Select completed orders only
# ---------------------------------------

completed_orders = orders[
    orders["status"] == "completed"
].copy()

print("Completed orders:", len(completed_orders))


# ---------------------------------------
# STEP 3: Define the snapshot date
# ---------------------------------------

snapshot_date = pd.Timestamp("2025-10-01 00:00:00")


# ---------------------------------------
# STEP 4: Calculate customer RFM
# ---------------------------------------

rfm = completed_orders.groupby(
    "customer_id"
).agg(

    # Latest completed order date
    last_order_date=("order_ts", "max"),

    # Number of unique completed orders
    frequency=("order_id", "nunique"),

    # Total amount spent
    monetary=("order_amount", "sum")

).reset_index()


# ---------------------------------------
# STEP 5: Calculate recency
# ---------------------------------------

rfm["recency"] = (
    snapshot_date - rfm["last_order_date"]
).dt.days


# ---------------------------------------
# STEP 6: Keep required RFM columns
# ---------------------------------------

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Round monetary values
rfm["monetary"] = rfm["monetary"].round(2)


# ---------------------------------------
# STEP 7: Display summary
# ---------------------------------------

print("\n========== RFM ANALYSIS ==========")

print("Customers with completed orders:", len(rfm))

print("\nRecency:")
print("Minimum:", rfm["recency"].min())
print("Median:", rfm["recency"].median())
print("Maximum:", rfm["recency"].max())

print("\nFrequency:")
print("Mean:", round(rfm["frequency"].mean(), 2))
print("Maximum:", rfm["frequency"].max())

print("\nMonetary:")
print("Mean:", round(rfm["monetary"].mean(), 2))
print("Maximum:", round(rfm["monetary"].max(), 2))

print("\nFirst 10 customers:")
print(rfm.head(10).to_string(index=False))


# ---------------------------------------
# STEP 8: Validate reference results
# ---------------------------------------

assert len(rfm) == 1175
assert rfm["recency"].median() == 19
assert rfm["recency"].min() == 0
assert rfm["recency"].max() == 180
assert rfm["frequency"].max() == 33
assert round(rfm["frequency"].mean(), 2) == 9.46
assert round(rfm["monetary"].mean(), 2) == 2552.06
assert round(rfm["monetary"].max(), 2) == 14766.55

print("\nALL RFM CHECKPOINTS PASSED")



# ==================================================
# PART B - TASK 2: K-MEANS CUSTOMER SEGMENTATION
# ==================================================

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# STEP 9: Select RFM features
features = rfm[["recency", "frequency", "monetary"]]

# Scale features so large monetary amounts do not
# dominate recency and frequency.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# STEP 10: Test K-means for K = 2 to 8
print("\n========== K-MEANS EVALUATION ==========")

results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    cluster_labels = model.fit_predict(scaled_features)

    inertia = model.inertia_

    silhouette = silhouette_score(
        scaled_features,
        cluster_labels
    )

    results.append({
        "K": k,
        "Inertia": round(inertia, 2),
        "Silhouette": round(silhouette, 3)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


# STEP 11: Choose four clusters
final_k = 4

kmeans = KMeans(
    n_clusters=final_k,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(
    scaled_features
)


# STEP 12: Study the customer segments
profiles = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
)

print("\n========== SEGMENT PROFILES ==========")
print(profiles.round(2))


# STEP 13: Assign business names dynamically

# Highest mean monetary value = Premium
premium_id = profiles["mean_monetary"].idxmax()

remaining = [
    i for i in profiles.index if i != premium_id
]

# Highest frequency among remaining = Frequent
frequent_id = profiles.loc[
    remaining, "mean_frequency"
].idxmax()

remaining.remove(frequent_id)

# Highest recency among remaining = At Risk
at_risk_id = profiles.loc[
    remaining, "mean_recency"
].idxmax()

remaining.remove(at_risk_id)

# Remaining cluster = Occasional
occasional_id = remaining[0]

segment_names = {
    premium_id: "Premium",
    frequent_id: "Frequent Shoppers",
    at_risk_id: "At Risk",
    occasional_id: "Occasional Shoppers"
}

segment_actions = {
    "Premium": "Offer loyalty rewards",
    "Frequent Shoppers": "Offer repeat-purchase deals",
    "At Risk": "Launch retention campaigns",
    "Occasional Shoppers": "Send personalized offers"
}

rfm["segment_name"] = (
    rfm["segment_id"].map(segment_names)
)

rfm["is_premium"] = (
    rfm["segment_id"] == premium_id
)

profiles["segment_name"] = (
    profiles.index.map(segment_names)
)

profiles["action"] = (
    profiles["segment_name"].map(segment_actions)
)

print("\n========== NAMED SEGMENTS ==========")
print(profiles.round(2).to_string())


# STEP 14: Apply PCA to two components
pca = PCA(n_components=2)

pca_features = pca.fit_transform(scaled_features)

explained_variance = pca.explained_variance_ratio_

print("\n========== PCA RESULTS ==========")

print(
    "PC1 explained variance:",
    round(explained_variance[0], 3)
)

print(
    "PC2 explained variance:",
    round(explained_variance[1], 3)
)

print(
    "Total explained variance:",
    round(explained_variance.sum(), 3)
)


# STEP 15: Save the required CSV output
Path("outputs").mkdir(exist_ok=True)

output_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

rfm[output_columns].to_csv(
    "outputs/customer_segments.csv",
    index=False
)


# STEP 16: Validate segmentation
assert len(rfm) == 1175
assert rfm["segment_id"].nunique() == 4
assert rfm["segment_name"].notna().all()
assert rfm["is_premium"].sum() == profiles.loc[
    premium_id, "customers"
]

assert abs(explained_variance.sum() - 0.852) < 0.02

print("\nCUSTOMER SEGMENTATION COMPLETED")
print("Saved: outputs/customer_segments.csv")
