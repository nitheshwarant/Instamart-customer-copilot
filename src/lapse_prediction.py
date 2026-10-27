
"""
Part B - Customer Lapse Prediction
Instamart Customer Intelligence & Support Copilot

Predict customers who will stop ordering in September 2025.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# STEP 1: CONFIGURATION
# ==================================================

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
END = pd.Timestamp("2025-10-01 00:00:00")

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL = [
    "favourite_category",
    "city",
    "gender",
]


# ==================================================
# STEP 2: BUILD THE CUSTOMER COHORT
# ==================================================

def build_cohort():

    # Load cleaned data
    orders = pd.read_csv(
        "data/clean/orders.csv",
        parse_dates=["order_ts"]
    )

    customers = pd.read_csv(
        "data/clean/customers.csv",
        parse_dates=["signup_date"]
    )

    products = pd.read_csv(
        "data/clean/products.csv"
    )

    # Only pre-September orders can create features
    before = orders[
        orders["order_ts"] < CUTOFF
    ].copy()

    completed = before[
        before["status"] == "completed"
    ].copy()

    # Add product categories
    completed = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Calculate customer features
    features = completed.groupby("customer_id").agg(
        last_completed=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )

    # Days since last completed order
    features["recency"] = (
        CUTOFF - features.pop("last_completed")
    ).dt.days

    # Calculate cancellation rate
    final_orders = before[
        before["status"].isin(["completed", "cancelled"])
    ]

    cancellation = final_orders.groupby(
        "customer_id"
    )["status"].agg(
        total="size",
        cancelled=lambda s: (s == "cancelled").sum()
    )

    features["cancellation_rate"] = (
        cancellation["cancelled"] / cancellation["total"]
    ).reindex(features.index)

    # Most frequently purchased product category
    category_counts = completed.groupby(
        ["customer_id", "category"]
    ).size().reset_index(name="count")

    favourites = (
        category_counts.sort_values(
            ["customer_id", "count", "category"],
            ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
    )

    features["favourite_category"] = favourites

    # Add customer information
    customer_info = customers.set_index(
        "customer_id"
    )[
        ["city", "age", "gender", "signup_date"]
    ]

    features = features.join(customer_info, how="left")

    # Customer tenure in days
    features["tenure_days"] = (
        CUTOFF - features.pop("signup_date")
    ).dt.days

    # Identify September completed orders
    september_completed = orders.loc[
        (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < END)
        & (orders["status"] == "completed"),
        "customer_id"
    ]

    # Label 1 = lapsed, 0 = not lapsed
    features["lapsed"] = (
        ~features.index.isin(september_completed)
    ).astype(int)

    features = features.reset_index()

    # Reference validation
    assert len(features) == 1156
    assert features["lapsed"].sum() == 443

    # No missing feature values
    assert features[
        NUMERIC + CATEGORICAL
    ].notna().all().all()

    return features


# ==================================================
# STEP 3: MODEL EVALUATION FUNCTIONS
# ==================================================

def scores(y_true, y_pred):

    return {
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


def print_metrics(name, y_true, predictions):

    result = scores(y_true, predictions)

    print(f"\n===== {name} =====")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall: {result['recall']:.3f}")
    print(f"F1 Score: {result['f1']:.3f}")

    return result


# ==================================================
# STEP 4: CREATE MODEL PIPELINE
# ==================================================

def create_pipeline(classifier):

    # Numeric features are standardized
    # Categorical features are one-hot encoded
    preprocessing = ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            NUMERIC
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            CATEGORICAL
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", classifier),
    ])


# ==================================================
# STEP 5: MAIN MODEL TRAINING
# ==================================================

def main():

    cohort = build_cohort()

    print("\n===== CUSTOMER COHORT =====")

    print("Customers:", len(cohort))
    print("Lapsed:", cohort["lapsed"].sum())

    print(
        "Lapse Rate:",
        round(cohort["lapsed"].mean() * 100, 2),
        "%"
    )

    # Input features and target
    X = cohort[NUMERIC + CATEGORICAL].copy()
    y = cohort["lapsed"].copy()

    customer_ids = cohort["customer_id"].copy()

    assert X.notna().all().all()

    # ==================================================
    # STEP 6: TRAIN / TEST SPLIT
    # ==================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test
    ) = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # ==================================================
    # STEP 7: BASELINE MODEL
    # ==================================================

    baseline = np.zeros(len(y_test), dtype=int)

    print("\n===== BASELINE =====")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, baseline), 3)
    )

    print(
        "F1:",
        round(
            f1_score(y_test, baseline, zero_division=0),
            3
        )
    )

    # ==================================================
    # STEP 8: LOGISTIC REGRESSION GRID SEARCH
    # ==================================================

    print("\n===== LOGISTIC REGRESSION =====")

    lr_search = GridSearchCV(
        create_pipeline(
            LogisticRegression(max_iter=1000)
        ),
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100]
        },
        cv=5,
        scoring="f1",
        n_jobs=1
    )

    lr_search.fit(X_train, y_train)

    print("Best parameters:", lr_search.best_params_)
    print("Best CV F1:", round(lr_search.best_score_, 3))

    # ==================================================
    # STEP 9: RANDOM FOREST GRID SEARCH
    # ==================================================

    print("\n===== RANDOM FOREST =====")

    rf_search = GridSearchCV(
        create_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        ),
        {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_leaf": [5, 10, 20]
        },
        cv=5,
        scoring="f1",
        n_jobs=1
    )

    rf_search.fit(X_train, y_train)

    print("Best parameters:", rf_search.best_params_)
    print("Best CV F1:", round(rf_search.best_score_, 3))

    # ==================================================
    # STEP 10: TEST SET EVALUATION
    # ==================================================

    lr_pred = lr_search.predict(X_test)
    rf_pred = rf_search.predict(X_test)

    lr_metrics = print_metrics(
        "Logistic Regression Test Results",
        y_test,
        lr_pred
    )

    rf_metrics = print_metrics(
        "Random Forest Test Results",
        y_test,
        rf_pred
    )

    # ==================================================
    # STEP 11: TRAINING-ONLY SCALER CHECK
    # ==================================================

    fitted_scaler = (
        lr_search.best_estimator_
        .named_steps["preprocessor"]
        .named_transformers_["numeric"]
    )

    frequency_mean = fitted_scaler.mean_[
        NUMERIC.index("frequency")
    ]

    print("\n===== LEAKAGE CHECK =====")

    print("Training scaler mean:", frequency_mean)
    print("Full cohort mean:", cohort["frequency"].mean())

    print(
        "The means differ because scaling was fitted "
        "only on training data."
    )

    # ==================================================
    # STEP 12: CLASS BALANCING EXPERIMENT
    # ==================================================

    print("\n===== CLASS BALANCING =====")

    print(
        "Before:",
        y_train.value_counts().sort_index().to_dict()
    )

    majority_count = int((y_train == 0).sum())

    minority_positions = np.flatnonzero(
        y_train.to_numpy() == 1
    )

    # Duplicate minority training samples only
    extra_positions = np.random.default_rng(42).choice(
        minority_positions,
        size=majority_count - len(minority_positions),
        replace=True
    )

    X_balanced = pd.concat(
        [X_train, X_train.iloc[extra_positions]],
        ignore_index=True
    )

    y_balanced = pd.concat(
        [y_train, y_train.iloc[extra_positions]],
        ignore_index=True
    )

    print(
        "After:",
        y_balanced.value_counts().sort_index().to_dict()
    )

    balanced_lr = clone(lr_search.best_estimator_)
    balanced_lr.fit(X_balanced, y_balanced)

    balanced_metrics = print_metrics(
        "Balanced Logistic Regression",
        y_test,
        balanced_lr.predict(X_test)
    )

    print(
        "Recall change:",
        round(
            balanced_metrics["recall"] - lr_metrics["recall"],
            3
        )
    )

    print(
        "Precision change:",
        round(
            balanced_metrics["precision"] - lr_metrics["precision"],
            3
        )
    )

    print(
        "F1 change:",
        round(
            balanced_metrics["f1"] - lr_metrics["f1"],
            3
        )
    )

    # ==================================================
    # STEP 13: FEATURE IMPORTANCE
    # ==================================================

    print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")

    fitted_forest = rf_search.best_estimator_

    forest_features = (
        fitted_forest.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    forest_importances = (
        fitted_forest.named_steps["classifier"]
        .feature_importances_
    )

    importance = pd.Series(
        forest_importances,
        index=forest_features
    ).sort_values(ascending=False)

    print(importance.head(5))

    print("\n===== LOGISTIC COEFFICIENT SIGNS =====")

    logistic = lr_search.best_estimator_

    logistic_features = (
        logistic.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficients = pd.Series(
        logistic.named_steps["classifier"].coef_[0],
        index=logistic_features
    )

    for feature in [
        "numeric__recency",
        "numeric__frequency",
        "numeric__monetary"
    ]:
        value = coefficients[feature]
        sign = "positive" if value > 0 else "negative"
        print(feature, sign, round(value, 4))

    # ==================================================
    # STEP 14: SELECT BEST MODEL
    # ==================================================

    if rf_metrics["f1"] >= lr_metrics["f1"]:

        selected_model = fitted_forest
        selected_name = "Random Forest"

    else:

        selected_model = logistic
        selected_name = "Logistic Regression"

    final_predictions = selected_model.predict(X_test)

    assert f1_score(
        y_test, final_predictions, zero_division=0
    ) >= 0.55

    print("\nSelected model:", selected_name)

    print(
        "Missing a lapsing customer may lose a retention "
        "opportunity, whereas targeting a loyal customer "
        "may waste an incentive. Both recall and "
        "precision matter."
    )

    # ==================================================
    # STEP 15: SAVE AND RELOAD MODEL
    # ==================================================

    model_path = OUT / "lapse_model.joblib"

    joblib.dump(selected_model, model_path)

    loaded_model = joblib.load(model_path)

    assert np.array_equal(
        selected_model.predict(X_test),
        loaded_model.predict(X_test)
    )

    print("Model saved:", model_path)
    print("Reload validation passed")

    # ==================================================
    # STEP 16: PREMIUM RETENTION LIST
    # ==================================================

    segments = pd.read_csv(
        OUT / "customer_segments.csv"
    )

    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    predictions = pd.DataFrame({
        "customer_id": id_test.to_numpy(),
        "lapse_probability": (
            selected_model.predict_proba(X_test)[:, 1]
        ),
        "actually_lapsed": y_test.to_numpy()
    })

    retention = predictions.merge(
        segments[["customer_id", "is_premium"]],
        on="customer_id",
        how="left"
    )

    retention = retention[
        retention["is_premium"].fillna(False)
    ]

    retention = retention.sort_values(
        "lapse_probability",
        ascending=False
    ).head(10)

    retention = retention[
        [
            "customer_id",
            "lapse_probability",
            "actually_lapsed"
        ]
    ]

    retention.to_csv(
        OUT / "retention_list.csv",
        index=False
    )

    print("\n===== RETENTION LIST =====")

    print("Customers:", len(retention))

    print(
        "Actually lapsed:",
        retention["actually_lapsed"].sum()
    )

    print(
        "Limitation: Premium segments were calculated "
        "using the full order window. The retention "
        "list is illustrative, not a leakage-free evaluation."
    )

    print("\nPART B LAPSE PREDICTION COMPLETED")


# ==================================================
# STEP 17: EXECUTE PROGRAM
# ==================================================

if __name__ == "__main__":
    main()
