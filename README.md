# Overview

## Instamart Customer Intelligence & Support Copilot

This capstone implements a reproducible, offline customer intelligence and support system using synthetic Instamart data.

It contains four connected parts:

| Part | Description |
|---|---|
| A | Data cleaning, validation, SQLite storage and SQL analysis |
| B | RFM segmentation, K-means clustering and customer lapse prediction |
| C | Privacy-safe AI support ticket classification and routing |
| D | Policy RAG assistant and simulated refund agent |

**Technology:** Python 3.12, pandas, NumPy, SQLite, scikit-learn, joblib and the supplied offline `mock_llm.py`.

All data, policies, customer records and refund actions are fictional. No external AI API, network request or payment service is required.

# Reproduce

Requirements: Python 3.11 or newer.

Run the following commands from the repository root, in order.

1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `source .venv/bin/activate` on macOS/Linux, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell.
3. Install packages: `python -m pip install -r requirements.txt`
4. Generate synthetic data: `python generate_data.py`
5. Audit raw data: `python src/audit.py`
6. Clean data and build SQLite: `python src/clean_data.py`
7. Validate cleaned data: `python src/validate.py`
8. Run SQL and pandas analysis: `python src/sql_analysis.py`
9. Generate customer segments: `python src/customer_segments.py`
10. Train and evaluate lapse models: `python src/lapse_prediction.py`
11. Run ticket triage: `python src/ticket_triage.py`
12. Evaluate policy RAG: `python src/policy_rag.py`
13. Test the initial refund agent: `python src/refund_agent.py`
14. Run all agent scenarios and demonstrations: `python src/agent_demo.py`

On the development machine, Windows Python 3.12 was installed separately from MSYS2 Python. If `python` points to MSYS2 during environment creation, use the official Windows Python executable explicitly.

The scripts regenerate the required CSV outputs, SQLite database, saved model, ticket-routing results, refund logs and checkpoint files. `agent_demo.py` resets and recreates its simulated refund logs.

# Part A

## Raw Data Audit

The generator produced 1,200 customer records, 40 products, 13,084 raw order rows and 164 support tickets.

### Dataset shapes and column quality

| Table | Shape |
|---|---|
| customers | 1,200 × 5 |
| products | 40 × 4 |
| orders | 13,084 × 9 |

| Column | Raw data type | Missing count | Missing % |
|---|---|---:|---:|
| customers.customer_id | str | 0 | 0.00 |
| customers.city | str | 0 | 0.00 |
| customers.signup_date | str | 0 | 0.00 |
| customers.age | float64 | 61 | 5.08 |
| customers.gender | str | 30 | 2.50 |
| products.product_id | str | 0 | 0.00 |
| products.product_name | str | 0 | 0.00 |
| products.category | str | 0 | 0.00 |
| products.price_inr | int64 | 0 | 0.00 |
| orders.order_id | str | 0 | 0.00 |
| orders.customer_id | str | 0 | 0.00 |
| orders.product_id | str | 0 | 0.00 |
| orders.quantity | int64 | 0 | 0.00 |
| orders.order_ts | str | 0 | 0.00 |
| orders.status | str | 0 | 0.00 |
| orders.discount_pct | int64 | 0 | 0.00 |
| orders.order_amount | str | 53 | 0.41 |
| orders.delivery_minutes | float64 | 2,541 | 19.42 |

### Raw quality findings

| Metric | Count |
|---|---:|
| Exact order duplicates | 119 |
| Pending order rows | 656 |
| Unique order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank ages | 61 |
| Ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Raw nonblank gender spellings | 10 |
| Blank genders | 30 |

The 120 non-numeric amounts include both blank entries and invalid text values.

## Cleaning Rules

Customers: City names were stripped of surrounding whitespace and converted to title case. Gender spellings were normalized to male, female or unknown. Six invalid ages were treated as missing; 67 total missing ages were imputed using the median age within each city. Signup dates were converted to datetime.

Orders: Timestamps were converted to datetime. The pipeline removed 119 exact duplicates and retained the latest timestamp for each order ID, removing 656 older records. Invalid order amounts were reconstructed from product price, quantity and discount. Negative delivery times were treated as missing. Missing delivery times for completed orders were imputed using their product-category median; delivery times for cancelled orders were left missing.

**Post-deduplication repairs:** 112 invalid order amounts were reconstructed, 105 negative delivery durations were cleared, and 774 missing delivery times for completed orders were filled using the product-category median. These post-deduplication counts differ from the raw audit because exact duplicates and superseded order records were removed first.

### Clean customer distribution

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

Gender distribution: female 610, male 560, unknown 30. No missing ages remain, and all ages are within 18–60.

### Clean-order validation

| Checkpoint | Result |
|---|---:|
| Unique orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders | 0 |
| Missing completed delivery times | 0 |
| Missing cancelled delivery times | 1,199 |
| Missing order amounts | 0 |
| Completed-order revenue | INR 2,998,668.80 |

The three cleaned datasets were saved as CSVs and stored in `data/instamart.db` in tables named `customers`, `products` and `orders`.

The validation script passed its reference assertions and printed `ALL CHECKPOINTS PASSED`.

## SQL Analysis

Seven queries were implemented in `sql/queries.sql`.

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE and HAVING | 30 qualifying customers |
| Q2 | Three-table join | 30 city-category revenue groups |
| Q3 | Scalar subquery | 11 products above average catalogue price |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | Two CTEs | 550 customers above average spend |
| Q6 | Ranking window functions | 8 rows where RANK differs from ROW_NUMBER |
| Q7 | Running revenue total | 183 days; INR 2,998,668.80 cumulative |

The largest city-category revenue group was Bengaluru / Electronics, at INR 233,909.45. The average customer completed spend was INR 2,552.06.

Top spending customers by city:

| City | Customer ID | Spend (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

RANK assigns tied spending values the same rank and skips subsequent rank positions; ROW_NUMBER assigns a unique sequential number even when values tie.

## Pandas Cross-Check

A `pivot_table` with margins reproduced the SQL completed-revenue total of INR 2,998,668.80.

| City | Revenue sum (INR) | Mean order amount (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory Data Analysis

**What is the data about?** The dataset contains synthetic customer purchases across six Indian cities, including product categories, transaction amounts, discounts, quantities and delivery times.

**What do its distributions look like?** Completed-order amounts are strongly right-skewed (skewness 4.609). The mean is INR 269.91, compared with a median of INR 135.00. Q1 is INR 64.80 and Q3 is INR 297.00.

**Are there outliers?** The IQR is INR 232.20. The lower and upper fences are INR -283.50 and INR 645.30. There are 897 upper outliers and no lower outliers.

**What relationships exist?** The strongest absolute correlation is between order amount and discount percentage (-0.095), a very weak negative relationship. Order amount and quantity have a correlation of approximately +0.087. Correlations of this size do not establish causation.

# Part B

## RFM Customer Segmentation

RFM features were calculated as of October 1, 2025, using completed orders only.

| Metric | Value |
|---|---:|
| Customers with completed orders | 1,175 |
| Median recency | 19 days |
| Recency range | 0–180 days |
| Mean frequency | 9.46 |
| Maximum frequency | 33 |
| Mean monetary | INR 2,552.06 |
| Maximum monetary | INR 14,766.55 |

The three RFM features were standardized with `StandardScaler`.

### K-means Evaluation

Models used `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

Four clusters were selected because K=4 provides a strong elbow improvement, the highest silhouette score and understandable business segments.

K=8 has lower inertia than K=2 because increasing cluster count reduces within-cluster distances. Inertia alone cannot determine the best K because more clusters may reduce interpretability and business usefulness.

### Final Segments

| Segment ID | Name | Customers | Mean recency | Mean frequency | Mean spend (INR) |
|---|---|---:|---:|---:|---:|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| 2 | Frequent Shoppers | 419 | 8.77 | 19.40 | 3,410.02 |
| 3 | At Risk | 182 | 109.04 | 2.20 | 685.99 |

The recommended actions are personalized offers for Occasional Shoppers, loyalty rewards for Premium customers, repeat-purchase offers for Frequent Shoppers and retention campaigns for At Risk customers.

Cluster numbers are arbitrary identifiers, not rankings. Premium is the segment with the highest mean monetary value.

PCA explained-variance ratios were 0.634 and 0.218, totaling approximately 0.852.

The segment assignments were saved in `outputs/customer_segments.csv`.

## Customer Lapse Prediction

The prediction cutoff was September 1, 2025. Customers with a completed order before this cutoff formed the cohort. A customer was labelled lapsed if they had no completed order during September 2025.

Features included pre-cutoff recency, frequency, monetary spend, average discount, cancellation rate, average completed-order delivery time, favourite category, city, age, gender and tenure.

No orders dated September 1, 2025 or later, or whole-window segment labels, were included in the prediction features.

| Checkpoint | Result |
|---|---:|
| Cohort customers | 1,156 |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training rows | 924 |
| Test rows | 232 |
| Always-not-lapsed baseline accuracy | 61.6% |
| Baseline lapsed-class F1 | 0 |

The data was split with stratification and `random_state=42`.

Numeric features were standardized, and categorical features were one-hot encoded with the first category dropped. Preprocessing was fitted using training data only. All feature columns were checked for missing values.

### Model Tuning and Evaluation

Logistic Regression used five-fold F1 grid search over C = 0.01, 0.1, 1, 10 and 100. Random Forest used 100 trees, `random_state=42`, and five-fold F1 grid search over depths 3, 5 and 8 and minimum leaf sizes 5, 10 and 20.

**Model comparison**

| Model | Best parameters | Best CV F1 | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | C = 100 | 0.651 | 0.644 | 0.652 | 0.648 |
| Random Forest | max_depth = 8, min_samples_leaf = 10 | 0.687 | 0.640 | 0.640 | 0.640 |
| Balanced Logistic Regression | C = 100 (not separately tuned) | Not separately tuned | 0.600 | 0.843 | 0.701 |

**Confusion matrices**

Rows represent actual class 0 (not lapsed) or 1 (lapsed); columns represent predicted class 0 or 1.

| Model | True negative | False positive | False negative | True positive |
|---|---:|---:|---:|---:|
| Logistic Regression | 111 | 32 | 31 | 58 |
| Random Forest | 111 | 32 | 32 | 57 |
| Balanced Logistic Regression | 93 | 50 | 14 | 75 |

**Training-only preprocessing and leakage prevention**

- Training scaler frequency mean: 8.114719.
- Full-cohort frequency mean: 8.028547.
- Training examples: 924; held-out test examples: 232.
- Training class counts before oversampling: 570 non-lapsed, 354 lapsed.
- Training class counts after oversampling: 570 non-lapsed, 570 lapsed.

The training-only scaler and full-cohort means differ because the training split is only part of the population. Encoders and scalers were fitted using training data within the model pipelines. Resampling used training data only, and neither September orders nor whole-window segment labels were included in model features.

**Class balancing and business trade-off**

Balancing increased recall from 0.652 to 0.843 (**+0.191**), reduced precision from 0.644 to 0.600 (**-0.044**), and improved F1 from 0.648 to 0.701 (**+0.053**). Recall changed the most. A retention team could identify more customers who would lapse, but would also target more customers who would otherwise continue ordering, potentially wasting incentives.

**Random Forest feature importance**

| Feature | Importance | Business interpretation |
|---|---:|---|
| Monetary | 0.277830 | Past spending strongly contributes to classification. |
| Frequency | 0.276787 | Ordering habits strongly contribute to classification. |
| Recency | 0.111219 | Days since the latest order provide a lapse signal. |
| Average delivery minutes | 0.061649 | Delivery experience contributes predictive information. |
| Average discount percentage | 0.053178 | Discount usage provides additional information. |

**Logistic Regression coefficient signs**

| Feature | Coefficient | Interpretation |
|---|---:|---|
| Recency | +0.0396 | Longer inactivity is associated with slightly higher predicted lapse odds, holding other model features constant. |
| Frequency | -1.6982 | More frequent ordering is associated with lower predicted lapse odds. |
| Monetary | +0.2846 | Higher spending has a positive conditional coefficient after controlling for other features; this does not establish causation. |

**Selected model and retention list**

The original tuned Logistic Regression was saved as the selected model because its held-out test F1 (0.648) was slightly higher than the tuned Random Forest's (0.640). The balanced Logistic Regression achieved a higher F1 of 0.701 but was retained as a separate experiment. Selecting a model by comparing performance on the test set can bias performance reporting; an independent final validation set or cross-validation selection should be used in future evaluations.

The selected model was saved to `outputs/lapse_model.joblib`, reloaded, and verified to reproduce its test predictions exactly.

The top 10 Premium test customers were ranked by lapse probability and saved to `outputs/retention_list.csv`; **3 of the 10 actually lapsed**. The Premium segment was derived from the entire order window, including September, so this retention list is illustrative and not leakage-free, although the prediction features are pre-cutoff only.

# Part C

## Support Ticket Guardrails

The triage pipeline processed 164 tickets in their original CSV order.

Four empty tickets were skipped, and four long tickets were flagged because their estimated input size exceeded 400 tokens. Neither group was sent to the model.

Before AI classification, the pipeline masked email addresses, Indian mobile phone numbers and order IDs.

| Personal information | Substitutions |
|---|---:|
| Emails | 55 |
| Phone numbers | 91 |
| Order identifiers | 132 |
| Total | 278 |

After masking, no identifier matching any of the three configured patterns remained in the redacted text.

## Classification and Validation

Two prompts were evaluated: zero-shot classification and few-shot classification with labelled examples. Both required JSON with one of four categories: billing, delivery, technical or fraud.

The pipeline used a maximum of three retries, exponential backoff and random jitter to recover transient errors.

Each full run recovered 19 temporary failures.

| Outcome | Tickets |
|---|---:|
| Valid JSON classification | 135 |
| Keyword fallback classification | 12 |
| Invalid AI output sent to human review | 9 |
| Skipped empty | 4 |
| Flagged long | 4 |

Valid JSON category values were accepted. Non-JSON replies were accepted only if exactly one allowed category was found using lowercase word extraction. Unsupported category values were sent to human review.

### Zero-shot versus Few-shot

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.7% | 100% |
| Total tokens | 9,400 | 12,052 |
| Tokens per classified ticket | 63.95 | 81.99 |

Zero-shot confusion table:

| True category | Predicted billing | Predicted delivery | Predicted fraud | Predicted technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Few-shot classification used 2,652 additional tokens, improving accuracy by 14.3 percentage points. It is worth the additional token cost when avoiding misrouted fraud and technical complaints is more important than minimizing token use.

## Token Budget and Routing

A zero-shot demonstration used a budget of 6,000 tokens. It classified 95 tickets, consumed 6,038 tokens and routed 56 remaining tickets to human review with `budget_exhausted`. The final successful model response may move the running count above the threshold; the guardrail prevents further calls once the running total reaches the budget.

Fraud tickets were routed to `security_team` with HIGH priority. Other successfully classified Premium tickets went to `premium_desk` with HIGH priority, and remaining valid tickets went to `support_queue` with NORMAL priority. Skipped, oversized, invalid and budget-exhausted tickets went to `human_review` with NORMAL priority.

Final outputs: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

# Part D

## Policy RAG Assistant

Five synthetic policy documents were split into chunks of 40 words with 10-word overlap, yielding 15 chunks:

| Policy | Chunks |
|---|---:|
| Refund | 4 |
| Delivery | 3 |
| Escalation | 3 |
| Payment | 3 |
| Privacy | 2 |

TF-IDF vectors and cosine similarity were used to rank relevant text. Term frequency measures how strongly a term appears in a chunk; inverse document frequency reduces the importance of terms that appear in many chunks. TF-IDF has limited understanding of paraphrases expressed with different vocabulary.

The system retrieved three relevant chunks for each question and asked `mock_llm.py` to answer only from those chunks.

### Retrieval Evaluation

| Metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Chunks | 15 | 13 |
| Correct source retrieval | 12/12 | 12/12 |
| Answers containing expected fact | 12/12 | 10/12 |
| Correct unanswerable fallbacks | 2/2 | 2/2 |

With zero overlap, two policy questions failed: the late-delivery refund threshold question and the failed-payment return-time question. Relevant details were split across chunk boundaries, illustrating the benefit of overlap.

## Refund Agent

The ReAct agent used five tools:

| Tool | Responsibility |
|---|---|
| get_order | Read order details from SQLite |
| search_policy | Retrieve relevant policy chunks |
| calculate_refund | Calculate eligibility using delivery time and amount |
| issue_refund | Append an eligible simulated refund to the ledger |
| escalate_to_human | Append a high-value refund escalation to the ledger |

Only completed orders delivered in more than 45 minutes were eligible for a full late-delivery refund. Eligible refunds up to INR 1,000 could be recorded automatically, while larger amounts required human escalation. The current simulated agent does not check the policy's seven-day request window; this remains a limitation to address before real-world use.

### Seven Scenario Results

| Scenario | Status | Refund or requested amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 872 |
| S2 | REFUND | 135.00 | 5 | 872 |
| S3 | ESCALATE | 1,999.00 | 5 | 859 |
| S4 | DENIED | 0.00 | 4 | 585 |
| S5 | DENIED | 0.00 | 3 | 322 |
| S6 | NOT_FOUND | 0.00 | 2 | 62 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,970 |

The INR 1,999 in S3 represents an eligible amount referred for approval; no automatic payment was issued.

### Safe Stopping

| Demonstration | Result |
|---|---|
| S7, maximum six model calls | FALLBACK_MAX_ITERATIONS |
| S1, token limit 200 | FALLBACK_TOKEN_LIMIT |
| S1, time limit zero seconds | FALLBACK_TIMEOUT |

Each fallback produced a human-handoff summary and retained the completed steps.

### Crash and Resume

For S1, a simulated crash occurred immediately after the second tool execution and its checkpoint was saved.

Recovery continued with `resume_agent()` without repeating the two completed tools.

Exact tool sequence:

`get_order → search_policy → calculate_refund → issue_refund`

The resumed run finished with REFUND for INR 128.25 after five model calls.

Checkpoint files are saved under `state/`. Seven scenario traces are saved to `outputs/agent_runs.jsonl`, and the ledger records are saved to `outputs/refund_ledger.jsonl`.

### ReAct versus Plan-and-Execute

ReAct chooses the next tool after reading each previous observation. In the measured scenarios, S1 used five model calls, S5 used three and S6 used two. The repeated calls incur additional estimated tokens, but the agent can react immediately when an order is missing, cancelled or otherwise ineligible.

A plan-and-execute approach could generate the entire tool plan in one model call and replan only when needed. It may use fewer model calls but risks following inappropriate steps if its assumptions prove incorrect.

For this simulated refund workflow, I prefer ReAct because order status and eligibility should be verified before recording any refund or escalation. The design also allows explicit stopping limits and checkpoint-based recovery. A production implementation would additionally require stronger transaction safety, authorization and idempotency controls.

# Limitations and Next Steps

The entire project uses synthetic data and the deterministic offline `mock_llm.py`. Results do not establish performance on real customers or live support traffic.

The clustering labels are business interpretations of mathematical groups rather than independently validated customer types.

The prediction model was evaluated on a fixed synthetic time window. Model drift, seasonal changes and performance on future data remain untested.

The Premium flag uses full-window data, including the September outcome window, so the retention list is illustrative and not a leakage-free evaluation.

The ticket redaction rules detect the specified patterns but cannot guarantee the removal of every possible form of personal information.

Policy retrieval uses TF-IDF rather than a semantic embedding model. It may fail on paraphrased questions, and the assistant is evaluated only against the supplied policy questions.

The refund agent records simulated actions rather than transferring real money. It also does not enforce the policy's seven-day request window. Real deployment would require authentication, authorization, audit controls, transactional idempotency, time-window validation and human approval integration.

Possible future improvements include temporal model validation, broader privacy detection, semantic retrieval, monitoring, role-based permissions and stronger recovery guarantees.
