
"""Part C: Instamart Support Ticket Triage."""

from pathlib import Path
import json
import random
import re
import time
import sys

import pandas as pd

# mock_llm.py is in the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mock_llm


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

ALLOWED = {"billing", "delivery", "technical", "fraud"}
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

OUTPUT_COLUMNS = [
    "ticket_id", "customer_id", "status", "category",
    "segment_name", "priority", "route", "tokens_used"
]

# Regular expressions must run in this order.
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")
]


# ==========================================
# STEP 2: ZERO-SHOT AND FEW-SHOT PROMPTS
# ==========================================

ZERO_SHOT = (
    "CLASSIFY the customer's complaint into exactly one of "
    "billing, delivery, technical, fraud. "
    "Reply with JSON only, using this exact shape: "
    '{"category": "...", "reason": "..."}'
)

FEW_SHOT = ZERO_SHOT + (
    "\nEXAMPLES:\n"
    'Unauthorized wallet transaction -> '
    '{"category":"fraud","reason":"unauthorized use"}\n'
    'Rider arrived late -> '
    '{"category":"delivery","reason":"late delivery"}'
)


# ==========================================
# STEP 3: MASK PERSONAL INFORMATION
# ==========================================

def redact(text):
    """Replace sensitive identifiers with placeholders."""
    counts = {}

    for name, pattern, placeholder in PATTERNS:
        text, counts[name] = pattern.subn(placeholder, text)

    return text, counts


def assert_no_pii(text):
    """Ensure no sensitive identifiers remain."""
    assert all(
        not pattern.search(text)
        for _, pattern, _ in PATTERNS
    ), "Sensitive information would be sent to the model"


# ==========================================
# STEP 4: VALIDATE AI RESPONSES
# ==========================================

def parse_response(content):
    """Accept valid JSON or one clear category keyword."""

    try:
        obj = json.loads(content)

    except (json.JSONDecodeError, TypeError):
        words = set(
            re.findall(r"[a-z]+", (content or "").lower())
        )

        matches = words & ALLOWED

        if len(matches) == 1:
            return "classified_keyword", matches.pop()

        return "flagged_invalid_output", ""

    if isinstance(obj, dict) and obj.get("category") in ALLOWED:
        return "classified_json", obj["category"]

    return "flagged_invalid_output", ""


# ==========================================
# STEP 5: MODEL CALLS WITH RETRIES
# ==========================================

def call_with_retry(ticket_id, messages, rng):
    """Allow one initial call and up to three retries."""

    for attempt in range(4):
        try:
            return mock_llm.chat(messages), attempt

        except mock_llm.TransientLLMError:
            if attempt == 3:
                return None, attempt

            wait = (
                BASE_DELAY
                * (2 ** attempt)
                * rng.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"number={attempt + 1} "
                f"wait={wait:.4f}s"
            )

            time.sleep(wait)

    return None, 3


# ==========================================
# STEP 6: ROUTE SUPPORT TICKETS
# ==========================================

def route_ticket(status, category, is_premium):

    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }

    if status in human_statuses:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


# ==========================================
# STEP 7: COMPLETE TRIAGE PIPELINE
# ==========================================

def run_pipeline(tickets, segments, prompt, label, budget=None):

    # Reset the simulated model for each full run.
    mock_llm.reset()

    rng = random.Random(42)

    total_tokens = 0
    retry_count = 0
    rows = []

    # Customer lookup from Part B.
    segment_lookup = segments.set_index("customer_id")

    premium_by_customer = (
        segment_lookup["is_premium"].to_dict()
    )

    name_by_customer = (
        segment_lookup["segment_name"].to_dict()
    )

    # Process in the original CSV order.
    for ticket in tickets.itertuples(index=False):

        text = ticket.text
        safe_text, _ = redact(text)

        assert_no_pii(safe_text)

        status = ""
        category = ""
        used = 0

        # Input guardrails
        if not text.strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        elif budget is not None and total_tokens >= budget:
            status = "budget_exhausted"

        else:
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": safe_text
                }
            ]

            # The user prompt must contain only redacted text.
            assert messages[-1]["content"] == safe_text

            for message in messages:
                assert_no_pii(message["content"])

            response, retries = call_with_retry(
                ticket.ticket_id,
                messages,
                rng
            )

            retry_count += retries

            if response is None:
                status = "flagged_invalid_output"

            else:
                used = response.usage.total_tokens
                total_tokens += used

                status, category = parse_response(
                    response.choices[0].message.content
                )

        premium = bool(
            premium_by_customer.get(ticket.customer_id, False)
        )

        route, priority = route_ticket(
            status, category, premium
        )

        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "segment_name": name_by_customer.get(
                ticket.customer_id, ""
            ),
            "priority": priority,
            "route": route,
            "tokens_used": used
        })

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    classified = result["status"].isin({
        "classified_json",
        "classified_keyword"
    })

    joined = result.merge(
        tickets[["ticket_id", "true_category"]],
        on="ticket_id",
        validate="one_to_one"
    )

    if classified.any():
        accuracy = (
            joined.loc[classified, "category"]
            == joined.loc[classified, "true_category"]
        ).mean()
    else:
        accuracy = 0.0

    # Print evaluation results.
    print(f"\n=== {label} ===")

    print(
        "Statuses:",
        result["status"].value_counts().to_dict()
    )

    print("Classified:", int(classified.sum()))

    print(
        "Transient errors recovered by retries:",
        retry_count
    )

    print(f"Accuracy among classified: {accuracy:.1%}")

    print("Total model tokens:", total_tokens)

    print(
        "Tokens per classified ticket:",
        round(total_tokens / max(1, int(classified.sum())), 2)
    )

    # Zero-shot confusion table
    if label == "Zero-shot full run":

        print("\nConfusion Table:")

        confusion = pd.crosstab(
            joined.loc[classified, "true_category"],
            joined.loc[classified, "category"]
        ).reindex(
            index=sorted(ALLOWED),
            columns=sorted(ALLOWED),
            fill_value=0
        )

        print(confusion)

    return result


# ==========================================
# STEP 8: RUN ALL EXPERIMENTS
# ==========================================

def main():

    Path("outputs").mkdir(exist_ok=True)

    tickets = pd.read_csv(
        "data/raw/tickets.csv",
        keep_default_na=False
    )

    segments = pd.read_csv(
        "outputs/customer_segments.csv",
        keep_default_na=False
    )

    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # ======================================
    # REDACTION AUDIT
    # ======================================

    substitution_counts = {
        name: 0 for name, _, _ in PATTERNS
    }

    for raw_text in tickets["text"]:

        masked, counts = redact(raw_text)

        assert_no_pii(masked)

        for key, n in counts.items():
            substitution_counts[key] += n

    print("Tickets:", len(tickets))

    print("Redactions:", substitution_counts)

    print(
        "Total redactions:",
        sum(substitution_counts.values())
    )

    assert len(tickets) == 164

    assert substitution_counts == {
        "email": 55,
        "phone": 91,
        "order_id": 132
    }

    # ======================================
    # ZERO-SHOT RUN
    # ======================================

    zero = run_pipeline(
        tickets,
        segments,
        ZERO_SHOT,
        "Zero-shot full run"
    )

    # ======================================
    # FEW-SHOT RUN
    # ======================================

    few = run_pipeline(
        tickets,
        segments,
        FEW_SHOT,
        "Few-shot full run"
    )

    # ======================================
    # TOKEN BUDGET DEMONSTRATION
    # ======================================

    demo_budget = 6000

    demo = run_pipeline(
        tickets,
        segments,
        ZERO_SHOT,
        "Zero-shot budget demo",
        demo_budget
    )

    demo.to_csv(
        "outputs/triage_budget_demo.csv",
        index=False
    )

    exhausted = int(
        (demo["status"] == "budget_exhausted").sum()
    )

    print(
        f"Budget demo: budget={demo_budget}; "
        f"budget_exhausted={exhausted}"
    )

    assert exhausted > 0

    # ======================================
    # FINAL FEW-SHOT TRIAGE
    # ======================================

    final = run_pipeline(
        tickets,
        segments,
        FEW_SHOT,
        "Final few-shot run",
        100000
    )

    final.to_csv(
        "outputs/triage_results.csv",
        index=False
    )

    # ======================================
    # REFERENCE VALIDATION
    # ======================================

    for result in (zero, few, final):

        counts = result["status"].value_counts()

        assert counts.get("skipped_empty", 0) == 4

        assert counts.get("flagged_long", 0) == 4

        assert counts.get("classified_json", 0) == 135

        assert counts.get("classified_keyword", 0) == 12

        assert counts.get("flagged_invalid_output", 0) == 9

    assert len(final) == 164

    assert int(
        (final["status"] == "budget_exhausted").sum()
    ) == 0

    print("\nPART C TICKET TRIAGE COMPLETED")

    print("Saved: outputs/triage_results.csv")

    print("Saved: outputs/triage_budget_demo.csv")


if __name__ == "__main__":
    main()
