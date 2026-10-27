
"""Part D: Offline refund agent with safe limits and recovery."""

import csv
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from policy_rag import PolicyRAG

DB = ROOT / "data" / "instamart.db"
LEDGER = ROOT / "outputs" / "refund_ledger.jsonl"
STATE = ROOT / "state"
RAG = None


# ==========================================
# STEP 1: DEFINE FIVE REFUND TOOLS
# ==========================================

def get_order(order_id):
    """Look up an order in the SQLite database."""

    with sqlite3.connect(DB) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, "
            "delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,)
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": (
            None if row[3] is None else float(row[3])
        )
    }


def search_policy(query):
    """Search policies using the existing TF-IDF retriever."""

    global RAG

    if RAG is None:
        RAG = PolicyRAG(overlap=10)

    return {"results": RAG.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate an eligible late-delivery refund."""

    amount = float(order_amount)

    if delivery_minutes is None or amount <= 0:
        return {"refund_inr": 0.0}

    if float(delivery_minutes) > 45:
        return {"refund_inr": round(amount, 2)}

    return {"refund_inr": 0.0}


def _record_action(tool_name, order_id, refund_inr):
    """Validate the refund and write a simulated ledger record."""

    order = get_order(order_id)

    if "error" in order or order["status"] != "completed":
        raise ValueError(
            "Refund requires an existing completed order"
        )

    eligible = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"]
    )["refund_inr"]

    amount = round(float(refund_inr), 2)

    if eligible <= 0 or not isclose(
        amount, eligible, abs_tol=0.01
    ):
        raise ValueError(
            "Refund amount does not match the policy"
        )

    if tool_name == "issue_refund" and amount > 1000:
        raise ValueError(
            "Refunds above INR 1000 require human approval"
        )

    if tool_name == "escalate_to_human" and amount <= 1000:
        raise ValueError(
            "Only refunds above INR 1000 need escalation"
        )

    LEDGER.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": amount
    }

    with LEDGER.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {"refund_inr": amount}


def issue_refund(order_id, refund_inr):
    """Record an eligible simulated automatic refund."""

    return _record_action(
        "issue_refund", order_id, refund_inr
    )


def escalate_to_human(order_id, refund_inr):
    """Send a high-value refund for simulated human review."""

    return _record_action(
        "escalate_to_human", order_id, refund_inr
    )


# ==========================================
# STEP 2: TOOL SPECIFICATIONS AND MAPPING
# ==========================================

def _spec(name, description, properties, required):
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


TOOLS_SPEC = [
    _spec(
        "get_order",
        "Look up an order in the cleaned SQLite database.",
        {"order_id": {"type": "string"}},
        ["order_id"]
    ),
    _spec(
        "search_policy",
        "Find relevant excerpts from policy files.",
        {"query": {"type": "string"}},
        ["query"]
    ),
    _spec(
        "calculate_refund",
        "Calculate the late-delivery refund amount.",
        {
            "order_amount": {"type": "number"},
            "delivery_minutes": {"type": "number"}
        },
        ["order_amount", "delivery_minutes"]
    ),
    _spec(
        "issue_refund",
        "Log an eligible simulated refund up to INR 1000.",
        {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}
        },
        ["order_id", "refund_inr"]
    ),
    _spec(
        "escalate_to_human",
        "Log a refund above INR 1000 for a human lead.",
        {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}
        },
        ["order_id", "refund_inr"]
    )
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# ==========================================
# STEP 3: CHECKPOINT HELPERS
# ==========================================

def _checkpoint_path(run_id):

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("Invalid run_id")

    return STATE / f"{run_id}.json"


def _save_checkpoint(state):
    """Save agent progress after each completed tool."""

    STATE.mkdir(parents=True, exist_ok=True)

    path = _checkpoint_path(state["run_id"])
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )

    temporary.replace(path)


def _finish(state, status, answer):
    """Build and save the final agent result."""

    result = {
        "status": status,
        "final_answer": answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"]
    }

    state["result"] = result
    _save_checkpoint(state)

    return result


def _fallback(state, reason):
    """Hand unfinished requests to a human."""

    last_tool = (
        state["tools_executed"][-1]
        if state["tools_executed"]
        else "none"
    )

    summary = (
        f"Human handoff: {reason} for "
        f"{state['scenario_id']}; "
        f"tools completed={len(state['steps'])}; "
        f"last tool={last_tool}."
    )

    return _finish(state, reason, summary)


# ==========================================
# STEP 4: REACT AGENT LOOP
# ==========================================

def _continue(state, crash_after_tool=None):

    start = time.monotonic()

    if "result" in state:
        return state["result"]

    while state["llm_calls"] < state["max_iterations"]:

        elapsed = (
            state["elapsed_seconds"]
            + time.monotonic() - start
        )

        if elapsed >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")

        # Ask the model for its next action.
        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # Stop before executing a requested tool if over budget.
        if state["tokens_used"] > state["token_limit"]:
            return _fallback(
                state, "FALLBACK_TOKEN_LIMIT"
            )

        message = response.choices[0].message

        # No tool calls means the model has finished.
        if not message.tool_calls:

            final = (
                message.content
                or "NEEDS_INFO: ask a human agent."
            )

            status = final.split(":", 1)[0].strip()

            return _finish(state, status, final)

        calls = []

        for tool_call in message.tool_calls:

            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": arguments
            })

        # Record the assistant tool request.
        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": calls
        })

        for tool_call in calls:

            name = tool_call["name"]
            arguments = tool_call["arguments"]

            try:
                function = TOOL_MAP[name]
                observation = function(**arguments)

            except (KeyError, TypeError, ValueError) as error:
                observation = {"error": str(error)}

            # Record the tool response.
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": json.dumps(observation)
            })

            state["steps"].append({
                "tool": name,
                "arguments": arguments,
                "observation": observation
            })

            state["tools_executed"].append(name)
            state["step_number"] = len(state["steps"])

            state["elapsed_seconds"] += (
                time.monotonic() - start
            )
            start = time.monotonic()

            # Save progress before simulating any crash.
            _save_checkpoint(state)

            if crash_after_tool == state["step_number"]:
                raise RuntimeError(
                    "SIMULATED CRASH AFTER CHECKPOINT"
                )

    return _fallback(
        state, "FALLBACK_MAX_ITERATIONS"
    )


# ==========================================
# STEP 5: START AN AGENT RUN
# ==========================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    *,
    run_id=None,
    crash_after_tool=None
):
    """Start a fresh refund agent run."""

    run_id = run_id or scenario_id

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,

        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT. Follow refund policy using tools."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,

        "elapsed_seconds": 0.0,
        "llm_calls": 0,
        "tokens_used": 0,
        "step_number": 0,

        "steps": [],
        "tools_executed": []
    }

    return _continue(
        state,
        crash_after_tool=crash_after_tool
    )


# ==========================================
# STEP 6: RESUME AFTER A CRASH
# ==========================================

def resume_agent(run_id):
    """Continue a saved run without replaying earlier tools."""

    path = _checkpoint_path(run_id)

    state = json.loads(
        path.read_text(encoding="utf-8")
    )

    return _continue(state)


# ==========================================
# STEP 7: TEST THE FIRST REFUND
# ==========================================

if __name__ == "__main__":

    path = ROOT / "data/raw/agent_scenarios.csv"

    with path.open(
        newline="",
        encoding="utf-8"
    ) as file:
        scenario = next(csv.DictReader(file))

    result = run_agent(
        scenario["message"],
        scenario["scenario_id"]
    )

    print(json.dumps(result, indent=2))
