
"""Part D: Evaluate refund scenarios, safety fallbacks and crash recovery."""

import csv
import json
from math import isclose

from refund_agent import (
    LEDGER,
    ROOT,
    STATE,
    resume_agent,
    run_agent,
)

RUNS_FILE = ROOT / "outputs" / "agent_runs.jsonl"
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"


# STEP 1: Get the refund amount recorded by the agent
def refund_amount(result):

    for step in reversed(result["steps"]):

        if step["tool"] in (
            "issue_refund",
            "escalate_to_human"
        ):
            return float(
                step["observation"]["refund_inr"]
            )

    return 0.0


# STEP 2: Compare result against expected outcome
def validate_scenario(scenario, result):

    expected_status = scenario["expected_outcome"]

    if expected_status == "FALLBACK":
        expected_status = "FALLBACK_MAX_ITERATIONS"

    assert result["status"] == expected_status, (
        f"{scenario['scenario_id']}: "
        f"{result['status']} != {expected_status}"
    )

    expected_amount = float(
        scenario["expected_refund_inr"]
    )

    actual_amount = refund_amount(result)

    assert isclose(
        actual_amount,
        expected_amount,
        abs_tol=0.01
    )

    tools = [
        step["tool"]
        for step in result["steps"]
    ]

    if expected_status == "REFUND":
        assert tools[-1] == "issue_refund"

    elif expected_status == "ESCALATE":
        assert tools[-1] == "escalate_to_human"
        assert "issue_refund" not in tools

    else:
        assert "issue_refund" not in tools
        assert "escalate_to_human" not in tools

    return actual_amount


# STEP 3: Run all seven scenarios
def main():

    with SCENARIOS_FILE.open(
        newline="",
        encoding="utf-8"
    ) as file:

        scenarios = list(csv.DictReader(file))

    assert len(scenarios) == 7

    RUNS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    STATE.mkdir(
        parents=True,
        exist_ok=True
    )

    # Reset previous simulated run logs
    RUNS_FILE.write_text("", encoding="utf-8")
    LEDGER.write_text("", encoding="utf-8")

    print("\n===== SEVEN REFUND SCENARIOS =====")

    print(
        f"{'ID':<5} {'STATUS':<26} "
        f"{'INR':>10} {'CALLS':>7} {'TOKENS':>8}"
    )

    with RUNS_FILE.open(
        "a",
        encoding="utf-8"
    ) as output:

        for scenario in scenarios:

            sid = scenario["scenario_id"]

            result = run_agent(
                scenario["message"],
                sid,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
                run_id=f"main_{sid}"
            )

            amount = validate_scenario(
                scenario,
                result
            )

            assert result["tokens_used"] <= 6000

            if sid == "S1":
                assert result["llm_calls"] == 5

            if sid == "S7":
                assert result["llm_calls"] == 6

            # Save full model and tool execution trace
            record = {
                "scenario_id": sid,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"]
            }

            output.write(
                json.dumps(record) + "\n"
            )

            print(
                f"{sid:<5} "
                f"{result['status']:<26} "
                f"{amount:>10.2f} "
                f"{result['llm_calls']:>7} "
                f"{result['tokens_used']:>8}"
            )

    # STEP 4: Token-limit safety demonstration
    s1 = scenarios[0]

    token_demo = run_agent(
        s1["message"],
        "S1",
        run_id="demo_token_S1",
        token_limit=200
    )

    assert (
        token_demo["status"]
        == "FALLBACK_TOKEN_LIMIT"
    )

    assert (
        "Human handoff"
        in token_demo["final_answer"]
    )

    # STEP 5: Timeout safety demonstration
    timeout_demo = run_agent(
        s1["message"],
        "S1",
        run_id="demo_timeout_S1",
        time_limit_seconds=0
    )

    assert (
        timeout_demo["status"]
        == "FALLBACK_TIMEOUT"
    )

    assert timeout_demo["llm_calls"] == 0
    assert timeout_demo["steps"] == []

    print("\n===== STOP-SAFELY DEMONSTRATIONS =====")

    print(
        "S1, token_limit=200:",
        token_demo["status"]
    )

    print(
        "S1, time_limit_seconds=0:",
        timeout_demo["status"]
    )

    # STEP 6: Simulate crash after second tool
    crash_run_id = "demo_crash_S1"

    try:

        run_agent(
            s1["message"],
            "S1",
            run_id=crash_run_id,
            crash_after_tool=2
        )

    except RuntimeError as error:

        assert (
            str(error)
            == "SIMULATED CRASH AFTER CHECKPOINT"
        )

    else:
        raise AssertionError(
            "Crash simulation did not occur"
        )

    # STEP 7: Verify saved checkpoint
    checkpoint = json.loads(
        (
            STATE / f"{crash_run_id}.json"
        ).read_text(encoding="utf-8")
    )

    assert checkpoint["tools_executed"] == [
        "get_order",
        "search_policy"
    ]

    # STEP 8: Resume without repeating previous tools
    resumed = resume_agent(crash_run_id)

    recovered_tools = [
        step["tool"]
        for step in resumed["steps"]
    ]

    assert recovered_tools == [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund"
    ]

    assert resumed["status"] == "REFUND"

    assert isclose(
        refund_amount(resumed),
        128.25,
        abs_tol=0.01
    )

    assert resumed["llm_calls"] == 5

    print("\n===== CRASH AND RESUME =====")

    print(
        "Tools executed once each:",
        " -> ".join(recovered_tools)
    )

    print(
        "Resumed status:",
        resumed["status"]
    )

    print(
        f"Resumed refund: INR "
        f"{refund_amount(resumed):.2f}"
    )

    # STEP 9: Validate generated logs
    records = [
        json.loads(line)
        for line in RUNS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 7

    ledger_records = [
        json.loads(line)
        for line in LEDGER.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert [
        record["tool"]
        for record in ledger_records
    ] == [
        "issue_refund",
        "issue_refund",
        "escalate_to_human",
        "issue_refund"
    ]

    print("\nPART D AGENT DEMONSTRATIONS COMPLETED")

    print(
        "Saved:",
        RUNS_FILE.relative_to(ROOT)
    )

    print(
        "Saved:",
        LEDGER.relative_to(ROOT)
    )

    print(
        "Checkpoints:",
        STATE.relative_to(ROOT)
    )


# STEP 10: Execute
if __name__ == "__main__":
    main()
