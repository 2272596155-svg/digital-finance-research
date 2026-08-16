#!/usr/bin/env python3
"""Offline reproducibility checks for the 440-v2 Prompt benchmark.

This script does not call any API. It validates the deterministic input,
preflight report, summary metrics, and 792 JSONL run records.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "recruitment_prompt_input_440_v2.json"
RUNS_DIR = ROOT / "runs_440_v2"
PREFLIGHT = RUNS_DIR / "preflight_report.json"
METRICS = RUNS_DIR / "benchmark_metrics.json"
RUNS = RUNS_DIR / "benchmark_runs.jsonl"
EXPECTED_PROMPTS = {"P1", "P2", "P3"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Invalid JSONL line {line_number}: {exc.msg}") from exc
    return records


def assert_no_secret_text(paths: list[Path]) -> None:
    forbidden = ("sk-", "api_key_value", "bearer ")
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"Potential secret-like text found in {path}: {token}"


def main() -> None:
    required = [INPUT, PREFLIGHT, METRICS, RUNS]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"Missing required files: {missing}"

    data = load_json(INPUT)
    preflight = load_json(PREFLIGHT)
    metrics = load_json(METRICS)
    runs = load_jsonl(RUNS)

    jobs = data["jobs"]
    benchmark_ids = data["benchmark_job_ids"]
    prompt_ids = {item["prompt_id"] for item in data["prompts"]}
    benchmark_set = set(benchmark_ids)
    benchmark_jobs = [job for job in jobs if job["job_id"] in benchmark_set]

    assert data["meta"]["version"] == "440-v2"
    assert len(jobs) == 439
    assert len({job["job_id"] for job in jobs}) == 439
    assert len(benchmark_ids) == 88
    assert len(benchmark_set) == 88
    assert len(benchmark_jobs) == 88
    assert sum(bool(job.get("ai_related")) for job in benchmark_jobs) == 44
    assert sum(not bool(job.get("ai_related")) for job in benchmark_jobs) == 44
    assert prompt_ids == EXPECTED_PROMPTS

    assert preflight["status"] == "READY"
    assert preflight["network_request_sent"] is False
    assert preflight["key_value_logged"] is False
    assert all(item["pass"] for item in preflight["checks"])
    assert preflight["planned_benchmark_calls"] == 792

    assert metrics["experiment_version"] == "440-v2"
    assert metrics["mode"] == "benchmark"
    assert metrics["planned_runs"] == 792
    assert metrics["recorded_runs"] == 792
    assert metrics["complete"] is True
    assert metrics["key_value_logged"] is False
    assert {row["prompt_id"] for row in metrics["metrics"]} == EXPECTED_PROMPTS

    assert len(runs) == 792
    assert len({row["run_id"] for row in runs}) == 792
    assert all(row.get("status") == "成功" for row in runs)
    assert Counter(row["prompt_id"] for row in runs) == Counter({"P1": 264, "P2": 264, "P3": 264})
    assert Counter(row["repeat"] for row in runs) == Counter({1: 264, 2: 264, 3: 264})
    assert sum(bool(row.get("schema_compliant")) for row in runs) == 791
    assert sum(not bool(row.get("schema_compliant")) for row in runs) == 1

    assert_no_secret_text([PREFLIGHT, METRICS, RUNS])

    print("PASS: 440-v2 reproducibility checks completed")
    print("jobs=439 benchmark=88 runs=792 successes=792 schema_compliant=791")


if __name__ == "__main__":
    main()

