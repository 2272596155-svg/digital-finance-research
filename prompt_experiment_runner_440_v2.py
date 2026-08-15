#!/usr/bin/env python3
"""DeepSeek Prompt experiment runner for the 440-row recruitment dataset.

Security guarantees:
- Reads the API key only from DEEPSEEK_API_KEY.
- Never writes or prints the key.
- Preflight never sends a network request.
- Existing run IDs are resumed and skipped to avoid duplicate billing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_SCALARS = ["job_title", "salary", "location", "education", "experience"]
REQUIRED_LISTS = ["hard_skills", "soft_skills", "ai_stack"]
EXPECTED_PROMPTS = ["P1", "P2", "P3"]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the 440-v2 recruitment Prompt experiment.")
    parser.add_argument("--input", default=str(here / "recruitment_prompt_input_440_v2.json"))
    parser.add_argument("--output-dir", default=str(here / "runs_440_v2"))
    parser.add_argument("--mode", choices=["preflight", "benchmark", "full"], default="preflight")
    parser.add_argument("--prompt-id", default=None, help="Required in full mode after selecting the winner.")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[\s\u3000,，。；;:：()（）【】\[\]·\-_/]+", "", str(value)).lower()


def canonical_list(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {canonical_text(value) for value in values if canonical_text(value)}


def strip_json_fence(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def validate_output(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["root_not_object"]
    errors: list[str] = []
    for key in REQUIRED_SCALARS:
        if key not in obj:
            errors.append(f"missing:{key}")
        elif obj[key] is not None and not isinstance(obj[key], str):
            errors.append(f"type:{key}")
    for key in REQUIRED_LISTS:
        if key not in obj:
            errors.append(f"missing:{key}")
        elif not isinstance(obj[key], list) or any(not isinstance(item, str) for item in obj[key]):
            errors.append(f"type:{key}")
    return errors


def validate_reference(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["reference_not_object"]
    errors: list[str] = []
    for key in REQUIRED_SCALARS:
        if key not in obj:
            errors.append(f"missing_reference:{key}")
        elif obj[key] is not None and not isinstance(obj[key], str):
            errors.append(f"reference_type:{key}")
    for key in REQUIRED_LISTS:
        if key not in obj:
            errors.append(f"missing_reference:{key}")
        elif not isinstance(obj[key], list) or any(not isinstance(item, str) for item in obj[key]):
            errors.append(f"reference_type:{key}")
    return errors


def scalar_accuracy(pred: dict[str, Any], ref: dict[str, Any]) -> float:
    return statistics.fmean(
        canonical_text(pred.get(field)) == canonical_text(ref.get(field))
        for field in REQUIRED_SCALARS
    )


def f1(predicted: Any, reference: Any) -> float | None:
    pred = canonical_list(predicted)
    ref = canonical_list(reference)
    if not ref:
        return None
    if not pred:
        return 0.0
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def list_f1(pred: dict[str, Any], ref: dict[str, Any]) -> float | None:
    scores = [f1(pred.get(field, []), ref.get(field, [])) for field in REQUIRED_LISTS]
    evaluable = [score for score in scores if score is not None]
    return statistics.fmean(evaluable) if evaluable else None


def list_f1_fields(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, float | None]:
    return {field: f1(pred.get(field, []), ref.get(field, [])) for field in REQUIRED_LISTS}


def hallucination_counts(pred: dict[str, Any], ref: dict[str, Any], source_text: str) -> tuple[int, int]:
    source = canonical_text(source_text)
    predicted_atoms = 0
    unsupported = 0
    for field in REQUIRED_SCALARS:
        value = pred.get(field)
        if value in (None, ""):
            continue
        predicted_atoms += 1
        pred_value = canonical_text(value)
        ref_value = canonical_text(ref.get(field))
        if not pred_value or (pred_value != ref_value and pred_value not in source):
            unsupported += 1
    for field in REQUIRED_LISTS:
        ref_values = canonical_list(ref.get(field, []))
        values = pred.get(field, []) if isinstance(pred.get(field), list) else []
        for value in values:
            atom = canonical_text(value)
            if not atom:
                continue
            predicted_atoms += 1
            if atom not in ref_values and atom not in source:
                unsupported += 1
    return unsupported, predicted_atoms


def output_atom_set(obj: dict[str, Any]) -> set[str]:
    atoms: set[str] = set()
    for field in REQUIRED_SCALARS:
        value = canonical_text(obj.get(field))
        if value:
            atoms.add(f"{field}:{value}")
    for field in REQUIRED_LISTS:
        atoms.update(f"{field}:{value}" for value in canonical_list(obj.get(field, [])))
    return atoms


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def call_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    prompt: dict[str, Any],
    source_text: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_template"].replace("{{SOURCE_TEXT}}", source_text)},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - start
            content = body["choices"][0]["message"].get("content") or ""
            return body, content, elapsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code == 429 or exc.code >= 500
            if attempt == 2 or not retryable:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek request failed after retries: {type(last_error).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
    return records


def run_preflight(data: dict[str, Any], output_dir: Path, model: str) -> int:
    meta = data.get("meta", {})
    jobs = data.get("jobs", [])
    prompts = data.get("prompts", [])
    benchmark_ids = data.get("benchmark_job_ids", [])
    benchmark_set = set(benchmark_ids)
    benchmark_jobs = [job for job in jobs if job.get("job_id") in benchmark_set]
    prompt_ids = [prompt.get("prompt_id") for prompt in prompts]
    reference_errors = sum(bool(validate_reference(job.get("reference"))) for job in jobs)
    blank_sources = sum(not str(job.get("source_text") or "").strip() for job in jobs)
    list_reference_coverage = {
        field: sum(bool(job.get("reference", {}).get(field)) for job in benchmark_jobs)
        for field in REQUIRED_LISTS
    }
    expected_jobs = int(meta.get("valid_jobs", len(jobs)))
    expected_benchmark = int(meta.get("benchmark_jobs", len(benchmark_ids)))
    expected_ai = int(meta.get("benchmark_ai_jobs", 44))
    expected_non_ai = int(meta.get("benchmark_non_ai_jobs", 44))
    checks = [
        {"check": "input_jobs", "expected": expected_jobs, "actual": len(jobs), "pass": len(jobs) == expected_jobs},
        {"check": "benchmark_jobs", "expected": expected_benchmark, "actual": len(benchmark_ids), "pass": len(benchmark_ids) == expected_benchmark},
        {"check": "benchmark_ai_jobs", "expected": expected_ai, "actual": sum(bool(job.get("ai_related")) for job in benchmark_jobs), "pass": sum(bool(job.get("ai_related")) for job in benchmark_jobs) == expected_ai},
        {"check": "benchmark_non_ai_jobs", "expected": expected_non_ai, "actual": sum(not bool(job.get("ai_related")) for job in benchmark_jobs), "pass": sum(not bool(job.get("ai_related")) for job in benchmark_jobs) == expected_non_ai},
        {"check": "prompt_ids", "expected": EXPECTED_PROMPTS, "actual": prompt_ids, "pass": prompt_ids == EXPECTED_PROMPTS},
        {"check": "unique_job_ids", "expected": len(jobs), "actual": len({job.get('job_id') for job in jobs}), "pass": len(jobs) == len({job.get('job_id') for job in jobs})},
        {"check": "unique_benchmark_ids", "expected": len(benchmark_ids), "actual": len(benchmark_set), "pass": len(benchmark_ids) == len(benchmark_set)},
        {"check": "benchmark_ids_exist", "expected": len(benchmark_ids), "actual": len(benchmark_jobs), "pass": len(benchmark_ids) == len(benchmark_jobs)},
        {"check": "reference_schema_errors", "expected": 0, "actual": reference_errors, "pass": reference_errors == 0},
        {"check": "blank_source_text", "expected": 0, "actual": blank_sources, "pass": blank_sources == 0},
        {"check": "api_key_env", "expected": "configured", "actual": "configured" if os.getenv("DEEPSEEK_API_KEY") else "missing", "pass": bool(os.getenv("DEEPSEEK_API_KEY"))},
    ]
    non_key_checks = checks[:-1]
    status = "READY" if all(item["pass"] for item in checks) else "BLOCKED_API_KEY" if all(item["pass"] for item in non_key_checks) else "INVALID_INPUT"
    report = {
        "status": status,
        "experiment_version": meta.get("version"),
        "text_basis": meta.get("text_basis"),
        "model": model,
        "api_key_env": "DEEPSEEK_API_KEY",
        "key_value_logged": False,
        "network_request_sent": False,
        "checks": checks,
        "planned_benchmark_calls": len(benchmark_ids) * len(prompts) * int(meta.get("benchmark_repeats", 3)),
        "planned_full_calls": len(jobs),
        "benchmark_list_reference_coverage": list_reference_coverage,
        "list_f1_rule": "只在参考列表非空时计算；空参考样本由幻觉率约束，不计作F1=1。",
    }
    write_json(output_dir / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status in {"READY", "BLOCKED_API_KEY"} else 2


def stability_by_prompt(runs: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("schema_compliant") and isinstance(run.get("parsed_output"), dict):
            grouped[(run["prompt_id"], run["job_id"])].append(run["parsed_output"])
    result: dict[str, list[float]] = defaultdict(list)
    for (prompt_id, _), outputs in grouped.items():
        if len(outputs) < 2:
            continue
        pairs = [jaccard(output_atom_set(a), output_atom_set(b)) for a, b in itertools.combinations(outputs, 2)]
        result[prompt_id].append(statistics.fmean(pairs))
    return result


def metric_row(subset: list[dict[str, Any]], prompt_id: str, repeats: int, stability_values: list[float]) -> dict[str, Any]:
    successful = [run for run in subset if run.get("status") == "成功"]
    scalar = statistics.fmean(run.get("scalar_accuracy", 0.0) for run in subset) if subset else 0.0
    list_values = [run.get("list_f1") for run in subset if run.get("list_f1") is not None]
    list_score = statistics.fmean(list_values) if list_values else None
    field_scores = {
        field: (
            statistics.fmean(values)
            if (values := [
                run.get("list_f1_fields", {}).get(field)
                for run in subset
                if run.get("list_f1_fields", {}).get(field) is not None
            ])
            else None
        )
        for field in REQUIRED_LISTS
    }
    unsupported = sum(int(run.get("unsupported_atoms", 0)) for run in subset)
    atoms = sum(int(run.get("predicted_atoms", 0)) for run in subset)
    schema = sum(bool(run.get("schema_compliant")) for run in subset) / len(subset) if subset else 0.0
    runtime = statistics.fmean(run["runtime_seconds"] for run in successful) if successful else None
    stability = statistics.fmean(stability_values) if stability_values else None
    return {
        "prompt_id": prompt_id,
        "runs": len(subset),
        "expected_runs": len({run["job_id"] for run in subset}) * repeats,
        "successes": len(successful),
        "failures": len(subset) - len(successful),
        "scalar_accuracy": scalar,
        "list_f1": list_score,
        "list_f1_by_field": field_scores,
        "list_f1_evaluable_runs": len(list_values),
        "list_f1_evaluable_runs_by_field": {
            field: sum(run.get("list_f1_fields", {}).get(field) is not None for run in subset)
            for field in REQUIRED_LISTS
        },
        "hallucination_rate": unsupported / atoms if atoms else 0.0,
        "stability": stability,
        "schema_compliance": schema,
        "avg_runtime_seconds": runtime,
        "prompt_tokens": sum(int(run.get("prompt_tokens") or 0) for run in subset),
        "completion_tokens": sum(int(run.get("completion_tokens") or 0) for run in subset),
    }


def summarize_runs(runs: list[dict[str, Any]], repeats: int, include_overall_score: bool) -> list[dict[str, Any]]:
    stability = stability_by_prompt(runs)
    rows: list[dict[str, Any]] = []
    for prompt_id in sorted({run["prompt_id"] for run in runs}):
        subset = [run for run in runs if run["prompt_id"] == prompt_id]
        rows.append(metric_row(subset, prompt_id, repeats, stability.get(prompt_id, [])))
    valid_runtimes = [row["avg_runtime_seconds"] for row in rows if row["avg_runtime_seconds"] is not None]
    best_runtime = min(valid_runtimes) if valid_runtimes else None
    for row in rows:
        runtime = row["avg_runtime_seconds"]
        row["speed_score"] = best_runtime / runtime if best_runtime and runtime else 0.0
        if include_overall_score and row["stability"] is not None:
            row["overall_score"] = 100 * (
                0.30 * row["scalar_accuracy"]
                + 0.20 * (row["list_f1"] if row["list_f1"] is not None else 0.0)
                + 0.25 * (1 - row["hallucination_rate"])
                + 0.15 * row["stability"]
                + 0.05 * row["schema_compliance"]
                + 0.05 * row["speed_score"]
            )
        else:
            row["overall_score"] = None
    return sorted(rows, key=lambda row: row["overall_score"] if row["overall_score"] is not None else -1, reverse=True)


def subgroup_metrics(runs: list[dict[str, Any]], repeats: int, dimension: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for value in sorted({str(run.get(dimension)) for run in runs}):
        subset = [run for run in runs if str(run.get(dimension)) == value]
        rows = summarize_runs(subset, repeats, include_overall_score=False)
        for row in rows:
            row[dimension] = value
            groups.append(row)
    return groups


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    model = args.model or meta.get("model") or "deepseek-v4-flash"
    temperature = args.temperature if args.temperature is not None else float(meta.get("temperature", 0.0))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(meta.get("max_tokens", 1200))

    if args.mode == "preflight":
        raise SystemExit(run_preflight(data, output_dir, model))

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing. Configure it in the Codex environment; never paste it into chat.")

    prompts = {prompt["prompt_id"]: prompt for prompt in data["prompts"]}
    if args.mode == "benchmark":
        repeats = args.repeats if args.repeats is not None else int(meta.get("benchmark_repeats", 3))
        benchmark_ids = set(data["benchmark_job_ids"])
        selected_jobs = [job for job in data["jobs"] if job["job_id"] in benchmark_ids]
        selected_prompts = [prompts[prompt_id] for prompt_id in EXPECTED_PROMPTS]
    else:
        repeats = args.repeats if args.repeats is not None else 1
        if repeats != 1:
            raise SystemExit("Full mode must use --repeats 1 to avoid unnecessary cost.")
        if not args.prompt_id or args.prompt_id not in prompts:
            raise SystemExit("Full mode requires --prompt-id P1, P2, or P3 after benchmark selection.")
        selected_jobs = data["jobs"]
        selected_prompts = [prompts[args.prompt_id]]

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.mode}_runs.jsonl"
    metrics_path = output_dir / f"{args.mode}_metrics.json"
    existing = load_jsonl(raw_path)
    experiment_version = str(meta.get("version"))
    if any(str(record.get("experiment_version")) != experiment_version for record in existing):
        raise SystemExit("Existing run file belongs to another experiment version. Use a new output directory.")
    if any(str(record.get("model")) != model for record in existing):
        raise SystemExit("Existing run file used another model. Use a new output directory.")
    existing_ids = {record["run_id"] for record in existing}

    planned = [
        (job, prompt, repeat, f"{experiment_version}-{job['job_id']}-{prompt['prompt_id']}-R{repeat}")
        for job in selected_jobs
        for prompt in selected_prompts
        for repeat in range(1, repeats + 1)
    ]
    new_records: list[dict[str, Any]] = []
    with raw_path.open("a", encoding="utf-8") as stream:
        for job, prompt, repeat, run_id in planned:
            if run_id in existing_ids:
                continue
            record: dict[str, Any] = {
                "run_id": run_id,
                "experiment_version": experiment_version,
                "mode": args.mode,
                "job_id": job["job_id"],
                "prompt_id": prompt["prompt_id"],
                "repeat": repeat,
                "model": model,
                "ai_related": bool(job.get("ai_related")),
                "category": job.get("category"),
                "quality": job.get("quality"),
                "source": job.get("source"),
            }
            try:
                body, content, elapsed = call_deepseek(
                    api_key, meta.get("base_url", "https://api.deepseek.com"), model, prompt,
                    job["source_text"], temperature, max_tokens,
                )
                record.update({
                    "status": "成功",
                    "runtime_seconds": elapsed,
                    "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": body.get("usage", {}).get("completion_tokens"),
                    "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
                    "raw_response": content,
                })
                try:
                    parsed = json.loads(strip_json_fence(content))
                    schema_errors = validate_output(parsed)
                    record["parsed_output"] = parsed
                    record["schema_errors"] = schema_errors
                    record["schema_compliant"] = not schema_errors
                    if not schema_errors:
                        record["scalar_accuracy"] = scalar_accuracy(parsed, job["reference"])
                        record["list_f1"] = list_f1(parsed, job["reference"])
                        record["list_f1_fields"] = list_f1_fields(parsed, job["reference"])
                        unsupported, atoms = hallucination_counts(parsed, job["reference"], job["source_text"])
                        record["unsupported_atoms"] = unsupported
                        record["predicted_atoms"] = atoms
                except json.JSONDecodeError as exc:
                    record.update({
                        "schema_compliant": False,
                        "schema_errors": [f"json_parse:{exc.msg}"],
                        "parsed_output": None,
                    })
            except Exception as exc:
                record.update({
                    "status": "失败",
                    "error": f"{type(exc).__name__}: request_failed",
                    "schema_compliant": False,
                    "parsed_output": None,
                })
            new_records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    all_records = existing + new_records
    planned_ids = {run_id for _, _, _, run_id in planned}
    relevant_records = [record for record in all_records if record.get("run_id") in planned_ids]
    overall = summarize_runs(relevant_records, repeats, include_overall_score=args.mode == "benchmark")
    metrics = {
        "experiment_version": experiment_version,
        "mode": args.mode,
        "text_basis": meta.get("text_basis"),
        "reference_type": meta.get("reference_type"),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repeats": repeats,
        "planned_runs": len(planned),
        "recorded_runs": len(relevant_records),
        "new_api_calls_this_execution": len(new_records),
        "resume_skipped_runs": len(existing_ids & planned_ids),
        "complete": len(relevant_records) == len(planned),
        "metrics": overall,
        "subgroups": {
            "ai_related": subgroup_metrics(relevant_records, repeats, "ai_related"),
            "category": subgroup_metrics(relevant_records, repeats, "category"),
            "quality": subgroup_metrics(relevant_records, repeats, "quality"),
            "source": subgroup_metrics(relevant_records, repeats, "source"),
        },
        "key_value_logged": False,
    }
    write_json(metrics_path, metrics)
    print(json.dumps({
        "status": "COMPLETE" if metrics["complete"] else "INCOMPLETE",
        "runs_file": str(raw_path),
        "metrics_file": str(metrics_path),
        "planned_runs": len(planned),
        "recorded_runs": len(relevant_records),
        "new_api_calls_this_execution": len(new_records),
        "resume_skipped_runs": metrics["resume_skipped_runs"],
        "metrics": overall,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
