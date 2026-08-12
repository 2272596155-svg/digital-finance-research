#!/usr/bin/env python3
"""Reproducible DeepSeek prompt benchmark for recruitment extraction.

The API key is read only from DEEPSEEK_API_KEY. It is never written to logs or files.
"""

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

REQUIRED_SCALARS = ["job_title", "salary", "location", "education", "experience"]
REQUIRED_LISTS = ["hard_skills", "soft_skills", "ai_stack"]


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the three-prompt recruitment extraction benchmark.")
    parser.add_argument("--input", default=str(here / "recruitment_prompt_input.json"))
    parser.add_argument("--output-dir", default=str(here / "runs"))
    parser.add_argument("--mode", choices=["preflight", "benchmark", "full"], default="preflight")
    parser.add_argument("--prompt-id", default="P3", help="Prompt used in full mode after benchmark selection.")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    return parser.parse_args()


def canonical_text(value):
    if value is None:
        return ""
    return re.sub(r"[\s\u3000,，。；;:：()（）【】\[\]·\-_/]+", "", str(value)).lower()


def canonical_list(values):
    if not isinstance(values, list):
        return set()
    return {canonical_text(value) for value in values if canonical_text(value)}


def strip_json_fence(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_output(obj):
    errors = []
    if not isinstance(obj, dict):
        return ["root_not_object"]
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


def scalar_accuracy(pred, ref):
    matches = []
    for field in REQUIRED_SCALARS:
        matches.append(canonical_text(pred.get(field)) == canonical_text(ref.get(field)))
    return sum(matches) / len(matches)


def f1(predicted, reference):
    pred = canonical_list(predicted)
    ref = canonical_list(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def list_f1(pred, ref):
    return statistics.fmean(f1(pred.get(field, []), ref.get(field, [])) for field in REQUIRED_LISTS)


def hallucination_counts(pred, ref, source_text):
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
        for value in pred.get(field, []) if isinstance(pred.get(field), list) else []:
            atom = canonical_text(value)
            if not atom:
                continue
            predicted_atoms += 1
            if atom not in ref_values and atom not in source:
                unsupported += 1
    return unsupported, predicted_atoms


def output_atom_set(obj):
    atoms = set()
    for field in REQUIRED_SCALARS:
        value = canonical_text(obj.get(field))
        if value:
            atoms.add(f"{field}:{value}")
    for field in REQUIRED_LISTS:
        atoms.update(f"{field}:{value}" for value in canonical_list(obj.get(field, [])))
    return atoms


def jaccard(left, right):
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def call_deepseek(api_key, base_url, model, prompt, source_text, temperature, max_tokens):
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
    last_error = None
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
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek request failed after retries: {type(last_error).__name__}: {last_error}")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_preflight(data, output_dir, model):
    prompts = data.get("prompts", [])
    jobs = data.get("jobs", [])
    benchmark_ids = set(data.get("benchmark_job_ids", []))
    prompt_ids = [prompt.get("prompt_id") for prompt in prompts]
    checks = [
        {"check": "input_jobs", "expected": 128, "actual": len(jobs), "pass": len(jobs) == 128},
        {"check": "benchmark_jobs", "expected": 24, "actual": len(benchmark_ids), "pass": len(benchmark_ids) == 24},
        {"check": "prompt_ids", "expected": ["P1", "P2", "P3"], "actual": prompt_ids, "pass": prompt_ids == ["P1", "P2", "P3"]},
        {"check": "unique_job_ids", "expected": len(jobs), "actual": len({job.get('job_id') for job in jobs}), "pass": len(jobs) == len({job.get('job_id') for job in jobs})},
        {"check": "api_key_env", "expected": "configured", "actual": "configured" if os.getenv("DEEPSEEK_API_KEY") else "missing", "pass": bool(os.getenv("DEEPSEEK_API_KEY"))},
    ]
    report = {
        "status": "READY" if all(item["pass"] for item in checks) else "BLOCKED_API_KEY" if all(item["pass"] for item in checks[:-1]) else "INVALID_INPUT",
        "model": model,
        "api_key_env": "DEEPSEEK_API_KEY",
        "key_value_logged": False,
        "checks": checks,
        "note": "预检不发起网络请求；模型运行前必须在执行环境配置DEEPSEEK_API_KEY。",
    }
    write_json(output_dir / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"READY", "BLOCKED_API_KEY"} else 2


def summarize_runs(runs, repeats):
    grouped_outputs = defaultdict(list)
    for run in runs:
        if run.get("schema_compliant") and run.get("parsed_output") is not None:
            grouped_outputs[(run["prompt_id"], run["job_id"])].append(run["parsed_output"])
    stability_by_prompt = defaultdict(list)
    for (prompt_id, _), outputs in grouped_outputs.items():
        if len(outputs) < 2:
            continue
        pair_scores = [jaccard(output_atom_set(a), output_atom_set(b)) for a, b in itertools.combinations(outputs, 2)]
        stability_by_prompt[prompt_id].append(statistics.fmean(pair_scores))

    summary = []
    prompt_ids = sorted({run["prompt_id"] for run in runs})
    successful_runtime = {prompt_id: [run["runtime_seconds"] for run in runs if run["prompt_id"] == prompt_id and run.get("status") == "成功"] for prompt_id in prompt_ids}
    prompt_runtime = {prompt_id: statistics.fmean(values) if values else None for prompt_id, values in successful_runtime.items()}
    best_runtime = min((value for value in prompt_runtime.values() if value is not None), default=None)
    for prompt_id in prompt_ids:
        subset = [run for run in runs if run["prompt_id"] == prompt_id]
        scalar = statistics.fmean(run.get("scalar_accuracy", 0) for run in subset)
        list_score = statistics.fmean(run.get("list_f1", 0) for run in subset)
        unsupported = sum(run.get("unsupported_atoms", 0) for run in subset)
        atoms = sum(run.get("predicted_atoms", 0) for run in subset)
        hallucination = unsupported / atoms if atoms else 0.0
        schema = sum(bool(run.get("schema_compliant")) for run in subset) / len(subset)
        stability = statistics.fmean(stability_by_prompt[prompt_id]) if stability_by_prompt[prompt_id] else None
        runtime = prompt_runtime[prompt_id]
        speed_score = best_runtime / runtime if best_runtime and runtime else 0.0
        overall = 100 * (
            0.30 * scalar + 0.20 * list_score + 0.25 * (1 - hallucination)
            + 0.15 * (stability or 0) + 0.05 * schema + 0.05 * speed_score
        )
        summary.append({
            "prompt_id": prompt_id,
            "runs": len(subset),
            "expected_runs": len({run["job_id"] for run in subset}) * repeats,
            "scalar_accuracy": scalar,
            "list_f1": list_score,
            "hallucination_rate": hallucination,
            "stability": stability,
            "schema_compliance": schema,
            "avg_runtime_seconds": runtime,
            "speed_score": speed_score,
            "overall_score": overall,
        })
    return sorted(summary, key=lambda row: row["overall_score"], reverse=True)


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    model = args.model or data.get("meta", {}).get("model") or "deepseek-v4-flash"
    repeats = args.repeats or (data.get("meta", {}).get("repeats", 3) if args.mode == "benchmark" else 1)
    if args.mode == "preflight":
        raise SystemExit(run_preflight(data, output_dir, model))

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not configured. Do not paste the key into chat; set it in the execution environment.")

    prompts = {prompt["prompt_id"]: prompt for prompt in data["prompts"]}
    if args.mode == "benchmark":
        selected_jobs = [job for job in data["jobs"] if job["job_id"] in set(data["benchmark_job_ids"])]
        selected_prompts = list(prompts.values())
    else:
        if args.prompt_id not in prompts:
            raise SystemExit(f"Unknown prompt id: {args.prompt_id}")
        selected_jobs = data["jobs"]
        selected_prompts = [prompts[args.prompt_id]]

    base_url = data.get("meta", {}).get("base_url", "https://api.deepseek.com")
    runs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.mode}_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for job in selected_jobs:
            for prompt in selected_prompts:
                for repeat in range(1, repeats + 1):
                    run_id = f"{job['job_id']}-{prompt['prompt_id']}-R{repeat}"
                    record = {"run_id": run_id, "job_id": job["job_id"], "prompt_id": prompt["prompt_id"], "repeat": repeat, "model": model}
                    try:
                        body, content, elapsed = call_deepseek(api_key, base_url, model, prompt, job["source_text"], args.temperature, args.max_tokens)
                        record.update({
                            "status": "成功", "runtime_seconds": elapsed,
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
                                unsupported, atoms = hallucination_counts(parsed, job["reference"], job["source_text"])
                                record["unsupported_atoms"] = unsupported
                                record["predicted_atoms"] = atoms
                        except json.JSONDecodeError as exc:
                            record.update({"schema_compliant": False, "schema_errors": [f"json_parse:{exc.msg}"], "parsed_output": None})
                    except Exception as exc:
                        record.update({"status": "失败", "error": f"{type(exc).__name__}: {exc}", "schema_compliant": False, "parsed_output": None})
                    runs.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()

    metrics = summarize_runs(runs, repeats)
    write_json(output_dir / f"{args.mode}_metrics.json", {"mode": args.mode, "model": model, "repeats": repeats, "metrics": metrics})
    print(json.dumps({"runs_file": str(raw_path), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
