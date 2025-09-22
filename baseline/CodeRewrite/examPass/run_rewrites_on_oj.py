#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Dict, List
from tqdm import tqdm
import sys, os
# Ensure project root is on sys.path so 'CodeRewrite' is importable when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from CodeRewrite.examPass.judge import judge_solution
import traceback


CONFIG = {
    # Where cached rewrites are stored
    "rewrites_dir": str(Path(__file__).parent.parent / "rewrites"),
    # Filters for selecting rewrites
    "model_name": "gpt-4o",
    "with_real_task": True,
    "rewritten_language": "c++",  # python | c | cpp | c++ | java
    # Optional filters
    "task_filter": None,              # substring inside task directory name
    "label_filter": None,             # "ai" | "human" | None
    # Judge configuration
    # If True, try to load per-task tests at split_benchmark_root/tasks/<task_name>/tests/tests.json
    "use_task_tests": True,
    # Root of split benchmark; default points to repo root/splitDescription/split_benchmark
    "split_benchmark_root": str(Path(__file__).resolve().parents[2] / "splitDescription" / "split_benchmark"),
    # Fallback global testcases file (optional). If not found and use_task_tests fails, the task is skipped.
    "testcases": None,
    "limit": 1000,
    "time_limit": 2.0,
    "compare_mode": "strip_punct",         # exact | strip | token
    # Sampling when many cases
    "random_sample_max": 5,
    "random_state": 42,
    # Output json file path for full details
    "output_file": str(Path(__file__).parent / "analysis_outputs" / "oj_full_summary.json"),
    # Where to save compile-failed workdir paths
    "compile_failed_list_file": str(Path(__file__).parent / "analysis_outputs" / "compile_failed_workdirs.txt"),
    # Exclude solutions whose original language is in this list (lowercased). These solutions' rewrites will be skipped.
    # Example: ["java", "python"]
    "exclude_original_languages": [],
}


def load_testcases(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        # try common wrappers
        if isinstance(data, dict):
            for key in ("tests", "cases", "samples", "examples"):
                lst = data.get(key)
                if isinstance(lst, list):
                    data = lst
                    break
        if not isinstance(data, list):
            raise ValueError("testcases JSON must be a list (or contain a list under tests/cases/samples) of {input, output}")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid testcase at index {i}")
        # normalize common field aliases
        if "input" not in item and "in" in item:
            item["input"] = item.get("in", "")
        if "output" not in item and "out" in item:
            item["output"] = item.get("out", "")
        if "input" not in item or "output" not in item:
            raise ValueError(f"Invalid testcase at index {i}: missing input/output")
    return data
def load_task_testcases(split_root: Path, task_name: str) -> List[Dict[str, str]] | None:
    """Load tests for a given task from split_benchmark_root/tasks/<task_name>/tests/tests.json.

    Returns None if not found or invalid.
    """
    tests_dir = split_root / "tasks" / task_name / "tests"
    for fname in ("tests.json", "test.json"):
        test_path = tests_dir / fname
        if not test_path.exists():
            continue
        try:
            return load_testcases(test_path)
        except Exception as e:
            print(f"[tests] failed to load {test_path}:", e)
            traceback.print_exc()
            continue
    return None



def iter_rewrites(
    rewrites_dir: Path,
    model_name: str,
    with_real_task: bool,
    rewritten_language: str,
    exclude_original_languages: List[str] | None = None,
    task_filter: str | None = None,
    label_filter: str | None = None,
):
    lang_key = (rewritten_language or "").strip().lower()
    results = []
    exclude_set = set((exclude_original_languages or []))

    for task_dir in sorted(p for p in rewrites_dir.iterdir() if p.is_dir()):
        if task_filter and task_filter not in task_dir.name:
            continue
        if "codenet" not in str(task_dir):
            continue
        for json_path in sorted(task_dir.glob("*.json")):
            name = json_path.name
            label = "ai" if name.endswith("_ai.json") else ("human" if name.endswith("_human.json") else None)
            if label_filter and label != label_filter:
                continue
            try:
                cache = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[rewrites] failed to read {json_path}:", e)
                traceback.print_exc()
                continue
            for entry in cache.get("code_rewrites", []):
                orig_lang = (entry.get("original_language") or "").lower()
                if orig_lang in exclude_set:
                    continue
                orig_code = entry.get("original_code", "")
                for r in entry.get("rewrites_list", []):
                    if (
                        (r.get("model_name") == model_name)
                        and (bool(r.get("with_real_task")) == with_real_task)
                        and ((r.get("rewritten_language") or "").lower() == lang_key)
                    ):
                        results.append((cache.get("task_name"), label, orig_lang, r.get("rewritten_code", ""), orig_code))
    return results


def main():
    cfg = CONFIG
    rewrites_dir = Path(cfg["rewrites_dir"]).resolve()
    with_real_task = bool(cfg["with_real_task"])
    split_root = Path(cfg["split_benchmark_root"]).resolve()
    global_cases = None
    if cfg.get("testcases"):
        tc_path = Path(cfg["testcases"]).resolve()
        if tc_path.exists():
            global_cases = load_testcases(tc_path)

    results = []
    count = 0
    # Cache task->cases to avoid repeated loads
    task_cases_cache: dict[str, List[Dict[str, str]] | None] = {}
    rng = random.Random(int(cfg.get("random_state", 42)))
    all_rewrites = iter_rewrites(
        rewrites_dir=rewrites_dir,
        model_name=cfg["model_name"],
        with_real_task=with_real_task,
        rewritten_language=cfg["rewritten_language"],
        exclude_original_languages=[(s or "").lower() for s in cfg.get("exclude_original_languages", [])],
        task_filter=cfg.get("task_filter"),
        label_filter=cfg.get("label_filter"),
    )
    # deterministic workdir naming counter per (task, rewritten_language, model, label)
    name_counter: Dict[str, int] = {}
    compile_failed_paths: List[str] = []
    for task_name, label, orig_lang, code, origin_code in tqdm(all_rewrites):
        # Resolve cases for this task
        if task_name not in task_cases_cache:
            cases_for_task = None
            test_source = "none"
            test_path_str = None
            if cfg.get("use_task_tests", True):
                cases_for_task = load_task_testcases(split_root, task_name)
                if cases_for_task is not None:
                    test_source = "task_tests"
                    test_path_str = str(split_root / "tasks" / task_name / "tests" / "tests.json")
            if cases_for_task is None and global_cases is not None:
                cases_for_task = global_cases
                test_source = "config_testcases"
                test_path_str = str(Path(cfg["testcases"]).resolve()) if cfg.get("testcases") else None
            task_cases_cache[task_name] = cases_for_task
            # Also cache meta so we can report
            task_cases_cache[task_name + "__meta__"] = (test_source, test_path_str)

        cases = task_cases_cache[task_name]
        test_source, test_path_str = task_cases_cache.get(task_name + "__meta__", ("none", None))
        if not cases:
            # Skip this task if no tests available
            results.append({
                "task_name": task_name,
                "label": label,
                "original_language": orig_lang,
                "rewritten_language": cfg["rewritten_language"],
                "model_name": cfg["model_name"],
                "with_real_task": with_real_task,
                "skipped": True,
                "reason": "no_testcases",
                "test_source": test_source,
                "test_path": test_path_str,
            })
            continue

        # Randomly sample up to N cases if many
        cases_to_use = cases
        max_n = int(cfg.get("random_sample_max", 10))
        if isinstance(cases, list) and len(cases) > max_n:
            # Use the first N cases deterministically instead of random sampling
            cases_to_use = cases[:max_n]

        # Build deterministic base name: task_rewrittenLanguage_originLabel_model_counter
        def _sanitize(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in (s or ""))
        key = f"{task_name}|{cfg['rewritten_language']}|{cfg['model_name']}|{label}"
        name_counter[key] = name_counter.get(key, 0) + 1
        base_name = f"{_sanitize(task_name)}_{_sanitize(cfg['rewritten_language'])}_{_sanitize(label or 'unknown')}_{_sanitize(cfg['model_name'])}_{name_counter[key]}"

        summary = judge_solution(
            code=code,
            language=cfg["rewritten_language"],
            testcases=cases_to_use,
            compare_mode=cfg["compare_mode"],
            time_limit_sec=cfg["time_limit"],
            origin_code=origin_code,
            workdir_base_name=base_name,
            write_testcase_json=True,
        )
        # Compute pass ratio and any-pass
        total_used = len(summary["cases"]) or 0
        passed_cnt = sum(1 for c in summary["cases"] if c.get("passed"))
        pass_ratio = (passed_cnt / total_used) if total_used else 0.0
        any_passed = passed_cnt > 0
        # Determine if any successful run (regardless of expected output)
        any_success = any(
            (not c.get("timed_out")) and (not c.get("compile_error")) and int(c.get("exit_code", 0)) == 0
            for c in summary["cases"]
        )
        # Classify failure reason if no successful run
        failure_reason = None
        if not any_success:
            # precedence: toolchain_missing -> compile_error -> timeout(all) -> runtime_error(all nonzero) -> unknown
            if any("Command not found" in (c.get("stderr") or "") for c in summary["cases"]):
                failure_reason = "toolchain_missing"
            elif any(c.get("compile_error") for c in summary["cases"]):
                failure_reason = "compile_error"
            elif all(c.get("timed_out") for c in summary["cases"]):
                failure_reason = "timeout"
            elif all(int(c.get("exit_code", 0)) != 0 for c in summary["cases"]):
                failure_reason = "runtime_error"
            else:
                failure_reason = "unknown"
        # collect compile-failed workdirs for this rewrite
        for c in summary.get("cases", []):
            if c.get("compile_error"):
                wd = c.get("workdir")
                if wd:
                    compile_failed_paths.append(wd)

        results.append({
            "task_name": task_name,
            "label": label,
            "original_language": orig_lang,
            "rewritten_language": cfg["rewritten_language"],
            "model_name": cfg["model_name"],
            "with_real_task": with_real_task,
            "test_source": test_source,
            "test_path": test_path_str,
            "all_passed": summary["all_passed"],
            "num_cases": summary["num_cases"],
            "num_cases_sampled": total_used,
            "pass_ratio": pass_ratio,
            "passed": any_passed,
            "any_success": any_success,
            "failure_reason": failure_reason,
            "cases": summary["cases"],
        })
        count += 1
        if count >= int(cfg["limit"]):
            break

    any_passed_count = sum(1 for r in results if r.get("passed"))
    all_passed_count = sum(1 for r in results if r.get("all_passed"))
    # Compile failure stats: no successful run
    # Count all entries without any successful run, including skipped (no testcases)
    compile_failed = [r for r in results if not r.get("any_success")]
    reason_dist: Dict[str, int] = {}
    reason_examples: Dict[str, List[str]] = {}
    import random as _rnd
    for r in compile_failed:
        reason = r.get("failure_reason") or r.get("reason") or "unknown"
        reason_dist[reason] = reason_dist.get(reason, 0) + 1
        # collect up to 10 unique workdirs for this reason
        if reason not in reason_examples:
            reason_examples[reason] = []
        # sample up to 10 examples
        candidates = []
        for c in (r.get("cases") or []):
            wd = c.get("workdir")
            if wd:
                candidates.append(wd)
        # if none, fallback to hint paths
        if not candidates:
            hint = r.get("test_path") or f"{r.get('task_name')}|{r.get('reason', 'unknown')}"
            if hint:
                candidates.append(hint)
        # shuffle and sample
        _rnd.shuffle(candidates)
        sampled = []
        for wd in candidates:
            if wd not in sampled:
                sampled.append(wd)
            if len(sampled) >= 10:
                break
        # extend reason_examples with sampled, keeping unique and cap 10 total
        for wd in sampled:
            if wd not in reason_examples[reason]:
                reason_examples[reason].append(wd)
            if len(reason_examples[reason]) >= 10:
                break

    # Overall failed distribution (sums should match `failed`): include wrong_answer
    failed_items = [r for r in results if not r.get("passed")]
    failed_reason_dist: Dict[str, int] = {}
    failed_reason_examples: Dict[str, List[str]] = {}
    for r in failed_items:
        if r.get("any_success"):
            reason = "wrong_answer"
        else:
            reason = r.get("failure_reason") or r.get("reason") or "unknown"
        failed_reason_dist[reason] = failed_reason_dist.get(reason, 0) + 1
        if reason not in failed_reason_examples:
            failed_reason_examples[reason] = []
        # sample up to 10 examples
        candidates = []
        for c in (r.get("cases") or []):
            wd = c.get("workdir")
            if wd:
                candidates.append(wd)
        if not candidates:
            hint = r.get("test_path") or f"{r.get('task_name')}|{r.get('reason', 'unknown')}"
            if hint:
                candidates.append(hint)
        _rnd.shuffle(candidates)
        sampled = []
        for wd in candidates:
            if wd not in sampled:
                sampled.append(wd)
            if len(sampled) >= 10:
                break
        for wd in sampled:
            if wd not in failed_reason_examples[reason]:
                failed_reason_examples[reason].append(wd)
            if len(failed_reason_examples[reason]) >= 10:
                break

    # Build full wrong_answer cases (per-case granularity)
    wrong_answer_cases: List[Dict] = []
    for r in results:
        if r.get("any_success") and not r.get("passed"):
            for c in (r.get("cases") or []):
                # wrong answer per-case: executed successfully but output mismatch
                if (not c.get("timed_out")) and (not c.get("compile_error")) and int(c.get("exit_code", 0)) == 0 and (not c.get("passed")):
                    wrong_answer_cases.append({
                        "task_name": r.get("task_name"),
                        "label": r.get("label"),
                        "model_name": r.get("model_name"),
                        "rewritten_language": r.get("rewritten_language"),
                        "with_real_task": r.get("with_real_task"),
                        "test_source": r.get("test_source"),
                        "test_path": r.get("test_path"),
                        "input": c.get("input"),
                        "expected": c.get("expected"),
                        "actual": (c.get("actual") or "")[:100],
                        "exit_code": c.get("exit_code"),
                        "time_ms": c.get("time_ms"),
                        "workdir": c.get("workdir"),
                    })

    # Build all failure cases by reason (including compile_error/timeout/runtime_error/toolchain_missing/unknown)
    failure_cases_by_reason: Dict[str, List[Dict]] = {}
    for r in results:
        if r.get("passed"):
            continue
        reason = ("wrong_answer" if r.get("any_success") else (r.get("failure_reason") or r.get("reason") or "unknown"))
        if reason not in failure_cases_by_reason:
            failure_cases_by_reason[reason] = []
        cases_list = r.get("cases") or []
        if cases_list:
            for c in cases_list:
                failure_cases_by_reason[reason].append({
                    "task_name": r.get("task_name"),
                    "label": r.get("label"),
                    "model_name": r.get("model_name"),
                    "rewritten_language": r.get("rewritten_language"),
                    "with_real_task": r.get("with_real_task"),
                    "test_source": r.get("test_source"),
                    "test_path": r.get("test_path"),
                    "input": c.get("input"),
                    "expected": c.get("expected"),
                    "actual": (c.get("actual") or "")[:100],
                    "stderr": c.get("stderr"),
                    "timed_out": c.get("timed_out"),
                    "compile_error": c.get("compile_error"),
                    "exit_code": c.get("exit_code"),
                    "time_ms": c.get("time_ms"),
                    "workdir": c.get("workdir"),
                })
        else:
            # no cases available (skipped/no tests) -> create a placeholder with hints
            failure_cases_by_reason[reason].append({
                "task_name": r.get("task_name"),
                "label": r.get("label"),
                "model_name": r.get("model_name"),
                "rewritten_language": r.get("rewritten_language"),
                "with_real_task": r.get("with_real_task"),
                "test_source": r.get("test_source"),
                "test_path": r.get("test_path"),
                "no_testcases": True,
            })

    output_payload = {
        "total_tested": len(results),
        "passed": any_passed_count,              # using any-pass rule
        "failed": len(results) - any_passed_count,
        "all_passed_count": all_passed_count,    # additional reference
        "compile_failed_count": len(compile_failed),
        "compile_failed_reason_distribution": reason_dist,
        "compile_failed_reason_examples": reason_examples,
        "failed_reason_distribution": failed_reason_dist,
        "failed_reason_examples": failed_reason_examples,
        "wrong_answer_cases": wrong_answer_cases,
        "failure_cases_by_reason": failure_cases_by_reason,
    }

    # Print to stdout (optional)
    # print(json.dumps(output_payload, ensure_ascii=False, indent=2))

    # Write to file if configured
    out_file = Path(cfg.get("output_file", "")).resolve() if cfg.get("output_file") else None
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] Full summary saved to: {out_file}")

    # Write compile-failed workdir list
    fail_list_path = Path(cfg.get("compile_failed_list_file", "")).resolve() if cfg.get("compile_failed_list_file") else None
    if fail_list_path:
        fail_list_path.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        ordered = []
        for p in compile_failed_paths:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        fail_list_path.write_text("\n".join(ordered), encoding="utf-8")
        print(f"[write] Compile-failed workdirs saved to: {fail_list_path}")


if __name__ == "__main__":
    main()

