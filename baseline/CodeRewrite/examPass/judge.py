#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import json

from .oj_executor import OJExecutor, ExecutionResult


def _normalize(text: str, mode: str) -> str:
    if text is None:
        return ""
    # exact: return as-is
    if mode == "exact":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "token":
        # Collapse whitespace
        return " ".join(text.strip().split())
    if mode == "strip_punct":
        # Remove commas, periods and all common whitespace characters
        # as per relaxed comparison rule 2
        # Faster translation: build table once per call
        translation_table = {ord(c): None for c in [',', '.', ' ', '\n', '\r', '\t']}
        return text.translate(translation_table)
    return text


def judge_solution(
    code: str,
    language: str,
    testcases: List[Dict[str, str]],
    compare_mode: str = "strip",
    time_limit_sec: float = 2.0,
    origin_code: str | None = None,
    workdir_base_name: str | None = None,
    write_testcase_json: bool = False,
) -> Dict:
    executor = OJExecutor()
    cases_out = []
    all_passed = True

    for idx, tc in enumerate(testcases):
        inp = tc.get("input", "")
        expected = tc.get("output", "")
        run_name = None
        if workdir_base_name:
            run_name = f"{workdir_base_name}_case{idx+1}"
        result: ExecutionResult = executor.run(code, language, stdin=inp, time_limit_sec=time_limit_sec, origin_code=origin_code, run_dir_name=run_name)
        actual = result.stdout
        if write_testcase_json and result.workdir:
            try:
                Path(result.workdir).mkdir(parents=True, exist_ok=True)
                (Path(result.workdir) / "test_case.json").write_text(json.dumps({"input": inp, "output": expected}, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        if compare_mode == "len_equal":
            passed = (
                not result.timed_out
                and result.exit_code == 0
                and len(actual or "") == len(expected or "")
            )
        else:
            passed = (
                not result.timed_out
                and result.exit_code == 0
                and _normalize(actual, compare_mode) == _normalize(expected, compare_mode)
            )
        if not passed:
            all_passed = False
        cases_out.append({
            "input": inp,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stderr": result.stderr,
            "time_ms": result.time_ms,
            "compile_error": result.compile_error,
            "workdir": result.workdir,
        })

    return {
        "all_passed": all_passed,
        "num_cases": len(testcases),
        "cases": cases_out,
    }

