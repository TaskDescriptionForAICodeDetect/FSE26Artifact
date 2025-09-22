#!/usr/bin/env python3
import sys
import os
import json
import concurrent.futures
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm

from baseutils.ioutils import get_all_solutions, Solution
from baseutils.stasticutils import evaluate_within_across_threshold
from CodeRewrite.crUtil import (
    CodeEmbedder,
    similarity_score,
    rewrite_n_times,
)


CONFIG: Dict[str, Any] = {
    "root_path": str("../splitDescription/split_benchmark"),
    "rewrite_base_dir": str(Path("./rewrites_cr_ablation").resolve()),
    "detector_llm": "gpt-4o",
    "embed_model": "microsoft/graphcodebert-base",
    "n_rewrite": 4,
    "thr_step": 0.01,
    "temperature": 1.0,
    "max_retry": 3,
    "rewrite_max_workers": 4,
    "with_real_task": True,
}



def load_task_parts(root_path: Path, task_name: str) -> Dict[str, str]:
    task_dir = root_path / "tasks" / task_name
    desc_json = task_dir / "description.json"
    desc_txt = task_dir / "description.txt"
    parts = {"task_core": "", "constraint": "", "io_requirement": "", "example": ""}
    if desc_json.exists():
        try:
            data = json.loads(desc_json.read_text(encoding="utf-8", errors="ignore"))
            for k in parts.keys():
                v = data.get(k, "")
                if isinstance(v, str):
                    parts[k] = v.strip()
        except Exception:
            pass
    if not parts["task_core"] and desc_txt.exists():
        parts["task_core"] = desc_txt.read_text(encoding="utf-8", errors="ignore").strip()
    return parts


def compose_core(parts: Dict[str, str]) -> str:
    return (parts.get("task_core") or "").strip()


def compose_core_plus_io(parts: Dict[str, str]) -> str:
    core = (parts.get("task_core") or "").strip()
    io = (parts.get("io_requirement") or "").strip()
    if not io:
        return core
    glue = "\n\nNote: The following is the IO requirement.\n"
    return f"{core}{glue}{io}"


def compose_core_plus_example(parts: Dict[str, str]) -> str:
    core = (parts.get("task_core") or "").strip()
    ex = (parts.get("example") or "").strip()
    if not ex:
        return core
    glue = "\n\nNote: The following is an example.\n"
    return f"{core}{glue}{ex}"


def prepare_rewrites_for_solutions(solutions: List[Solution], config: Dict[str, Any]):
    print("\n" + "="*20 + f" Phase 1 (tag={config['tag']}): Preparing Rewrites " + "="*20)

    rewrite_cache_dir: Path = Path(config['rewrite_cache_dir'])
    rewrite_cache_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[tuple, List[Solution]] = defaultdict(list)
    for s in solutions:
        label_str = "ai" if s.label == 1 else "human"
        grouped[(s.task_name, label_str)].append(s)

    for (task_name, label_str), group in tqdm(grouped.items(), desc="Processing Task Groups"):
        task_cache_dir = rewrite_cache_dir / task_name
        task_cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = task_cache_dir / f"{task_name}_{label_str}.json"

        if json_path.exists():
            try:
                cache_data = json.loads(json_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                cache_data = {}
        else:
            cache_data = {}

        if not cache_data:
            cache_data = {
                "task_name": task_name,
                "task_description": group[0].description,
                "code_rewrites": []
            }

        lang_model_to_entry: Dict[tuple, Dict[str, Any]] = {}
        representative: Dict[tuple, Solution] = {}
        for sol in group:
            key = (sol.language, getattr(sol, "model", None))
            if key in lang_model_to_entry:
                continue
            entry = None
            for e in cache_data.get("code_rewrites", []):
                if e.get("original_language") == sol.language and e.get("original_model") == getattr(sol, "model", None):
                    entry = e
                    break
            if not entry:
                entry = {
                    "original_code": sol.code,
                    "original_language": sol.language,
                    "original_label": sol.label,
                    "original_model": getattr(sol, "model", None),
                    "rewrites_list": []
                }
                cache_data["code_rewrites"].append(entry)
            lang_model_to_entry[key] = entry
            representative[key] = sol

        jobs: List[tuple] = []
        for key, entry in lang_model_to_entry.items():
            existing = [
                r for r in entry.get("rewrites_list", [])
                if r.get("model_name") == config["detector_llm"] and r.get("with_real_task") == config["with_real_task"]
            ]
            need = config["n_rewrite"] - len(existing)
            if need <= 0:
                continue
            sol = representative[key]
            print(f"Generating {need} rewrites for {sol.task_name} ({label_str}, {sol.language}, origin={getattr(sol,'model',None)})...")
            jobs.append((key, sol, need))

        if jobs:
            max_workers = int(config.get("rewrite_max_workers", 4))
            print(f"Submitting {len(jobs)} rewrite jobs with max_workers={max_workers} ...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {
                    executor.submit(rewrite_n_times, sol, config, need): key
                    for (key, sol, need) in jobs
                }
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        codes = future.result()
                    except Exception as e:
                        print(f"  [WARN] rewrite job failed for key={key}: {e}")
                        codes = []
                    entry = lang_model_to_entry[key]
                    sol = representative[key]
                    entry.setdefault("rewrites_list", [])
                    entry["rewrites_list"].extend([
                        {
                            "rewritten_code": c,
                            "model_name": config["detector_llm"],
                            "rewritten_language": sol.language,
                            "with_real_task": config["with_real_task"],
                        }
                        for c in codes
                    ])

        json_path.write_text(json.dumps(cache_data, indent=4), encoding='utf-8')


def run_detection_for_solutions(solutions: List[Solution], config: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "="*20 + f" Phase 2 (tag={config['tag']}): Running Detection " + "="*20)

    all_pass = True
    missing: List[str] = []

    for s in tqdm(solutions, desc="Checking Cached Rewrites"):
        label_str = "ai" if s.label == 1 else "human"
        cache_dir = Path(config['rewrite_cache_dir']) / s.task_name
        json_path = cache_dir / f"{s.task_name}_{label_str}.json"
        has_match = False
        if json_path.exists():
            try:
                cache = json.loads(json_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                cache = {}
            entries = cache.get('code_rewrites', [])
            for e in entries:
                if e.get('original_language') == s.language and e.get('original_model') == getattr(s, 'model', None):
                    rewrites = [r for r in e.get('rewrites_list', []) if r.get('model_name') == config['detector_llm'] and r.get('with_real_task') == config['with_real_task']]
                    if rewrites:
                        has_match = True
                    break
        if not has_match:
            all_pass = False
            missing.append(f"{s.task_name}:{label_str}:{s.language}")

    if not all_pass and missing:
        print(f"[CR-Ablation] Missing rewrites for {len(missing)} solutions (showing up to 20):")
        for x in missing[:20]:
            print(f"  - {x}")
    return {"solutions_count": len(solutions), "all_pass": bool(all_pass)}


def main():
    root_path = Path(CONFIG["root_path"]).resolve()
    base_rewrite_dir = Path(CONFIG["rewrite_base_dir"]).resolve()
    print(f"root_path: {root_path}")
    print(f"rewrite_base_dir: {base_rewrite_dir}")

    solutions = get_all_solutions(str(root_path))

    task_parts: Dict[str, Dict[str, str]] = {}
    for s in tqdm(solutions, desc="Loading parts per task"):
        if s.task_name not in task_parts:
            task_parts[s.task_name] = load_task_parts(root_path, s.task_name)

    def build_descs(compose_fn):
        return [compose_fn(task_parts[s.task_name]) for s in solutions]

    desc_core = build_descs(compose_core)
    desc_core_io = build_descs(compose_core_plus_io)
    desc_core_example = build_descs(compose_core_plus_example)
    desc_origin = [s.description for s in solutions]

    runs = {
        "core_only": desc_core,
        "core_plus_io": desc_core_io,
        "core_plus_example": desc_core_example,
        "origin": desc_origin,
    }

    for tag, desc_list in runs.items():
        modified_solutions: List[Solution] = []
        for s, new_desc in zip(solutions, desc_list):
            modified = replace(s, description=new_desc)
            modified_solutions.append(modified)

        rewrite_dir = base_rewrite_dir / tag
        tag_config = {**CONFIG, "tag": tag, "rewrite_cache_dir": str(rewrite_dir)}

        prepare_rewrites_for_solutions(modified_solutions, tag_config)
        run_detection_for_solutions(modified_solutions, tag_config)


if __name__ == "__main__":
    main()

