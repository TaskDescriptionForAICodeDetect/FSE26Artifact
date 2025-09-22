import sys
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from baseutils.ioutils import get_all_solutions, Solution
from CodeRewrite.crUtil import (
    CodeEmbedder, 
    similarity_score, 
    rewrite_n_times,
    build_rewrite_prompt,
)


def get_solution_entry_by_lang_model(cache_data: Dict, language: str, original_model: str | None) -> Dict | None:
    entries = cache_data.get("code_rewrites", [])
    for entry in entries:
        if entry.get("original_language") == language and entry.get("original_model") == original_model:
            return entry
    return None


def prepare_rewrites(config: Dict[str, Any]):
    print("\n" + "="*20 + " Phase 1: Preparing Rewrites " + "="*20)

    solutions = get_all_solutions(config['root_path'])

    config["rewrite_cache_dir"].mkdir(parents=True, exist_ok=True)
    grouped_solutions: Dict[tuple, List[Solution]] = defaultdict(list)
    for s in solutions:
        label_str = "ai" if s.label == 1 else "human"
        grouped_solutions[(s.task_name, label_str)].append(s)

    def process_group(task_name: str, label_str: str, solution_group: List[Solution]):
        task_cache_dir = config["rewrite_cache_dir"] / task_name
        task_cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = task_cache_dir / f"{task_name}_{label_str}.json"

        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    cache_data = json.load(f)
                except json.JSONDecodeError:
                    cache_data = {}
        else:
            cache_data = {}

        if not cache_data:
            cache_data = {
                "task_name": task_name,
                "task_description": solution_group[0].description,
                "code_rewrites": []
            }

        lang_model_to_entry: Dict[tuple, Dict[str, Any]] = {}
        representative_solution_by_lang_model: Dict[tuple, Solution] = {}

        for solution in solution_group:
            key = (solution.language, getattr(solution, "model", None))
            if key not in lang_model_to_entry:
                entry = get_solution_entry_by_lang_model(cache_data, solution.language, getattr(solution, "model", None))
                if not entry:
                    entry = {
                        "original_code": solution.code,
                        "original_language": solution.language,
                        "original_label": solution.label,
                        "original_model": getattr(solution, "model", None),
                        "rewrites_list": []
                    }
                    cache_data["code_rewrites"].append(entry)
                lang_model_to_entry[key] = entry
                representative_solution_by_lang_model[key] = solution

        for (language, original_model), entry in lang_model_to_entry.items():
            existing_rewrites = [
                r for r in entry["rewrites_list"]
                if r.get("model_name") == config["detector_llm"] and
                   r.get("with_real_task") == config["with_real_task"]
            ]
            needed_n = config["n_rewrite"] - len(existing_rewrites)
            if needed_n <= 0:
                continue

            sol = representative_solution_by_lang_model[(language, original_model)]
            print(f"\nGenerating {needed_n} rewrites for {sol.task_name} ({label_str}, {language}, origin={original_model})...")
            new_rewrites = rewrite_n_times(sol, config, n_needed=needed_n)

            new_rewrites_data = [
                {
                    "rewritten_code": code,
                    "model_name": config["detector_llm"],
                    "rewritten_language": language,
                    "with_real_task": config["with_real_task"]
                } for code in new_rewrites
            ]
            entry["rewrites_list"].extend(new_rewrites_data)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4)
        return json_path

    for (task_name, label_str), solution_group in tqdm(grouped_solutions.items(), desc="Processing Task Groups"):
        process_group(task_name, label_str, solution_group)

    sol_count = len(solutions)
    if sol_count > 0:
        pass
    return {
        "solutions_count": sol_count,
    }


def run_detection(config: Dict[str, Any]) -> Dict:
    print("\n" + "="*20 + " Phase 2: Running Detection " + "="*20)
    
    solutions = get_all_solutions(config['root_path'])

    all_pass = True
    missing: List[str] = []

    for solution in tqdm(solutions, desc="Checking Cached Rewrites"):
        label_str = "ai" if solution.label == 1 else "human"
        task_cache_dir = config["rewrite_cache_dir"] / solution.task_name
        json_path = task_cache_dir / f"{solution.task_name}_{label_str}.json"

        has_match = False
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    cache_data = json.load(f)
                except json.JSONDecodeError:
                    cache_data = {}
            solution_entry = get_solution_entry_by_lang_model(cache_data, solution.language, getattr(solution, "model", None))
            if solution_entry:
                relevant_rewrites = [
                    r for r in solution_entry.get("rewrites_list", [])
                    if r.get("model_name") == config.get("detector_llm") and r.get("with_real_task") == config.get("with_real_task")
                ]
                has_match = len(relevant_rewrites) > 0
        if not has_match:
            all_pass = False
            missing.append(f"{solution.task_name}:{label_str}:{solution.language}")

    if not all_pass and missing:
        print(f"[CodeRewrite] Missing rewrites for {len(missing)} solutions (showing up to 20):")
        for x in missing[:20]:
            print(f"  - {x}")

    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        "solutions_count": int(len(solutions)),
        "all_pass": bool(all_pass),
    }


def run_all_experiments():
    
    base_config = {
        "root_path": Path("../splitDescription/split_benchmark"),
        "rewrite_cache_dir": Path("./rewrites"),
        "detector_llm": "gpt-4o",
        "embed_model": "./gcdbt",
        "n_rewrite": 4,
        "thr_step": 0.01,
        "temperature": 0.5,
    }

    experiment_configs = [
        {"with_real_task": False},
    ]

    all_results = []
    for exp_config in experiment_configs:
        config = {**base_config, **exp_config}
        
        print("\n" + "#"*70)
        print("### Starting New Experiment ###")
        for key, val in config.items():
             print(f"  - {key}: {val}")
        print("#"*70 + "\n")
        
        prep_stats = prepare_rewrites(config)
        result = run_detection(config)
        try:
            sol_count = prep_stats.get("solutions_count", 0)
            if sol_count > 0:
                pass
        except Exception:
            pass
        all_results.append(result)

    summary_path = "coderewrite_experiment_summary.json"
    print(f"\n{'='*30}\nExperiment summary saved to -> {summary_path}\n{'='*30}")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    run_all_experiments()
