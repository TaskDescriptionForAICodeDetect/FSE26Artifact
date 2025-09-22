#!/usr/bin/env python3

import sys, os, json, pickle, hashlib
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from baseutils.ioutils import get_all_solutions, Solution
from baseutils.stasticutils import evaluate_within_across_threshold
from PERPLEXITY.utils import prepare_texts_and_indices, score_texts_by_code_part_with_checkpoint


CONFIG: Dict[str, Any] = {
    "root_path": "../../splitDescription/split_benchmark",
    "tokenizer_path": "/models/LLM-Research/Llama-3___2-3B",
    "max_tokens": 4096,
    "timeout": 120,
    "combination_method": "prompt",
    "prompt_template": None,
}


def generate_filename_prefix(tag: str) -> str:
    parts = ["ablation", tag]
    return "_".join(parts)


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


def compose_desc_core(parts: Dict[str, str]) -> str:
    return (parts.get("task_core") or "").strip()


def compose_desc_core_plus_io(parts: Dict[str, str]) -> str:
    core = (parts.get("task_core") or "").strip()
    io = (parts.get("io_requirement") or "").strip()
    if not io:
        return core
    glue = "\n\nNote: The following is the IO requirement.\n"
    return f"{core}{glue}{io}"


def compose_desc_core_plus_example(parts: Dict[str, str]) -> str:
    core = (parts.get("task_core") or "").strip()
    example = (parts.get("example") or "").strip()
    if not example:
        return core
    glue = "\n\nNote: The following is an example.\n"
    return f"{core}{glue}{example}"


def run_single_ablation(tag: str, codes: List[str], base_descriptions: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"], use_fast=True, trust_remote_code=True)

    texts_to_score, code_token_indices = prepare_texts_and_indices(
        codes=codes,
        descriptions=base_descriptions,
        tokenizer=tokenizer,
        use_description=True,
        combination_method=CONFIG["combination_method"],
        prompt_template=CONFIG["prompt_template"],
    )

    print(f"Starting scoring for {tag}...")
    scores = score_texts_by_code_part_with_checkpoint(
        texts_to_score,
        code_token_indices,
        tokenizer,
        max_tokens=CONFIG["max_tokens"],
        score_type="log_rank",
        timeout=CONFIG["timeout"],
        checkpoint_path=f"{generate_filename_prefix(tag)}_checkpoint.pkl",
    )

    print("Calculating evaluation metrics...")
    labels = None
    return scores


def main():
    root_path = Path(CONFIG["root_path"]).resolve()
    tokenizer_path = CONFIG["tokenizer_path"]
    print(f"root_path: {root_path}")
    print(f"tokenizer: {tokenizer_path}")

    solutions: List[Solution] = get_all_solutions(str(root_path))
    codes = [s.code for s in solutions]
    origin_descs = [s.description for s in solutions]

    task_to_parts: Dict[str, Dict[str, str]] = {}
    for s in tqdm(solutions, desc="Loading parts per task"):
        if s.task_name not in task_to_parts:
            task_to_parts[s.task_name] = load_task_parts(root_path, s.task_name)

    desc_core = [compose_desc_core(task_to_parts[s.task_name]) for s in solutions]
    desc_core_io = [compose_desc_core_plus_io(task_to_parts[s.task_name]) for s in solutions]
    desc_core_example = [compose_desc_core_plus_example(task_to_parts[s.task_name]) for s in solutions]

    runs = {
        "core_only": desc_core,
        "core_plus_io": desc_core_io,
        "core_plus_example": desc_core_example,
        "origin": origin_descs,
    }

    all_scores: Dict[str, List[float]] = {}
    for tag, descs in runs.items():
        scores = run_single_ablation(tag, codes, descs)
        all_scores[tag] = scores

        with open(f"{generate_filename_prefix(tag)}_scores.pkl", "wb") as f:
            pickle.dump((solutions, scores), f)

        labels = np.array([s.label for s in solutions], dtype=int)
        scores_np = np.array(scores, dtype=float)
        print(f"\n==== Evaluation for {tag} ====")
        evaluate_within_across_threshold(
            solutions=solutions,
            labels=labels,
            scores=scores_np,
            step=0.1,
            test_size=0.3,
            random_state=42,
        )

    with open("ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in all_scores.items()}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

