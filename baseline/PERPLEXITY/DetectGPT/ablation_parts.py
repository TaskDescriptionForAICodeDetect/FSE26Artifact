#!/usr/bin/env python3
import sys
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from baseutils.ioutils import Solution
from baseutils.stasticutils import evaluate_within_across_threshold
from PERPLEXITY.utils import get_prompt_template, score_texts_by_code_part_with_checkpoint
from PERPLEXITY.DetectGPT.detect_existing_pertubed import (
    derive_results_prefix,
    filter_subset,
)


CONFIG: Dict[str, Any] = {
    "root_path": "../../splitDescription/split_benchmark",
    "perturb_data_path": "dgpt_n4_f015_tok_llama32-3b_code_only_fill_gpt-4o-mini_perturbed.pkl",
    "subset_per_dataset": 100,
    "tokenizer_name": "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B",
    "max_tokens_per_batch": 2048,
    "timeout": 120,
    "desc_apply_to": "orig_only",
    "prompt_template_name": None,
}


def generate_filename_prefix(tag: str, base_prefix: str) -> str:
    return f"{base_prefix}_ablation_{tag}"


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


def build_texts_and_indices(
    solutions: List[Solution],
    perturbed_texts: List[List[str]],
    *,
    use_description: bool,
    desc_apply_to: str,
    tokenizer,
    prompt_template_name: str | None,
) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    prompt_template = get_prompt_template(prompt_template_name) if prompt_template_name else "{desc}\\n\\n{code}"

    texts_to_score: List[str] = []
    code_token_indices: List[Tuple[int, int]] = []
    group_sizes: List[int] = []

    for i, sol in enumerate(solutions):
        desc = sol.description if (use_description and sol.description) else None

        prefix_text = ""
        suffix_text = ""
        if desc is not None:
            tpl_parts = prompt_template.split('{code}')
            if len(tpl_parts) != 2:
                raise ValueError("prompt_template must contain exactly one '{code}' placeholder.")
            prefix_text = tpl_parts[0].format(desc=desc)
            suffix_text = tpl_parts[1].format(desc=desc)

        full_text = f"{prefix_text}{sol.code}{suffix_text}"
        start_idx = len(tokenizer.encode(prefix_text, add_special_tokens=False))
        end_idx = start_idx + len(tokenizer.encode(sol.code, add_special_tokens=False))
        texts_to_score.append(full_text)
        code_token_indices.append((start_idx, end_idx))

        for pert in perturbed_texts[i]:
            if desc is not None and desc_apply_to == "orig_and_perturbed":
                full_pert_text = f"{prefix_text}{pert}{suffix_text}"
                s_idx = len(tokenizer.encode(prefix_text, add_special_tokens=False))
                e_idx = s_idx + len(tokenizer.encode(pert, add_special_tokens=False))
            else:
                full_pert_text = pert
                s_idx = 0
                e_idx = len(tokenizer.encode(full_pert_text, add_special_tokens=False))

            texts_to_score.append(full_pert_text)
            code_token_indices.append((s_idx, e_idx))

        group_sizes.append(1 + len(perturbed_texts[i]))

    return texts_to_score, code_token_indices, group_sizes


def score_with_descriptions(
    solutions: List[Solution],
    perturbed_texts: List[List[str]],
    *,
    descriptions: List[str],
    tokenizer_name: str,
    desc_apply_to: str,
    prompt_template_name: str | None,
    max_tokens_per_batch: int,
    timeout: int,
    results_prefix: str,
) -> np.ndarray:
    assert len(solutions) == len(descriptions)

    replaced: List[Solution] = []
    for s, d in zip(solutions, descriptions):
        replaced.append(
            Solution(
                code=s.code,
                label=s.label,
                task_name=s.task_name,
                description=d or "",
                language=s.language,
                model=s.model,
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts_to_score, code_token_indices, group_sizes = build_texts_and_indices(
        replaced,
        perturbed_texts,
        use_description=True,
        desc_apply_to=desc_apply_to,
        tokenizer=tokenizer,
        prompt_template_name=prompt_template_name,
    )

    mean_code_scores = score_texts_by_code_part_with_checkpoint(
        texts_to_score,
        code_token_indices,
        tokenizer,
        max_tokens=max_tokens_per_batch,
        score_type="log_probs",
        timeout=timeout,
        checkpoint_path=f"{results_prefix}_scoring_checkpoint.pkl",
    )

    final_scores, cursor = [], 0
    for size in group_sizes:
        group_scores = mean_code_scores[cursor: cursor + size]
        cursor += size
        if not group_scores or group_scores[0] is None:
            final_scores.append(0.0); continue
        orig_ent = group_scores[0]
        pert_ents = [s for s in group_scores[1:] if s is not None]
        if not pert_ents:
            final_scores.append(0.0); continue
        mu = np.mean([-e for e in pert_ents])
        std = np.std([-e for e in pert_ents]) + 1e-6
        score = -((-(orig_ent) - mu) / std)
        final_scores.append(score)

    return np.array(final_scores, dtype=float)


def main():
    root_path = Path(CONFIG["root_path"]).resolve()
    perturb_path = CONFIG["perturb_data_path"]
    subset_per_dataset = int(CONFIG["subset_per_dataset"])
    tokenizer_name = CONFIG["tokenizer_name"]
    max_tokens_per_batch = int(CONFIG["max_tokens_per_batch"])
    timeout = int(CONFIG["timeout"])
    desc_apply_to = CONFIG["desc_apply_to"]
    prompt_template_name = CONFIG["prompt_template_name"]

    if not Path(perturb_path).exists():
        raise FileNotFoundError(f"Perturbation data file not found: {perturb_path}")

    with open(perturb_path, "rb") as f:
        saved_data = pickle.load(f)

    solutions_sub, perturbed_sub = filter_subset(saved_data, per_dataset=subset_per_dataset)
    print(f"Number of solutions after filtering: {len(solutions_sub)} (should be around {subset_per_dataset * 2})")

    task_to_parts: Dict[str, Dict[str, str]] = {}
    for s in tqdm(solutions_sub, desc="Loading parts per task"):
        if s.task_name not in task_to_parts:
            task_to_parts[s.task_name] = load_task_parts(root_path, s.task_name)

    desc_core = [compose_desc_core(task_to_parts[s.task_name]) for s in solutions_sub]
    desc_core_io = [compose_desc_core_plus_io(task_to_parts[s.task_name]) for s in solutions_sub]
    desc_core_example = [compose_desc_core_plus_example(task_to_parts[s.task_name]) for s in solutions_sub]
    origin_descs = [s.description for s in solutions_sub]

    runs = {
        "core_only": desc_core,
        "core_plus_io": desc_core_io,
        "core_plus_example": desc_core_example,
        "origin": origin_descs,
    }

    base_prefix = derive_results_prefix(perturb_path, f"subset_apps{subset_per_dataset}_codenet{subset_per_dataset}")
    all_scores: Dict[str, List[float]] = {}

    for tag, descs in runs.items():
        suffix_parts = [f"apply-{desc_apply_to}"]
        if prompt_template_name:
            suffix_parts.append(f"prompt-{prompt_template_name}")
        results_prefix = generate_filename_prefix(tag, base_prefix) + "_" + "_".join(suffix_parts)

        print(f"\n==== Starting DetectGPT Ablation ({tag}) ====")
        scores = score_with_descriptions(
            solutions_sub,
            perturbed_sub,
            descriptions=descs,
            tokenizer_name=tokenizer_name,
            desc_apply_to=desc_apply_to,
            prompt_template_name=prompt_template_name,
            max_tokens_per_batch=max_tokens_per_batch,
            timeout=timeout,
            results_prefix=results_prefix,
        )
        all_scores[tag] = [float(x) for x in scores]

        with open(f"{results_prefix}_scores.pkl", "wb") as f:
            pickle.dump((solutions_sub, scores), f)

        labels = np.array([s.label for s in solutions_sub], dtype=int)
        flipped_scores = -scores
        print(f"\n==== Evaluation for {tag} ====")
        evaluate_within_across_threshold(
            solutions=solutions_sub,
            labels=labels,
            scores=flipped_scores,
            test_size=0.3,
            random_state=42,
        )

    with open("ablation_detectgpt_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

