#!/usr/bin/env python3
import sys
import os
import pickle
from pathlib import Path
from typing import List
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from baseutils.ioutils import Solution, infer_dataset_source_from_task, print_solution_sample_distribution
from baseutils.stasticutils import evaluate_within_across_threshold
from PERPLEXITY.utils import get_prompt_template, score_texts_by_code_part_with_checkpoint
from transformers import AutoTokenizer


def derive_results_prefix(perturb_data_path: str, subset_tag: str) -> str:
    base = Path(perturb_data_path).name
    if base.endswith("_perturbed.pkl"):
        base = base[: -len("_perturbed.pkl")]
    return f"{base}_{subset_tag}"


def select_task_names_by_dataset(solutions: List[Solution], per_dataset: int) -> set:
    apps_tasks = sorted({s.task_name for s in solutions if infer_dataset_source_from_task(s.task_name) == "apps"})
    codenet_tasks = sorted({s.task_name for s in solutions if infer_dataset_source_from_task(s.task_name) == "codenet"})
    return set(apps_tasks[: per_dataset]).union(set(codenet_tasks[: per_dataset]))


def filter_subset(saved_data: dict, per_dataset: int) -> tuple[List[Solution], list[list[str]]]:
    solutions: List[Solution] = saved_data["solutions"]
    perturbed_texts: list[list[str]] = saved_data["perturbed_texts"]

    selected_tasks = select_task_names_by_dataset(solutions, per_dataset)
    keep_indices = [i for i, s in enumerate(solutions) if s.task_name in selected_tasks]

    sub_solutions = [solutions[i] for i in keep_indices]
    sub_perturbed = [perturbed_texts[i] for i in keep_indices]

    return sub_solutions, sub_perturbed


def score_subset_and_evaluate(
    solutions: List[Solution],
    perturbed_texts: list[list[str]],
    *,
    use_description: bool = False,
    prompt_template_name: str | None = None,
    tokenizer_name: str = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B",
    desc_apply_to: str = "orig_only",
    max_tokens_per_batch: int = 2048,
    timeout: int = 120,
    results_prefix: str = "subset",
):
    prompt_template = get_prompt_template(prompt_template_name) if prompt_template_name else "{desc}\\n\\n{code}"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts_to_score: list[str] = []
    code_token_indices: list[tuple[int, int]] = []
    group_sizes: list[int] = []

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
        orig_ent, pert_ents = group_scores[0], [s for s in group_scores[1:] if s is not None]
        if not pert_ents:
            final_scores.append(0.0); continue
        mu = np.mean([-e for e in pert_ents])
        std = np.std([-e for e in pert_ents]) + 1e-6
        score = -((-(orig_ent) - mu) / std)
        final_scores.append(score)

    scores_arr = np.array(final_scores, dtype=float)

    print("\n--- Subset Evaluation (apps/codenet 100 each) ---")
    labels = np.array([s.label for s in solutions], dtype=int)
    flipped_scores = -scores_arr
    evaluate_within_across_threshold(
        solutions=solutions,
        labels=labels,
        scores=flipped_scores,
        test_size=0.3,
        random_state=42,
    )


def run(**config):
    perturb_path = config.get("perturb_data_path", "dgpt_n4_f015_tok_llama32-3b_code_only_fill_gpt-4o-mini_perturbed.pkl")
    subset_per_dataset = int(config.get("subset_per_dataset", 100))
    use_description = bool(config.get("use_description", False))
    prompt_template_name = config.get("prompt_template_name", None)
    max_tokens_per_batch = int(config.get("max_tokens_per_batch", 2048))
    tokenizer_name = config.get("tokenizer_name", "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B")
    desc_apply_to = config.get("desc_apply_to", "orig_only")
    timeout = int(config.get("timeout", 120))

    exp_cfg = {
        "perturb_data_path": perturb_path,
        "subset_per_dataset": subset_per_dataset,
        "use_description": use_description,
        "desc_apply_to": desc_apply_to,
        "prompt_template_name": prompt_template_name,
        "tokenizer_name": tokenizer_name,
        "max_tokens_per_batch": max_tokens_per_batch,
        "timeout": timeout,
    }
    print("\n" + "=" * 24 + " Experiment Settings (detect_existing_pertubed.run) " + "=" * 24)
    for k, v in exp_cfg.items():
        print(f"  - {k}: {v}")
    print("=" * 70)

    if not Path(perturb_path).exists():
        raise FileNotFoundError(f"Perturbation data file not found: {perturb_path}")

    with open(perturb_path, "rb") as f:
        saved_data = pickle.load(f)

    solutions_sub, perturbed_sub = filter_subset(saved_data, per_dataset=subset_per_dataset)
    print(f"Number of solutions after filtering: {len(solutions_sub)} (should be around {subset_per_dataset * 2})")
    print_solution_sample_distribution(solutions_sub)

    results_prefix = derive_results_prefix(perturb_path, f"subset_apps{subset_per_dataset}_codenet{subset_per_dataset}")
    suffix_parts = [f"desc{int(use_description)}"]
    if use_description:
        suffix_parts.append(f"apply-{desc_apply_to}")
        if prompt_template_name:
            suffix_parts.append(f"prompt-{prompt_template_name}")
    results_prefix = f"{results_prefix}_" + "_".join(suffix_parts)

    print(f"Results file prefix: {results_prefix}")

    score_subset_and_evaluate(
        solutions_sub,
        perturbed_sub,
        use_description=use_description,
        prompt_template_name=prompt_template_name,
        tokenizer_name=tokenizer_name,
        desc_apply_to=desc_apply_to,
        max_tokens_per_batch=max_tokens_per_batch,
        timeout=timeout,
        results_prefix=results_prefix,
    )


def run_default():
    base_config = {
        "perturb_data_path": "dgpt_n4_f015_tok_llama32-3b_code_only_fill_gpt-4o_samp100_rs42_perturbed.pkl",
        "subset_per_dataset": 100,
        "use_description": False,
        "desc_apply_to": "orig_only",
        "prompt_template_name": "default",
        "max_tokens_per_batch": 2048,
        "tokenizer_name": "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B",
        "timeout": 120,
    }
    run(**base_config)


def run_experiments():
    base_config = {
        "perturb_data_path": "dgpt_n4_f015_tok_llama32-3b_code_only_fill_gpt-4o-mini_perturbed.pkl",
        "subset_per_dataset": 100,
        "use_description": True,
        "desc_apply_to": "orig_only",
        "prompt_template_name": "default",
        "max_tokens_per_batch": 2048,
        "tokenizer_name": "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B",
        "timeout": 120,
    }

    experiment_configs = [
        {"use_description": True,  "desc_apply_to": "orig_only"},
        {"use_description": True,  "desc_apply_to": "orig_and_perturbed"},
    ]

    for cfg in experiment_configs:
        merged = {**base_config, **cfg}
        print("\n" + "=" * 20 + f" Running configuration: {merged}" + "=" * 20)
        run(**merged)


if __name__ == "__main__":
    run_default()

