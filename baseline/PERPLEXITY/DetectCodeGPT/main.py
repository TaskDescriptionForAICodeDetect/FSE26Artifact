#!/usr/bin/env python3
import sys
import os
import json
import pickle
import random
import hashlib
from pathlib import Path
from typing import Dict, List
import numpy as np
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from baseutils.ioutils import get_all_solutions, Solution, sample_solutions_by_task
from baseutils.stasticutils import search_best_threshold, evaluate_within_across_threshold
from baseutils.taskUtil import get_description_processor
from PERPLEXITY.utils import get_prompt_template, prepare_texts_and_indices, score_texts_by_code_part_with_checkpoint

TOKENIZER = AutoTokenizer.from_pretrained("xxx/models/LLM-Research/Llama-3___2-3B", use_fast=True, trust_remote_code=True)
random.seed(0)
np.random.seed(0)

def generate_filename_prefix(config: dict) -> str:
    perturb_parts = [
        "detectcodegpt",
        f"n{config['n_perturbations']}",
        f"r{str(config['perturb_ratio']).replace('.', '')}"
    ]
    
    # Part 2: Text composition settings (from LogRank)
    text_parts = []
    if not config.get('use_description'):
        text_parts.append('code_only')
    else:
        text_parts.append('with_desc')
        if config.get('combination_method'):
            text_parts.append(f"combo_{config['combination_method']}")
        if config.get('desc_processor_name'):
            text_parts.append(f"proc_{config['desc_processor_name']}")
        if config.get('prompt_template'):
            prompt_hash = hashlib.md5(config['prompt_template'].encode()).hexdigest()[:8]
            text_parts.append(f"prompt_{prompt_hash}")

    return "_".join(perturb_parts) + "__" + "_".join(text_parts)


def _create_style_perturbation(code: str, config: dict) -> str:
    words = code.split(' ')
    num_to_perturb = int(len(words) * config['perturb_ratio'])
    if num_to_perturb > 0:
        perturb_indices = random.sample(range(len(words)), k=min(num_to_perturb, len(words)))
        for i in sorted(perturb_indices, reverse=True):
            num_spaces = np.random.poisson(config['whitespace_lambda'])
            words.insert(i, ' ' * num_spaces)
    perturbed_code = " ".join(words)

    lines = perturbed_code.split('\\n')
    num_lines_to_perturb = int(len(lines) * config['perturb_ratio'])
    if num_lines_to_perturb > 0:
        perturb_line_indices = random.sample(range(len(lines)), k=min(num_lines_to_perturb, len(lines)))
        new_lines = []
        perturb_set = set(perturb_line_indices)
        for i, line in enumerate(lines):
            new_lines.append(line)
            if i in perturb_set:
                num_newlines = np.random.poisson(config['newline_lambda']) + 1
                new_lines.extend([''] * num_newlines)
        return '\\n'.join(new_lines)
    return perturbed_code

def main(**config) -> dict:
    if config.get('prompt_template_name'):
        config['prompt_template'] = get_prompt_template(config['prompt_template_name'])
    sample_per_dataset = config.get('sample_per_dataset', 100)

    print("\n" + "="*50)
    print("--- Running DetectCodeGPT experiment with configuration: ---")
    for key, val in config.items():
        val_str = str(val)
        if len(val_str) > 100: val_str = val_str[:97] + "..."
        print(f"  - {key}: {val_str}")
    print("="*50 + "\n")

    filename_prefix = generate_filename_prefix(config)
    t_overall_start = time.time()
    perturb_cache_prefix = f"detectcodegpt_n{config['n_perturbations']}_r{str(config['perturb_ratio']).replace('.', '')}_s{sample_per_dataset}"
    perturb_cache_path = Path(f"{perturb_cache_prefix}_perturbed.pkl")
    checkpoint_path = f"{filename_prefix}_s{sample_per_dataset}_checkpoint.pkl"

    t_proc_start = time.time()
    fill_seconds_total = 0.0
    if perturb_cache_path.exists() and not config.get("force_perturb", False):
        print(f"Loaded perturbed data from cache: {perturb_cache_path}")
        with open(perturb_cache_path, "rb") as f:
            solutions, all_perturbed_codes = pickle.load(f)
    else:
        print("Loading solutions and generating perturbations...")
        solutions = get_all_solutions(config["root_path"])
        solutions = sample_solutions_by_task(solutions, sample_per_dataset=sample_per_dataset, random_state=42)
        all_perturbed_codes = []
        t_fill_start = time.time()
        for sol in tqdm(solutions, desc="Generating style perturbations"):
            perturbed_group = [_create_style_perturbation(sol.code, config) for _ in range(config["n_perturbations"])]
            all_perturbed_codes.append(perturbed_group)
        fill_seconds_total = time.time() - t_fill_start
        
        print(f"Saved perturbed data to cache: {perturb_cache_path}")
        with open(perturb_cache_path, "wb") as f:
            pickle.dump((solutions, all_perturbed_codes), f)

    print("Preparing texts to score...")
    desc_processor = get_description_processor(config.get('desc_processor_name'))
    
    all_codes_for_prep = []
    all_descs_for_prep = []
    group_sizes = []

    for i, sol in enumerate(solutions):
        processed_desc = desc_processor(sol.description) if desc_processor and config.get('use_description') else sol.description
        
        all_codes_for_prep.append(sol.code)
        all_descs_for_prep.append(processed_desc)
        
        perturbed_codes = all_perturbed_codes[i]
        all_codes_for_prep.extend(perturbed_codes)
        all_descs_for_prep.extend([processed_desc] * len(perturbed_codes))
        
        group_sizes.append(1 + len(perturbed_codes))

    texts_to_score, code_token_indices = prepare_texts_and_indices(
        codes=all_codes_for_prep,
        descriptions=all_descs_for_prep,
        tokenizer=TOKENIZER,
        use_description=config.get('use_description', True),
        combination_method=config.get('combination_method', 'direct'),
        prompt_template=config.get('prompt_template')
    )
    processing_seconds_total = time.time() - t_proc_start

    t_score_start = time.time()
    all_log_ranks = score_texts_by_code_part_with_checkpoint(
        texts_to_score=texts_to_score,
        code_token_indices=code_token_indices,
        tokenizer=TOKENIZER,
        max_tokens=config["max_tokens_per_batch"],
        score_type="log_rank",
        timeout=config["timeout"],
        checkpoint_path=checkpoint_path
    )
    scoring_seconds_total = time.time() - t_score_start
    
    final_scores = []
    current_pos = 0
    for size in tqdm(group_sizes, desc="Computing NPR scores"):
        group_ranks = all_log_ranks[current_pos : current_pos + size]
        current_pos += size
        original_rank, perturbed_ranks = group_ranks[0], group_ranks[1:]
        
        if original_rank is None or any(r is None for r in perturbed_ranks):
            final_scores.append(0.0)
            continue
            
        mean_perturbed_rank = np.mean(perturbed_ranks)
        std_perturbed_rank = np.std(perturbed_ranks) + 1e-6
        npr_score = (original_rank - mean_perturbed_rank) / std_perturbed_rank
        final_scores.append(npr_score)

    labels = np.array([s.label for s in solutions])
    scores = np.array(final_scores)
    
    flipped_scores = -scores
    best_thr_flipped, best_f1 = search_best_threshold(labels, flipped_scores, config["thr_step"])
    best_thr = -best_thr_flipped
    y_pred = (scores > best_thr).astype(int)

    t_cls_start = time.time()
    evaluate_within_across_threshold(
        solutions=solutions,
        labels=labels,
        scores=flipped_scores,
        step=config["thr_step"],
        test_size=0.3,
        random_state=42,
    )
    classification_seconds_total = time.time() - t_cls_start
    overall_seconds_total = time.time() - t_overall_start

    n = max(1, len(solutions))
    timing = {
        "num_solutions": len(solutions),
        "processing_seconds_total": processing_seconds_total,
        "scoring_seconds_total": scoring_seconds_total,
        "classification_seconds_total": classification_seconds_total,
        "overall_seconds_total": overall_seconds_total,
        "fill_seconds_total": float(fill_seconds_total),
        "avg_processing_seconds_per_solution": processing_seconds_total / n,
        "avg_scoring_seconds_per_solution": scoring_seconds_total / n,
        "avg_classification_seconds_per_solution": classification_seconds_total / n,
        "avg_overall_seconds_per_solution": overall_seconds_total / n,
        "avg_fill_seconds_per_solution": float(fill_seconds_total) / n,
    }
    print("Timing summary:", timing)
    with open(f"{filename_prefix}_timing.json", "w", encoding='utf-8') as f:
        json.dump(timing, f, indent=4)

    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items() if k != 'prompt_template'},
    }

def run_all_experiments():
    base_config = {
        "root_path": Path("../../splitDescription/split_benchmark"),
        "thr_step": 0.01,
        "force_perturb": False,
        "max_tokens_per_batch": 4096,
        "timeout": 120,
        "sample_per_dataset": 100,
        "n_perturbations": 3,
        "perturb_ratio": 0.5,
        "whitespace_lambda": 2,
        "newline_lambda": 2,
    }
    
    experiment_configs = [
        {"use_description": False},
        {
            "use_description": True,
            "combination_method": 'direct',
            "desc_processor_name": 'simple',
        },
        {
            "use_description": True,
            "combination_method": 'prompt',
            "prompt_template_name": 'default',
            "desc_processor_name": 'simple'
        },
    ]

    all_results = []
    for exp_config in experiment_configs:
        full_config = {**base_config, **exp_config}
        result = main(**full_config)
        all_results.append(result)

    summary_path = "detectcodegpt_experiment_summary.json"
    print(f"\n{'='*30}\nExperiment summary saved to -> {summary_path}\n{'='*30}")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    run_all_experiments()
