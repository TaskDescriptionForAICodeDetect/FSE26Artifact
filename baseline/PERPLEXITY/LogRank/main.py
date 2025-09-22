#!/usr/bin/env python3
import sys, os, json, argparse, pickle, hashlib
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from baseutils.ioutils import get_all_solutions, Solution
from baseutils.taskUtil import get_description_processor
from baseutils.stasticutils import search_best_threshold, evaluate_within_across_threshold
from PERPLEXITY.utils import get_prompt_template, prepare_texts_and_indices, score_texts_by_code_part_with_checkpoint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B", use_fast=True, trust_remote_code=True
)

def generate_filename_prefix(config: dict) -> str:
    prefix_parts = []
    if not config.get('use_description'):
        prefix_parts.append('code_only')
    else:
        prefix_parts.append('with_desc')
        if config.get('combination_method'):
            prefix_parts.append(f"combo_{config.get('combination_method')}")
        if config.get('desc_processor_name'):
            prefix_parts.append(f"proc_{config.get('desc_processor_name')}")
        
        if config.get('prompt_template'):
            prompt_hash = hashlib.md5(config.get('prompt_template').encode()).hexdigest()[:8]
            prefix_parts.append(f"prompt_{prompt_hash}")

    return "_".join(prefix_parts)


def main(
    root_path: str,
    step: float = 0.01,
    max_tokens: int = 2048,
    timeout: int = 120,
    use_description: bool = True,
    desc_processor_name: str = None,
    combination_method: str = 'direct',
    prompt_template_name: str = None
) -> dict:
    config = locals()
    prompt_template = None
    if config.get('prompt_template_name'):
        prompt_template = get_prompt_template(config['prompt_template_name'])
        config['prompt_template'] = prompt_template

    print("\n" + "="*50)
    print("--- Running experiment with the following configuration: ---")
    for key, val in config.items():
        if key != 'root_path' and val is not None:
             print(f"  - {key}: {val if len(str(val)) < 80 else str(val)[:77] + '...'}")
    print("="*50 + "\n")

    filename_prefix = generate_filename_prefix(config)
    t_overall_start = time.time()
    score_file_path = f"{filename_prefix}_scores.pkl"
    checkpoint_path = f"{filename_prefix}_checkpoint.pkl"

    selected_processor = get_description_processor(desc_processor_name) if use_description else None
    solutions = get_all_solutions(root_path, process_desc_func=selected_processor)
    
    codes = [s.code for s in solutions]
    descriptions = [s.description for s in solutions] if use_description else None

    texts_to_score, code_token_indices = prepare_texts_and_indices(
        codes, descriptions, tokenizer, use_description, combination_method, prompt_template
    )
    
    print("Starting scoring process (resumable)...")
    scores = score_texts_by_code_part_with_checkpoint(
        texts_to_score,
        code_token_indices,
        tokenizer,
        max_tokens=max_tokens,
        score_type="log_rank",
        timeout=timeout,
        checkpoint_path=checkpoint_path
    )

    print(f"All batches processed. Saving final scores to -> {score_file_path}")
    with open(score_file_path, "wb") as f:
        pickle.dump((solutions, scores), f)
    
    labels = np.array([s.label for s in solutions], dtype=int)
    scores = np.array(scores, dtype=float)

    t_cls_start = time.time()
    evaluate_within_across_threshold(
        solutions=solutions,
        labels=labels,
        scores=scores,
        step=step,
        test_size=0.3,
        random_state=42,
    )

    n = max(1, len(solutions))
    output_json_path = f"{filename_prefix}_results.json"
    print(f"Saving scores to -> {output_json_path}")
    with open(output_json_path, "w", encoding='utf-8') as f:
        json.dump(
            [
                {**s.__dict__, 'score': float(score)}
                for s, score in zip(solutions, scores)
            ],
            f, indent=2, ensure_ascii=False
        )

    return {
        "config": {k: v for k, v in config.items() if k != 'root_path'},
    }


def run_all_experiments():
    base_path = "/root/autodl-tmp/com_benchmark"
    
    experiment_configs = [
        {
            "use_description": False,
        },
    ]

    all_results = []
    for config in experiment_configs:
        full_config = {
            "root_path": base_path,
            "step": 0.1,
            "max_tokens": 4096,
            "timeout": 120,
            "use_description": True,
            "desc_processor_name": None,
            "combination_method": 'direct',
            "prompt_template_name": None,
            **config
        }
        result = main(**full_config)
        all_results.append(result)

    print("\n\n" + "="*30 + " Experiment Summary Report " + "="*30)
    for result in all_results:
        config_str = generate_filename_prefix(result['config'])
        print(f"\n--- Configuration: {config_str} ---")
        print(f"  - Experiment finished. See logs above for detailed evaluation results.")

    summary_path = "logrank_experiment_summary.json"
    print(f"\n{'='*30}\nExperiment summary report saved to -> {summary_path}\n{'='*30}")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_all_experiments()
