import sys, os, json, argparse, pickle, random, hashlib
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from baseutils.stasticutils import evaluate_within_across_threshold
from baseutils.ioutils import get_all_solutions, Solution
from baseutils.taskUtil import get_description_processor
from PERPLEXITY.utils import get_prompt_template, prepare_texts_and_indices, score_texts_by_code_part_with_checkpoint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B", use_fast=True, trust_remote_code=True)

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
    max_tokens: int = 4096,
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
    score_file_path = f"{filename_prefix}_entropy_scores.pkl"
    checkpoint_path = f"{filename_prefix}_entropy_checkpoint.pkl"

    selected_processor = get_description_processor(desc_processor_name) if use_description else None
    solutions = get_all_solutions(root_path, process_desc_func=selected_processor)
    
    codes = [s.code for s in solutions]
    descriptions = [s.description for s in solutions] if use_description else None

    texts_to_score, code_token_indices = prepare_texts_and_indices(
        codes, descriptions, tokenizer, use_description, combination_method, prompt_template
    )
    
    print("Starting scoring process (resumable)...")
    t_score_start = time.time()
    scores = score_texts_by_code_part_with_checkpoint(
        texts_to_score,
        code_token_indices,
        tokenizer,
        max_tokens=max_tokens,
        score_type="entropy",
        timeout=timeout,
        checkpoint_path=checkpoint_path
    )
    scoring_seconds_total = time.time() - t_score_start
    
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
    output_json_path = f"{filename_prefix}_entropy_results.json"
    print(f"Saving detailed results to -> {output_json_path}")
    with open(output_json_path, "w", encoding='utf-8') as f:
        json.dump(
            [
                {**s.__dict__, 'score': float(score)}
                for s, score in zip(solutions, scores)
            ],
            f, indent=2, ensure_ascii=False
        )

    return {"config": {k: v for k, v in config.items() if k != 'root_path'}}


def run_all_experiments():
    base_path = "../../splitDescription/split_benchmark"
    
    experiment_configs = [
        {
            "use_description": False,
        },
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

    print("\n\n" + "="*30 + " Experiments finished (see console for classification reports) " + "="*30)

    summary_path = "entropy_experiment_summary.json"
    print(f"Experiment summary saved to -> {summary_path}")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_all_experiments()
