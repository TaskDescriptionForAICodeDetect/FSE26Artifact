import sys
import os
import json
import pickle
import random
import math
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from baseutils.ioutils import (
    get_all_solutions,
    Solution,
    sample_solutions_by_task,
    print_solution_sample_distribution,
)
from baseutils.taskUtil import get_description_processor
from baseutils.stasticutils import evaluate_within_across_threshold
from PERPLEXITY.utils import get_prompt_template
from PERPLEXITY.DetectGPT.fillUtil import MASK_TOKEN, fill_one
from PERPLEXITY.inferutil import lightweight_batch_by_words
from PERPLEXITY.lmsync_util import score_many
from PERPLEXITY.DetectGPT.progress_util import run_with_checkpointing_parallel

def generate_filename_prefix(config: dict) -> str:
    prefix_parts = ['dgpt']

    prefix_parts.append(f"n{config['n_perturbations']}")
    prefix_parts.append(f"f{str(config['perturb_fraction']).replace('.', '')}")
    
    tokenizer_short_name = config['tokenizer_name'].split('/')[-1]
    prefix_parts.append(f"tok_{tokenizer_short_name}")

    if not config.get('use_description'):
        prefix_parts.append('code_only')
    else:
        prefix_parts.append('with_desc')
        if config.get('prompt_template_name'):
            prefix_parts.append(f"prompt_{config.get('prompt_template_name')}")
            
    fill_model_name = config.get('fill_model', 'gpt-3.5-turbo').split('/')[-1]
    prefix_parts.append(f"fill_{fill_model_name}")
    
    if config.get('enable_sampling', True):
        sample_per_dataset = int(config.get('sample_per_dataset', 100))
        rs = int(config.get('random_state', 42))
        prefix_parts.append(f"samp{sample_per_dataset}_rs{rs}")
            
    return "_".join(prefix_parts)

def _create_perturbation(tokens, config):
    num_tokens, span_length = len(tokens), config['span_length']
    if num_tokens <= span_length: return tokens[:]
    num_spans_by_fraction = math.ceil((num_tokens / span_length) * config['perturb_fraction'])
    num_spans_to_perturb = min(num_spans_by_fraction, config['max_perturb_spans'])
    possible_starts = list(range(0, num_tokens - span_length + 1))
    random.shuffle(possible_starts)
    selected_starts, used_indices = [], set()
    for start in possible_starts:
        if len(selected_starts) >= num_spans_to_perturb: break
        if not any(i in used_indices for i in range(start, start + span_length)):
            selected_starts.append(start)
            for i in range(start, start + span_length): used_indices.add(i)
    if not selected_starts: return tokens[:]
    new_tokens, i = [], 0
    selected_starts_set = set(selected_starts)
    while i < num_tokens:
        if i in selected_starts_set:
            new_tokens.append(MASK_TOKEN); i += span_length
        else:
            new_tokens.append(tokens[i]); i += 1
    return new_tokens

def generate_and_save_perturbations(solutions: list[Solution], config, tokenizer):
    print("--- Step 1: Start generating perturbed texts ---")
    all_masked_texts_flat = []
    
    prompt_template = get_prompt_template(config['prompt_template_name']) if config.get('prompt_template_name') else "{desc}\\n\\n{code}"

    for sol in tqdm(solutions, desc="Generating masks locally"):
        toks = tokenizer.tokenize(sol.code)
        for _ in range(config['n_perturbations']):
            masked_toks = _create_perturbation(toks, config)
            masked_code_str = tokenizer.convert_tokens_to_string(masked_toks)
            
            if config['use_description'] and sol.description:
                full_masked_text = prompt_template.format(desc=sol.description, code=masked_code_str)
            else:
                full_masked_text = masked_code_str
            all_masked_texts_flat.append(full_masked_text)

    
    fill_func = lambda text: fill_one(
        text,
        model=config.get('fill_model', 'local'),
        temperature=config.get('fill_temperature', 0.7),
        max_retry=config.get('fill_max_retry', 3),
        max_new_tokens=config.get('fill_max_new_tokens', 256),
    )

    import time as _time
    _t_fill_start = _time.time()
    all_filled_texts_flat = run_with_checkpointing_parallel(
        all_items=all_masked_texts_flat,
        process_item_func=fill_func,
        checkpoint_path=f"{config['filename_prefix']}_perturb_checkpoint.pkl",
        max_workers=1,
        desc="Single-threaded local filling"
    )
    fill_seconds_total = _time.time() - _t_fill_start

    all_perturbed_texts = []
    current_pos = 0
    for _ in solutions:
        n_p = config['n_perturbations']
        all_perturbed_texts.append(all_filled_texts_flat[current_pos : current_pos + n_p])
        current_pos += n_p

    data_to_save = {"solutions": solutions, "perturbed_texts": all_perturbed_texts}
    with open(config['perturb_data_path'], "wb") as f: pickle.dump(data_to_save, f)
    return fill_seconds_total

def calculate_scores_from_saved_data(config):
    with open(config['perturb_data_path'], "rb") as f: saved_data = pickle.load(f)

    solutions, all_perturbed_texts = saved_data["solutions"], saved_data["perturbed_texts"]

    all_texts_to_score, group_sizes = [], []
    prompt_template = get_prompt_template(config['prompt_template_name']) if config.get('prompt_template_name') else "{desc}\\n\\n{code}"
    
    for i, sol in enumerate(solutions):
        if config['use_description'] and sol.description:
            all_texts_to_score.append(prompt_template.format(desc=sol.description, code=sol.code))
        else:
            all_texts_to_score.append(sol.code)
            
        all_texts_to_score.extend(all_perturbed_texts[i])
        group_sizes.append(1 + len(all_perturbed_texts[i]))

    batches = lightweight_batch_by_words(all_texts_to_score, max_tokens=config['max_tokens_per_batch'])
    
    def get_avg_scores_for_batch(batch):
        batch_log_probs = score_many(batch, score_type="log_probs")
        return [sum(s) / len(s) if s else 0 for s in batch_log_probs]

    scored_batches = run_with_checkpointing_parallel(
        all_items=batches,
        process_item_func=get_avg_scores_for_batch,
        checkpoint_path=f"{config['filename_prefix']}_scoring_checkpoint.pkl",
        max_workers=1,
        desc="Single-threaded local scoring"
    )

    all_scores = [item for batch in scored_batches for item in batch]
    
    final_scores, current_pos = [], 0
    for size in tqdm(group_sizes, desc="Calculating final scores"):
        group_scores = all_scores[current_pos: current_pos + size]
        current_pos += size
        if not group_scores or group_scores[0] is None:
            final_scores.append(0.0); continue
        orig_ent, pert_ents = group_scores[0], [s for s in group_scores[1:] if s is not None]
        if not pert_ents:
            final_scores.append(0.0); continue
        mu, std = np.mean([-e for e in pert_ents]), np.std([-e for e in pert_ents]) + 1e-6
        score = -((-(orig_ent) - mu) / std)
        final_scores.append(score)

    return solutions, np.array(final_scores, dtype=float)

def main(**config):
    config['filename_prefix'] = generate_filename_prefix(config)
    config['perturb_data_path'] = f"{config['filename_prefix']}_perturbed.pkl"
    config['results_path'] = f"{config['filename_prefix']}_results.json"
    
    print(f"\n{'='*50}\n--- Starting Experiment: {config['filename_prefix']} ---")
    t_overall_start = time.time()

    processing_seconds_total = 0.0
    scoring_seconds_total = 0.0
    classification_seconds_total = 0.0

    if config['force_perturb'] or not Path(config['perturb_data_path']).exists():
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], trust_remote_code=True)
        processor = get_description_processor(config.get('desc_processor_name'))
        solutions = get_all_solutions(config['root_path'], process_desc_func=processor)
        
        if config.get('enable_sampling', True):
            solutions = sample_solutions_by_task(
                solutions,
                sample_per_dataset=int(config.get('sample_per_dataset', 100)),
                random_state=int(config.get('random_state', 42)),
            )
            print_solution_sample_distribution(solutions)
        t_proc_start = time.time()
        fill_seconds_total = generate_and_save_perturbations(solutions, config, tokenizer)
        processing_seconds_total = time.time() - t_proc_start
    else:
        print(f"Found existing perturbation data file: {config['perturb_data_path']}.")

    t_score_start = time.time()
    solutions, scores = calculate_scores_from_saved_data(config)
    scoring_seconds_total = time.time() - t_score_start
    
    print("\n--- Step 3: Evaluating results ---")
    labels = np.array([s.label for s in solutions], dtype=int)
    flipped_scores = -scores
    t_cls_start = time.time()
    evaluate_within_across_threshold(
        solutions=solutions,
        labels=labels,
        scores=flipped_scores,
        test_size=config.get('test_size', 0.3),
        random_state=config.get('random_state', 42)
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
        "avg_processing_seconds_per_solution": processing_seconds_total / n,
        "avg_scoring_seconds_per_solution": scoring_seconds_total / n,
        "avg_classification_seconds_per_solution": classification_seconds_total / n,
        "avg_overall_seconds_per_solution": overall_seconds_total / n,
        "fill_seconds_total": float(locals().get("fill_seconds_total", 0.0)),
        "avg_fill_seconds_per_solution": float(locals().get("fill_seconds_total", 0.0)) / n,
    }
    print("Timing summary:", timing)
    with open(f"{config['filename_prefix']}_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    

def run_all_experiments():
    base_path = "../../splitDescription/split_benchmark"
    base_config = {
        "root_path": base_path,
        "test_size": 0.3, "random_state": 42,
        "enable_sampling": True,
        "sample_per_dataset": 20,
        "force_perturb": False, "span_length": 2,
        "n_perturbations": 4, "perturb_fraction": 0.15, "max_perturb_spans": 20,
        "fill_model": "local",
        "fill_temperature": 0.7, "fill_max_retry": 3, "fill_max_workers": 8,
        "score_max_workers": 4,
        "max_tokens_per_batch": 2048,
        "tokenizer_name": "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B",
        "use_description": False, "desc_processor_name": None,
        "prompt_template_name": None
    }

    experiment_configs = [
        {"use_description": False},
        {"use_description": True, "prompt_template_name": 'default'},
    ]

    for exp_config in experiment_configs:
        main(**{**base_config, **exp_config})

if __name__ == "__main__":
    run_all_experiments()