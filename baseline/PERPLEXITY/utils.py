import sys, os, json, argparse, random, pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


MAX_SOL_PER_TYPE = 1
LANGUAGE_FILTER  = "python"
random.seed(0)


def search_best_threshold(labels: np.ndarray, scores: np.ndarray, step: float = 0.01):
    min_s, max_s = scores.min(), scores.max()
    thresholds = np.arange(min_s, max_s + step, step)
    best_thr, best_f1 = None, -1.0

    for thr in thresholds:
        preds = (scores < thr).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1


def get_prompt_template(name: str) -> str:
    templates = {
        'default': "Below is a programming problem description, followed by a solution to it.\\n\\n### Description\\n{desc}\\n\\n### Solution\\n{code}",
        'simple': "Description: {desc}\\n\\nCode:\\n{code}",
    }
    if name not in templates:
        raise ValueError(f"Prompt template '{name}' not found. Available templates are: {list(templates.keys())}")
    return templates[name]


def prepare_texts_and_indices(
    codes, descriptions, tokenizer, use_description, combination_method, prompt_template
):
    texts_to_score = []
    code_token_indices = []

    if descriptions is None:
        descriptions = [None] * len(codes)

    for code, desc in tqdm(zip(codes, descriptions), total=len(codes), desc="Preparing texts and indices"):
        prefix_text = ""
        suffix_text = ""

        if use_description and desc is not None:
            if combination_method == 'prompt' and prompt_template:
                template_parts = prompt_template.split('{code}')
                if len(template_parts) != 2:
                    raise ValueError("prompt_template must contain exactly one '{code}' placeholder.")
                prefix_text = template_parts[0].format(desc=desc)
                suffix_text = template_parts[1].format(desc=desc)
            else:
                prefix_text = f"{desc}\\n\\n---\\n\\n"
        
        prefix_tokens_len = len(tokenizer.encode(prefix_text, add_special_tokens=False))
        code_tokens_len = len(tokenizer.encode(code, add_special_tokens=False))
        
        start_idx = prefix_tokens_len
        end_idx = prefix_tokens_len + code_tokens_len
        
        texts_to_score.append(prefix_text + code + suffix_text)
        code_token_indices.append((start_idx, end_idx))
        
    return texts_to_score, code_token_indices


def score_texts_by_code_part(
    texts_to_score, code_token_indices, tokenizer, max_tokens, score_type, timeout
):
    from .inferutil import batch_by_tokens
    from .lmsync_util import score_many

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    code_batches = batch_by_tokens(texts_to_score, tokenizer, max_tokens)
    
    mean_scores = []
    text_cursor = 0
    for batch_codes in tqdm(code_batches, desc="Scoring batches and processing results"):
        batch_token_scores = score_many(
            batch_codes,
            score_type=score_type,
            timeout=timeout,
        )
        
        for i in range(len(batch_codes)):
            token_scores = batch_token_scores[i]
            start_idx, end_idx = code_token_indices[text_cursor]

            if end_idx > len(token_scores):
                print(f"Warning: Sample {text_cursor}'s end index {end_idx} is out of bounds for token length {len(token_scores)}. Slicing to the end.")
                code_scores = token_scores[start_idx:]
            else:
                code_scores = token_scores[start_idx:end_idx]

            if not code_scores:
                print(f"Warning: Sample {text_cursor} resulted in an empty score slice. Defaulting to 0.0.")
                mean_scores.append(0.0)
            else:
                mean_scores.append(np.mean(code_scores))
            
            text_cursor += 1
            
    return mean_scores


def score_texts_by_code_part_with_checkpoint(
    texts_to_score, code_token_indices, tokenizer, max_tokens, score_type, timeout, checkpoint_path
):
    from .inferutil import batch_by_tokens
    from .lmsync_util import score_many

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            results_map = pickle.load(f)
        print(f"Successfully loaded {len(results_map)} completed batches from '{checkpoint_path}'.")
    else:
        results_map = {}

    all_batches = batch_by_tokens(texts_to_score, tokenizer, max_tokens)
    
    batch_to_text_indices = []
    current_cursor = 0
    for batch in all_batches:
        batch_len = len(batch)
        batch_to_text_indices.append(list(range(current_cursor, current_cursor + batch_len)))
        current_cursor += batch_len

    batches_to_process_indices = [i for i in range(len(all_batches)) if i not in results_map]

    if not batches_to_process_indices:
        print("All batches have been processed in previous runs.")
    else:
        print(f"Total {len(all_batches)} batches, {len(batches_to_process_indices)} of which need to be processed.")

        pbar = tqdm(total=len(batches_to_process_indices), desc="Scoring batches with checkpointing")
        try:
            for batch_idx in batches_to_process_indices:
                batch_codes = all_batches[batch_idx]
                
                batch_token_scores = score_many(
                    batch_codes,
                    score_type=score_type,
                    timeout=timeout,
                )
                
                results_map[batch_idx] = batch_token_scores
                pbar.update(1)

                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(results_map, f)

        finally:
            pbar.close()
            print(f"\nCheckpoint saved to '{checkpoint_path}'.")
    
    mean_scores = []
    for batch_idx in range(len(all_batches)):
        batch_token_scores = results_map[batch_idx]
        original_indices_in_batch = batch_to_text_indices[batch_idx]

        for i, text_cursor in enumerate(original_indices_in_batch):
            token_scores = batch_token_scores[i]
            start_idx, end_idx = code_token_indices[text_cursor]

            if end_idx > len(token_scores):
                print(f"Warning: Sample {text_cursor}'s end index {end_idx} is out of bounds for token length {len(token_scores)}. Slicing to the end.")
                code_scores = token_scores[start_idx:]
            else:
                code_scores = token_scores[start_idx:end_idx]

            if not code_scores:
                print(f"Warning: Sample {text_cursor} resulted in an empty score slice. Defaulting to 0.0.")
                mean_scores.append(0.0)
            else:
                mean_scores.append(np.mean(code_scores))
                
    return mean_scores


