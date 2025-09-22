import sys, os, json, pickle
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from Embedding.utils.AST import code_with_ast_task, code_with_task, code_with_ast
from baseutils.ioutils import (
    get_all_solutions,
    Solution,
)
from baseutils.taskUtil import get_description_processor
from baseutils.stasticutils import evaluate_within_across_classifier

def get_local_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str | None = None,
    torch_dtype: str | None = None,
) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            try:
                mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            except Exception:
                mps_available = False
            device = "mps" if mps_available else "cpu"

    if torch_dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        if isinstance(torch_dtype, str):
            dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(torch_dtype.lower(), torch.float32)
        else:
            dtype = torch_dtype

    print(f"Using device: {device}, dtype: {str(dtype)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype if device in ("cuda",) else None,
        ).to(device).eval()
    except Exception:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def embed_batch(batch: List[str]) -> torch.Tensor:
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if model.config.is_encoder_decoder:
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            token_embeddings = outputs.pooler_output
            return token_embeddings.cpu()
        last_hidden = outputs[0]
        sum_embeddings = torch.sum(last_hidden * attention_mask.unsqueeze(-1), dim=1)
        seq_lengths = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        embeddings = sum_embeddings / seq_lengths
        return embeddings.cpu()

    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing Local Embeddings"):
        feats.append(embed_batch(texts[i: i + batch_size]))
    return torch.cat(feats).numpy()



def generate_filename_prefix(config: dict) -> str:
    """Generates a unique and readable filename prefix based on the config."""
    prefix_parts = ['embedding', 'local']
    
    model_name = config.get('local_model_name', 'unknown').split('/')[-1]
    max_len = config.get('max_length')
    if max_len:
        prefix_parts.append(f"len_{max_len}")
    prefix_parts.append(f"model_{model_name}")
    
    code_type = config.get('code_type', 'plain')
    prefix_parts.append(f"type_{code_type}")

    if code_type in ['task', 'ast_task'] and config.get('desc_processor_name'):
        prefix_parts.append(f"proc_{config['desc_processor_name']}")

    return "_".join(prefix_parts)

def main(**config) -> Dict:
    """Main function to run a single embedding-based classification experiment."""
    print("\n" + "="*50)
    print("--- Starting Experiment with a new configuration ---")
    for key, val in config.items():
        print(f"  - {key}: {val}")
    print("="*50 + "\n")

    filename_prefix = generate_filename_prefix(config)
    embedding_cache_path = Path(f"./{filename_prefix}_embeddings.pkl")

    processor = get_description_processor(config['desc_processor_name']) if config.get('desc_processor_name') else None
    if config.get('desc_processor_name'):
        print(f"Applying description processor: {config['desc_processor_name']}")
    solutions = get_all_solutions(config['root_path'], process_desc_func=processor)

    code_type = config.get("code_type", "plain")
    processed_codes = []
    if code_type == "plain":
        processed_codes = [s.code for s in solutions]
    elif code_type == "ast":
        processed_codes = [code_with_ast(s.code) for s in tqdm(solutions, desc="Appending AST")]
    elif code_type == "task":
        processed_codes = [code_with_task(s.code, s.description) for s in tqdm(solutions, desc="Appending Task")]
    elif code_type == "ast_task":
        processed_codes = [code_with_ast_task(s.code, s.description) for s in tqdm(solutions, desc="Appending AST & Task")]

    if embedding_cache_path.exists():
        print(f"Loading cached embeddings from {embedding_cache_path}...")
        with open(embedding_cache_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("No cached embeddings found. Computing now...")
        embeddings = get_local_embeddings(
            processed_codes,
            config['local_model_name'],
            config['batch_size'],
            config.get('max_length', 1024),
            device=config.get('device'),
            torch_dtype=config.get('torch_dtype'),
        )
        print(f"Saving embeddings to {embedding_cache_path}...")
        with open(embedding_cache_path, "wb") as f:
            pickle.dump(embeddings, f)

    labels = np.array([s.label for s in solutions])

    if len(embeddings) != len(labels):
        print(f"[ERROR] Mismatch between embeddings count ({len(embeddings)}) and labels count ({len(labels)}).")
        print("This is likely due to a stale cache. Please delete the .pkl file and rerun.")
        print(f"Cache file to delete: {embedding_cache_path}")
        return { 
            "config": config,
            "error": "Data-cache mismatch",
            "classification_report": None, 
            "confusion_matrix": None 
        }

    evaluate_within_across_classifier(
        solutions=solutions,
        X=embeddings,
        labels=labels,
        test_size=config.get('test_size', 0.3),
        random_state=config.get('random_state', 42),
        rf_params=config.get('rf_params', {}),
    )

    return {"config": config}


def run_all_experiments():
    """Defines and runs a series of experiments."""
    base_path = "../splitDescription/split_benchmark/"
    
    base_config = {
        "root_path": base_path,
        "batch_size": 128,
        "test_size": 0.8,
        "random_state": 42,
        "max_length": 1024,
        "local_model_name": "../../Qwen3-Embedding-0.6B",
        "rf_params": dict(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42),
    }

    experiment_configs = [
        {"code_type": "ast"},
        {"code_type": "ast_task"},
    ]

    all_results = []
    for exp_config in experiment_configs:
        full_config = {**base_config, **exp_config}
        result = main(**full_config)
        all_results.append(result)

    summary_path = "embedding_experiment_summary.json"
    print(f"\n{'='*30}\nExperiment summary saved to -> {summary_path}\n{'='*30}")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_all_experiments()
