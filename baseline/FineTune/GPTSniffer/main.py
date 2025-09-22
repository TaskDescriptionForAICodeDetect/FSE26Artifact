import sys, os
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from baseutils.ioutils import get_all_solutions, Solution
from baseutils.stasticutils import evaluate_within_across_threshold

from FineTune.GPTSniffer.extractor import apply_extractor, ExtractorConfig
from FineTune.GPTSniffer.tokenizer_util import build_tokenizer, tokenize_solutions
from FineTune.GPTSniffer.classifier import CodeDataset, train_classifier


def generate_filename_prefix(config: dict) -> str:
    parts = ["gptsniff", config.get('backbone', 'codebert').split('/')[-1]]
    if config.get('use_extractor', True):
        parts.append("ext")
    parts.append(f"max{config.get('max_length', 256)}")
    if config.get('attach_task', False):
        parts.append("task")
    return "_".join(parts)


def train_and_score(solutions: List[Solution], config: dict):
    if config.get('use_extractor', True):
        solutions = apply_extractor(solutions, ExtractorConfig())

    tok = build_tokenizer(config.get('backbone', 'microsoft/codebert-base'))

    if config.get('attach_task', False):
        def _attach(s: Solution) -> Solution:
            lang = (s.language or "").lower()
            if lang in ("java", "c", "c++"):
                prefix = f"// Task: {s.task_name}\n"
            else:
                prefix = f"# Task: {s.task_name}\n"
            return Solution(
                code=prefix + s.code,
                label=s.label,
                task_name=s.task_name,
                description=s.description,
                language=s.language,
                model=s.model,
            )
        solutions = [_attach(s) for s in solutions]

    input_ids, attn_mask, labels = tokenize_solutions(
        solutions,
        tok,
        max_length=config.get('max_length', 256),
    )
    
    tasks = [s.task_name for s in solutions]
    unique_tasks = sorted(list(set(tasks)))
    train_tasks, val_tasks = train_test_split(unique_tasks, test_size=0.2, random_state=config.get('random_state', 42))
    task_to_indices = {}
    for idx, t in enumerate(tasks):
        task_to_indices.setdefault(t, []).append(idx)
    train_idx = [i for t in train_tasks for i in task_to_indices[t]]
    val_idx = [i for t in val_tasks for i in task_to_indices[t]]

    train_ds = CodeDataset([input_ids[i] for i in train_idx], [attn_mask[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = CodeDataset([input_ids[i] for i in val_idx], [attn_mask[i] for i in val_idx], [labels[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 16), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 16))

    model = train_classifier(
        train_loader,
        val_loader,
        model_name=config.get('backbone', 'microsoft/codebert-base'),
        epochs=config.get('epochs', 1),
        lr=config.get('lr', 2e-5)
    )
    
    import torch
    from torch.nn import functional as F
    model.eval()
    device = next(model.parameters()).device
    
    all_scores = []
    with torch.no_grad():
        for ids, mask in zip(input_ids, attn_mask):
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            attn = torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x, attention_mask=attn, labels=None)
            prob_ai = F.softmax(logits, dim=-1)[0, 1].item()
            all_scores.append(prob_ai)
    
    return solutions, all_scores


def main(**config):
    prefix = generate_filename_prefix(config)
    print(f"\n==== Running GPTSniffer: {prefix} ====")
    solutions = get_all_solutions(config['root_path'])
    solutions, scores = train_and_score(solutions, config)

    print("\n--- Evaluating (within / across) ---")
    from numpy import array
    labels = array([s.label for s in solutions])
    flipped_scores = -array(scores)
    evaluate_within_across_threshold(
        solutions=solutions,
        labels=labels,
        scores=flipped_scores,
        test_size=config.get('test_size', 0.3),
        random_state=config.get('random_state', 42),
    )
    


def run_all():
    base = {
        'root_path': "../../splitDescription/split_benchmark",
        'backbone': '../../../cdbert',
        'use_extractor': True,
        'max_length': 500,
        
        'batch_size': 2,
        'epochs': 1,
        'lr': 2e-5,
        'test_size': 0.8,
        'random_state': 42,
    }

    main(**{**base, 'attach_task': False})
    main(**{**base, 'attach_task': True})


if __name__ == "__main__":
    run_all()

