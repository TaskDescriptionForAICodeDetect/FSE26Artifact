from typing import List
from transformers import AutoTokenizer

def batch_by_tokens(
    texts: List[str],
    tokenizer: "AutoTokenizer",
    max_tokens: int,
) -> List[List[str]]:
    batches: List[List[str]] = []
    batch, cur_tokens = [], 0
    for t in texts:
        tok_len = len(tokenizer(t, add_special_tokens=False)["input_ids"])
        if batch and cur_tokens + tok_len > max_tokens:
            batches.append(batch)
            batch, cur_tokens = [], 0
        batch.append(t)
        cur_tokens += tok_len
    if batch:
        batches.append(batch)
    return batches

def lightweight_batch_by_words(texts: List[str], max_tokens: int) -> List[List[str]]:
    batches: List[List[str]] = []
    if not texts:
        return batches

    batch: List[str] = []
    cur_tokens = 0
    for raw in texts:
        t = raw if isinstance(raw, str) else str(raw)
        words = t.split()
        token_count = len(words)

        if token_count > max_tokens:
            if batch:
                batches.append(batch)
            truncated_text = " ".join(words[:max_tokens])
            batches.append([truncated_text])
            batch, cur_tokens = [], 0
            continue

        if batch and cur_tokens + token_count > max_tokens:
            batches.append(batch)
            batch, cur_tokens = [], 0

        batch.append(t)
        cur_tokens += token_count

    if batch:
        batches.append(batch)

    return batches