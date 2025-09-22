from __future__ import annotations
import os, statistics
from typing import Sequence, List, Dict, Any, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

__all__ = ["score_many"]


def _is_valid_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    must_have_any = [
        "tokenizer.json", "tokenizer.model", "vocab.json",
    ]
    must_have_config = [
        "config.json", "generation_config.json"
    ]
    files = set(os.listdir(path))
    has_tok = any(name in files for name in must_have_any)
    has_cfg = any(name in files for name in must_have_config)
    return has_tok and has_cfg


def _resolve_model_dir() -> str:
    env_dir = os.getenv("LLAMA_MODEL_DIR")
    fixed = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B"
    if env_dir and _is_valid_model_dir(env_dir):
        return env_dir
    if _is_valid_model_dir(fixed):
        return fixed
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "llama32-3b"),
        os.path.join(here, "..", "..", "llama32-3b"),
        os.path.join(here, "..", "..", "..", "llama32-3b"),
    ]
    for c in candidates:
        c_abs = os.path.abspath(c)
        if _is_valid_model_dir(c_abs):
            return c_abs
    return fixed

MODEL_DIR = _resolve_model_dir()
MAX_LENGTH = int(os.getenv("LLAMA_MAX_LENGTH", "4096"))
USE_DEVICE_MAP_AUTO = os.getenv("LLAMA_USE_DEVICE_MAP_AUTO", "0") == "1"

_tokenizer = None
_model = None
_device = None
_dtype = None
_use_auto = False


def _load_model_once():
    global _tokenizer, _model, _device, _dtype, _use_auto
    if _tokenizer is not None and _model is not None:
        return
    if torch.cuda.is_available():
        _device = "cuda"
    else:
        try:
            mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        except Exception:
            mps_available = False
        _device = "mps" if mps_available else "cpu"
    _dtype = torch.float16 if _device in ("cuda", "mps") else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _use_auto = False

    try:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=_dtype if _device in ("cuda", "mps") else None,
            trust_remote_code=True,
        ).to(_device).eval()
        _use_auto = False
        print(f"[lmsync_util] model loaded on device={_device}, dtype={_dtype}")
    except Exception as e:
        print(e)
        print(f"[lmsync_util] load to {_device} with dtype={_dtype} failed: {e}; retry CPU")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
        ).eval()
        _device = "cpu"
        _dtype = torch.float32
        _use_auto = False
        print("[lmsync_util] model loaded on CPU as fallback")


def _score_batch_local(texts: Sequence[str], metrics: List[str]) -> List[Dict[str, Any]]:
    _load_model_once()

    enc = _tokenizer(
        list(texts), return_tensors="pt", padding="longest", truncation=True, max_length=MAX_LENGTH
    )
    if not _use_auto and _device in ("cuda", "mps"):
        enc = {k: v.to(_device) for k, v in enc.items()}
    ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.inference_mode():
        logits = _model(input_ids=ids, attention_mask=attention_mask).logits[:, :-1]

    labels = ids[:, 1:]
    valid = attention_mask[:, 1:]

    nll = None
    if "perplexity" in metrics:
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    tok_logprob = None
    log_rank = None
    entropy = None
    if set(metrics) & {"log_probs", "log_rank", "entropy"}:
        log_probs = torch.log_softmax(logits, dim=-1)
        tok_logprob = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        if "log_rank" in metrics:
            ranks = (log_probs.exp() > tok_logprob.exp().unsqueeze(-1)).sum(-1) + 1
            log_rank = torch.log(ranks.float())
        if "entropy" in metrics:
            entropy = (-log_probs.exp() * log_probs).sum(-1)
        del log_probs

    del logits

    results: List[Dict[str, Any]] = []
    B = ids.size(0)
    for b in range(B):
        seq_len = int(valid[b].sum().item()) or 1
        item: Dict[str, Any] = {}
        if "log_probs" in metrics and tok_logprob is not None:
            item["log_probs"] = tok_logprob[b, :seq_len].cpu().tolist()
        if "log_rank" in metrics and log_rank is not None:
            item["log_rank"] = log_rank[b, :seq_len].cpu().tolist()
        if "entropy" in metrics and entropy is not None:
            item["entropy"] = entropy[b, :seq_len].cpu().tolist()
        if "perplexity" in metrics and nll is not None:
            item["perplexity"] = float(torch.exp(nll[b, :seq_len].mean()).item())
        results.append(item)

    del ids, labels, valid, attention_mask
    if nll is not None: del nll
    if tok_logprob is not None: del tok_logprob
    if log_rank is not None: del log_rank
    if entropy is not None: del entropy

    return results


def _extract_one(item: Dict[str, Any], score_type: str) -> Union[float, List[float], Dict[str, Any]]:
    if score_type == "perplexity":
        return float(item.get("perplexity", 0.0))
    if score_type in {"entropy", "log_rank", "log_probs"}:
        return item.get(score_type, [])
    if score_type.startswith("mean_"):
        field = score_type.replace("mean_", "")
        vals = item.get(field, [])
        return statistics.fmean(vals) if vals else 0.0
    if score_type == "all":
        return item
    raise ValueError(f"Unknown score_type: {score_type}")


def score_many(
    texts: Sequence[str],
    score_type: str = "perplexity",
    *,
    api_url: str = "http://127.0.0.1:8005",
    timeout: int = 120,
) -> List[Any]:
    _map: Dict[str, List[str]] = {
        "perplexity": ["perplexity"],
        "log_probs":  ["log_probs"],
        "log_rank":   ["log_rank"],
        "entropy":    ["entropy"],
        "all":        ["log_probs", "log_rank", "entropy", "perplexity"],
    }
    metrics: List[str] = _map.get(score_type, ["perplexity"])

    batch_json_results = _score_batch_local(texts, metrics)
    return [_extract_one(item, score_type) for item in batch_json_results]

