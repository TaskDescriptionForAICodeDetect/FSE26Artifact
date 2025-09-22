#!/usr/bin/env python3

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from baseutils.ioutils import Solution
from LLMRequest.askLLM import ask_LLM

DEFAULT_EMBED_MODEL = "microsoft/graphcodebert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_COMMENT_PAT = re.compile(r"^\s*#.*$", re.M)
_TRIPLE_STR = re.compile(r'"""(.*?)"""|''\'(.*?)\'\'\'', re.S)

def strip_comments(code: str) -> str:
    code = re.sub(_TRIPLE_STR, "", code)
    code = re.sub(_COMMENT_PAT, "", code)
    return "\n".join(ln for ln in code.splitlines() if ln.strip())

def extract_first_code_block(text: str) -> str | None:
    pattern = re.compile(r"```[\w\s]*\n(.*?)\n```", re.S)
    match = pattern.search(text)
    return match.group(1).strip() if match else None

def build_rewrite_prompt(solution: Solution, with_real_task: bool) -> List[Dict[str, str]]:
    cleaned_code = strip_comments(solution.code)
    
    instruction = (
        f"Your task is to first provide a step-by-step explanation of the given {solution.language} code's functionality, "
        f"and then rewrite the code in {solution.language}. "
        "The rewritten code should be enclosed in a single markdown code block. "
        "Do not add any other words or clarifications outside the code block."
    )
    
    if with_real_task:
        language = solution.language
        description = solution.description
        prompt_content = (
            f"Please write a solution for the following problem in {language}:\n\n"
            f"{description}\n\n"
            f"Provide only the code without any explanations."
        )
    else:
        prompt_content = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Original Code:\n```\n{cleaned_code}\n```"
        )
        
    return [{"role": "user", "content": prompt_content}]


def rewrite_n_times(
    solution: Solution,
    config: Dict[str, Any],
    n_needed: int,
    *,
    tokenizer: Optional[AutoTokenizer] = None,
) -> List[str]:
    rewritten_codes: List[str] = []

    for _ in range(n_needed):
        try:
            messages = build_rewrite_prompt(solution, config["with_real_task"])
            reply = ask_LLM(
                config["detector_llm"],
                messages,
                temperature=config["temperature"],
                max_retry=config.get("max_retry", 3)
            )
            if reply and isinstance(reply, str):
                extracted = extract_first_code_block(reply) or reply
                if extracted:
                    cleaned = strip_comments(extracted)
                    if cleaned:
                        rewritten_codes.append(cleaned)
        except Exception as e:
            print(f"  [WARN] An error occurred during rewrite: {e}")
            continue

    return rewritten_codes

class CodeEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: AutoModel = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print(f"CodeEmbedder initialized with {model_name} on {DEVICE}")

    @torch.no_grad()
    def embed(self, code_str: str) -> torch.Tensor:
        toks = self.tokenizer(code_str, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        out = self.model(**toks).last_hidden_state[:, 0]
        return F.normalize(out, dim=-1).squeeze(0).cpu()

def similarity_score(code: str, rewrites: List[str], embedder: CodeEmbedder) -> float:
    if not rewrites:
        return 0.0
    src_emb = embedder.embed(code)
    sims = [F.cosine_similarity(src_emb, embedder.embed(r), dim=0).item() for r in rewrites]
    return sum(sims) / len(sims)
