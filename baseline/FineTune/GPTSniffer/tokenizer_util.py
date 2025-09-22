from typing import List, Tuple
from transformers import AutoTokenizer
from pathlib import Path
import os

from baseutils.ioutils import Solution


def build_tokenizer(model_name: str = "microsoft/codebert-base"):
    def _resolve_local_path(name: str) -> str:
        try:
            p = Path(name)
            candidates = [
                p,
                Path(__file__).resolve().parent / name,
                Path(__file__).resolve().parents[2] / Path(name).name,
                Path.cwd() / name,
            ]
            for c in candidates:
                try:
                    if c.exists():
                        resolved = str(c.resolve())
                        print(f"[GPTSniffer] Resolved tokenizer path: {resolved}")
                        return resolved
                except Exception:
                    continue
        except Exception:
            pass
        return name

    model_name_resolved = _resolve_local_path(model_name)
    if not Path(model_name_resolved).exists():
        candidates_hint = [
            str(Path(model_name)),
            str((Path(__file__).resolve().parent / model_name)),
            str((Path(__file__).resolve().parents[2] / Path(model_name).name)),
            str((Path.cwd() / model_name)),
        ]
        raise FileNotFoundError(
            "[GPTSniffer] Local tokenizer directory not found, please confirm the path. Tried: " + ", ".join(candidates_hint)
        )
    return AutoTokenizer.from_pretrained(model_name_resolved, local_files_only=True)


def tokenize_solutions(
    solutions: List[Solution],
    tokenizer,
    max_length: int = 256,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    input_ids_list: List[List[int]] = []
    attn_masks_list: List[List[int]] = []
    labels = [s.label for s in solutions]

    for s in solutions:
        encoded = tokenizer(
            s.code,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids_list.append(encoded["input_ids"])
        attn_masks_list.append(encoded["attention_mask"])

    return input_ids_list, attn_masks_list, labels

