import sys
import os
import threading
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from LLMRequest.askLLM import ask_LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig

MASK_TOKEN = "<<<mask>>>"

_gen_lock = threading.Lock()
_gen_tokenizer: Optional[AutoTokenizer] = None
_gen_model: Optional[torch.nn.Module] = None
_gen_device: Optional[str] = None
_gen_dtype = None
_gen_model_id: Optional[str] = None

def _is_valid_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    has_tok = any(n in files for n in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    has_cfg = any(n in files for n in ("config.json", "generation_config.json"))
    return has_tok and has_cfg


def _resolve_fill_model() -> str:
    env_dir = os.getenv("FILL_MODEL_DIR")
    if env_dir and _is_valid_model_dir(env_dir):
        return env_dir
    fixed = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-3B"
    if _is_valid_model_dir(fixed):
        return fixed
    llama_dir = os.getenv("LLAMA_MODEL_DIR")
    if llama_dir and _is_valid_model_dir(llama_dir):
        return llama_dir
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def _resolve_device_and_dtype() -> tuple[str, Optional[torch.dtype]]:
    forced = os.getenv("LLAMA_DEVICE")
    if forced in {"cuda", "mps", "cpu"}:
        device = forced
    else:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            try:
                mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            except Exception:
                mps_available = False
            device = "mps" if mps_available else "cpu"

    env_dtype = (os.getenv("LLAMA_DTYPE") or "").lower()
    if env_dtype in {"float16", "fp16"}:
        dtype = torch.float16
    elif env_dtype in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif env_dtype in {"float32", "fp32"}:
        dtype = torch.float32
    else:
        if device == "cuda":
            dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        elif device == "mps":
            dtype = None
        else:
            dtype = torch.float32
    return device, dtype


def _load_local_generator_once(model_id: Optional[str] = None):
    global _gen_tokenizer, _gen_model, _gen_device, _gen_dtype, _gen_model_id
    target_model_id = model_id or _resolve_fill_model()
    if _gen_tokenizer is not None and _gen_model is not None and _gen_model_id == target_model_id:
        return
    with _gen_lock:
        if _gen_tokenizer is not None and _gen_model is not None and _gen_model_id == target_model_id:
            return
        _gen_device, _gen_dtype = _resolve_device_and_dtype()

        _gen_tokenizer = AutoTokenizer.from_pretrained(target_model_id, use_fast=True, trust_remote_code=True)
        if _gen_tokenizer.pad_token is None:
            _gen_tokenizer.pad_token = _gen_tokenizer.eos_token

        try:
            cfg = AutoConfig.from_pretrained(target_model_id, trust_remote_code=True)
            if getattr(cfg, "is_encoder_decoder", False):
                _gen_model = AutoModelForSeq2SeqLM.from_pretrained(
                    target_model_id,
                    torch_dtype=_gen_dtype if _gen_dtype is not None else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                _gen_model = AutoModelForCausalLM.from_pretrained(
                    target_model_id,
                    torch_dtype=_gen_dtype if _gen_dtype is not None else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            _gen_model = _gen_model.to(_gen_device).eval()
            _gen_model_id = target_model_id
            print(f"[DetectGPT.fill] local generator '{target_model_id}' on device={_gen_device}, dtype={_gen_dtype}")
        except Exception as e:
            print(f"[DetectGPT.fill] load local generator failed: {e}; falling back to CPU")
            _gen_model = AutoModelForSeq2SeqLM.from_pretrained(target_model_id, trust_remote_code=True).eval()
            _gen_device = "cpu"
            _gen_dtype = torch.float32
            _gen_model_id = target_model_id


def _generate_local(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, model_id: Optional[str] = None) -> str:
    _load_local_generator_once(model_id)
    inputs = _gen_tokenizer(prompt, return_tensors="pt")
    if _gen_device in ("cuda", "mps"):
        inputs = {k: v.to(_gen_device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = _gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=_gen_tokenizer.eos_token_id,
        )
    text = _gen_tokenizer.decode(out[0], skip_special_tokens=True)
    return text

def fill_one(
    masked_text: str,
    model: str = "local",
    temperature: float = 0.7,
    max_retry: int = 3,
    max_new_tokens: int = 256,
    local_model_name: Optional[str] = None,
    **kwargs,
) -> str:
    prompt = (
        "Your task is to act as a code completion engine. I will provide a piece of code with a "
        "masked section, indicated by '<<<mask>>>'. Please fill in this mask and return the "
        "complete, filled code block.\n\n"
        "IMPORTANT: Do not add any explanations, introductory text, code block formatting (like ```), or any other text. "
        "Your response should be ONLY the raw, completed code snippet.\n\n"
        "Here is the masked code:\n\n"
        "{masked_code}"
    )
    
    messages = [
        {"role": "user", "content": prompt.format(masked_code=masked_text)}
    ]
    
    try:
        if str(model).lower().startswith("local"):
            override = local_model_name
            if ":" in str(model):
                override = str(model).split(":", 1)[1].strip() or override
            filled_code = _generate_local(
                messages[0]["content"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model_id=override,
            )
        else:
            filled_code = ask_LLM(model=model, messages=messages, temperature=temperature, max_retry=max_retry)

        if filled_code.startswith("```") and filled_code.endswith("```"):
            lines = filled_code.split('\n')
            if len(lines) > 1:
                filled_code = '\n'.join(lines[1:-1])
        return filled_code.strip()
    except Exception as e:
        print(f"[ERROR] Filling failed for a text snippet due to: {e}. Returning original text.")
        return masked_text
