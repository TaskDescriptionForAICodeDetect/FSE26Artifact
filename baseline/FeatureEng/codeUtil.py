"""
This module provides utilities for extracting lexical and structural features from source code
using tree-sitter for robust parsing.

Required packages:
pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-cpp tree-sitter-c scikit-learn numpy tqdm
"""

from __future__ import annotations
import importlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple
import numpy as np
from tree_sitter import Language, Parser
import keyword as py_keyword
_LANG_PKG: Dict[str, str] = {
    "python": "tree_sitter_python",
    "java": "tree_sitter_java",
    "c++": "tree_sitter_cpp",
    "c": "tree_sitter_c",
}
_CTRL_NODES = {
    "if_statement", "for_statement", "while_statement", "switch_statement",
    "for_in_statement", "elif_clause", "else_clause", "try_statement", "with_statement",
    "do_statement", "case_statement", "catch_clause", "finally_clause",
}
@lru_cache
def _get_parser(lang: str) -> Parser:
    pkg = _LANG_PKG.get(lang)
    if not pkg:
        raise ValueError(f"Language '{lang}' is not supported. Add it to _LANG_PKG.")
    mod = importlib.import_module(pkg)
    return Parser(Language(mod.language()))
def classify_lines_ts(code: str, lang: str) -> Tuple[List[str], int, int]:
    lines = code.splitlines()
    n_lines = len(lines)
    if n_lines == 0:
        return [], 0, 0
    blank_mask = [ln.strip() == "" for ln in lines]
    comment_mask = [False] * n_lines
    parser = _get_parser(lang)
    tree = parser.parse(code.encode("utf8"))
    q = [tree.root_node]
    while q:
        node = q.pop(0)
        if "comment" in node.type:
            for i in range(node.start_point[0], node.end_point[0] + 1):
                comment_mask[i] = True
        q.extend(node.children)
    code_lines, blank_cnt, comment_cnt = [], 0, 0
    for i, raw_line in enumerate(lines):
        if blank_mask[i]:
            blank_cnt += 1
        elif comment_mask[i]:
            comment_cnt += 1
        else:
            code_lines.append(raw_line)
    return code_lines, blank_cnt, comment_cnt

def line_length_stats(code_lines: Sequence[str]) -> Tuple[float, float]:
    if not code_lines:
        return 0.0, 0.0
    lengths = [len(line) for line in code_lines]
    return np.mean(lengths), float(np.max(lengths))

def compute_max_nesting(code: str, lang: str) -> int:
    parser = _get_parser(lang)
    tree = parser.parse(code.encode("utf8"))
    def _max_depth(node, depth=0):
        new_depth = depth + 1 if node.type in _CTRL_NODES else depth
        if not node.children:
            return new_depth
        return max([_max_depth(c, new_depth) for c in node.children])
    return _max_depth(tree.root_node)
def _compute_line_masks(code: str, lang: str) -> Tuple[List[str], List[bool], List[bool]]:
    lines = code.splitlines()
    n_lines = len(lines)
    if n_lines == 0:
        return [], [], []
    blank_mask = [ln.strip() == "" for ln in lines]
    comment_mask = [False] * n_lines
    parser = _get_parser(lang)
    tree = parser.parse(code.encode("utf8"))
    q = [tree.root_node]
    while q:
        node = q.pop(0)
        if "comment" in node.type:
            for i in range(node.start_point[0], node.end_point[0] + 1):
                if 0 <= i < n_lines:
                    comment_mask[i] = True
        q.extend(node.children)
    return lines, blank_mask, comment_mask

def _iter_function_like_nodes(lang: str, root_node):
    type_map = {
        "python": {"function_definition"},
        "java": {"method_declaration", "constructor_declaration"},
        "c": {"function_definition"},
        "c++": {"function_definition"},
    }
    targets = type_map.get(lang, set())
    q = [root_node]
    while q:
        node = q.pop(0)
        if node.type in targets:
            yield node
        q.extend(node.children)

def compute_function_stats(code: str, lang: str) -> Tuple[int, float]:
    lines, blank_mask, comment_mask = _compute_line_masks(code, lang)
    if not lines:
        return 0, 0.0
    parser = _get_parser(lang)
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node
    body_locs: List[int] = []
    for fn in _iter_function_like_nodes(lang, root):
        start = fn.start_point[0]
        end = fn.end_point[0]
        loc = 0
        for i in range(start, end + 1):
            if 0 <= i < len(lines) and not blank_mask[i] and not comment_mask[i]:
                loc += 1
        body_locs.append(loc)
    total_functions = len(body_locs)
    avg_loc = float(np.mean(body_locs)) if body_locs else 0.0
    return total_functions, avg_loc

def _get_language_keywords(lang: str) -> set:
    if lang == "python":
        return set(py_keyword.kwlist) | {"True", "False", "None"}
    if lang == "java":
        return {
            "abstract","assert","boolean","break","byte","case","catch","char","class","const","continue",
            "default","do","double","else","enum","extends","final","finally","float","for","goto","if",
            "implements","import","instanceof","int","interface","long","native","new","package","private",
            "protected","public","return","short","static","strictfp","super","switch","synchronized","this",
            "throw","throws","transient","try","void","volatile","while","true","false","null","record","yield","var"
        }
    if lang == "c":
        return {
            "auto","break","case","char","const","continue","default","do","double","else","enum","extern",
            "float","for","goto","if","inline","int","long","register","restrict","return","short","signed",
            "sizeof","static","struct","switch","typedef","union","unsigned","void","volatile","while",
            "_Alignas","_Alignof","_Atomic","_Bool","_Complex","_Generic","_Imaginary","_Noreturn","_Static_assert","_Thread_local"
        }
    if lang == "c++":
        return {
            "alignas","alignof","and","and_eq","asm","auto","bitand","bitor","bool","break","case","catch",
            "char","char16_t","char32_t","class","compl","concept","const","consteval","constexpr","constinit","const_cast",
            "continue","co_await","co_return","co_yield","decltype","default","delete","do","double","dynamic_cast","else",
            "enum","explicit","export","extern","false","final","float","for","friend","goto","if","inline","int",
            "long","mutable","namespace","new","noexcept","not","not_eq","nullptr","operator","or","or_eq","override",
            "private","protected","public","register","reinterpret_cast","requires","return","short","signed","sizeof",
            "static","static_assert","static_cast","struct","switch","template","this","thread_local","throw","true",
            "try","typedef","typeid","typename","union","unsigned","using","virtual","void","volatile","wchar_t",
            "while","xor","xor_eq"
        }
    return set()

def compute_keyword_token_ratio(code: str, lang: str) -> float:
    code_lines, _, _ = classify_lines_ts(code, lang)
    if not code_lines:
        return 0.0
    text = "\n".join(code_lines)
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", text)
    if not tokens:
        return 0.0
    kws = _get_language_keywords(lang)
    kw_count = sum(1 for t in tokens if t in kws)
    return float(kw_count) / float(len(tokens))

def extract_features(code: str, lang: str, task_name: str) -> np.ndarray:
    try:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Input code is empty or not a string.")
        _, blank_cnt, _ = classify_lines_ts(code, lang)
        max_nest = compute_max_nesting(code, lang)
        total_functions, avg_func_loc = compute_function_stats(code, lang)
        keyword_ratio = compute_keyword_token_ratio(code, lang)
        feats = np.array([
            float(avg_func_loc),
            float(blank_cnt),
            float(max_nest),
            float(total_functions),
            float(keyword_ratio),
        ], dtype=np.float32)
        return feats
    except (ValueError, Exception) as e:
        print(f"\n{'='*20} PARSE FAILED {'='*20}")
        print(f"Task: {task_name}, Lang: {lang}")
        print(f"Reason: {e}")
        print(f"Code Snippet:\n------------------------------\n{code}\n")
        return np.zeros(5, dtype=np.float32)
