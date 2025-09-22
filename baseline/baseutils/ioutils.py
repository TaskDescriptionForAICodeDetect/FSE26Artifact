import sys, os, json, argparse, random, pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional


MAX_SOL_PER_TYPE = 1
LANGUAGE_FILTER  = "python"
random.seed(0)

def iter_samples(root: Path, max_per_type: int = MAX_SOL_PER_TYPE, lang: str = LANGUAGE_FILTER):
    tasks_dir = root / "tasks"
    all_tasks = [i for i in tasks_dir.iterdir()]
    all_tasks.sort()
    for task in all_tasks:
        if not task.is_dir():
            continue

        ai_file    = task / "solutions" / "ai"    / "ai_solution.json"
        human_file = task / "solutions" / "human" / "human_solution.json"
        if not (ai_file.exists() and human_file.exists()):
            continue

        ai_recs = [rec for rec in json.loads(ai_file.read_text())
                   if rec.get("language", "").lower() == lang]
        if len(ai_recs) > max_per_type:
            ai_recs = random.sample(ai_recs, max_per_type)
        for rec in ai_recs:
            yield 1, rec["code"], task.name

        human_recs = [rec for rec in json.loads(human_file.read_text())
                      if rec.get("language", "").lower() == lang]
        if len(human_recs) > max_per_type:
            human_recs = random.sample(human_recs, max_per_type)
        for rec in human_recs:
            yield 0, rec["code"], task.name


def get_task_label(root_path):
    samples = list(iter_samples(Path(root_path)))
    codes = [code for _, code, _ in samples]
    label_list = [y for y, _, _ in samples]
    task_list = [t for _, _, t in samples]
    return codes, label_list, task_list


def iter_samples_with_desc(root: Path, max_per_type: int = MAX_SOL_PER_TYPE, lang: str = LANGUAGE_FILTER, process_desc_func=None):
    tasks_dir = root / "tasks"
    all_tasks = [i for i in tasks_dir.iterdir()]
    all_tasks.sort()
    for task in all_tasks:
        if not task.is_dir():
            continue

        desc_file = task / "description.txt"
        if not desc_file.exists():
            continue
        
        description = desc_file.read_text()
        if process_desc_func:
            description = process_desc_func(description)

        ai_file    = task / "solutions" / "ai"    / "ai_solution.json"
        human_file = task / "solutions" / "human" / "human_solution.json"
        if not (ai_file.exists() and human_file.exists()):
            continue

        ai_recs = [rec for rec in json.loads(ai_file.read_text())
                   if rec.get("language", "").lower() == lang]
        if len(ai_recs) > max_per_type:
            ai_recs = random.sample(ai_recs, max_per_type)
        for rec in ai_recs:
            yield 1, rec["code"], task.name, description

        human_recs = [rec for rec in json.loads(human_file.read_text())
                      if rec.get("language", "").lower() == lang]
        if len(human_recs) > max_per_type:
            human_recs = random.sample(human_recs, max_per_type)
        for rec in human_recs:
            yield 0, rec["code"], task.name, description

def get_task_label_with_desc(root_path, process_desc_func=None):
    samples = list(iter_samples_with_desc(Path(root_path), process_desc_func=process_desc_func))
    codes = [code for _, code, _, _ in samples]
    label_list = [y for y, _, _, _ in samples]
    task_list = [t for _, _, t, _ in samples]
    descriptions = [d for _, _, _, d in samples]
    return codes, label_list, task_list, descriptions

@dataclass
class Solution:
    code: str
    label: int
    task_name: str
    description: str
    language: str
    model: Optional[str] = "Human"

def iter_solutions(root: Path, process_desc_func=None, supported_languages=None):
    if supported_languages is None:
        supported_languages = ["python", "c", "c++", "java"]
    
    tasks_dir = root / "tasks"
    all_tasks = sorted([i for i in tasks_dir.iterdir() if i.is_dir()])

    for task in tqdm(all_tasks, desc="Loading solutions"):
        desc_file = task / "description.txt"
        if not desc_file.exists():
            continue
        description = desc_file.read_text()
        if process_desc_func:
            description = process_desc_func(description)

        dataset_source = task.name.split('_')[0].lower() if '_' in task.name else "unknown"

        ai_file = task / "solutions" / "ai" / "ai_solution.json"
        if ai_file.exists():
            ai_recs = json.loads(ai_file.read_text())
            recs_by_lang_model = {}
            for rec in ai_recs:
                lang = rec.get("language", "unknown").lower()
                model = rec.get("model", "unknown")
                if lang not in recs_by_lang_model:
                    recs_by_lang_model[lang] = {}
                if model not in recs_by_lang_model[lang]:
                    recs_by_lang_model[lang][model] = []
                recs_by_lang_model[lang][model].append(rec)
            
            for lang in recs_by_lang_model:
                if supported_languages and lang not in supported_languages:
                    continue
                if dataset_source == "codenet" and lang == "python":
                    continue
                for model in recs_by_lang_model[lang]:
                    chosen_rec = recs_by_lang_model[lang][model][0]
                    yield Solution(
                        code=chosen_rec["code"],
                        label=1,
                        task_name=task.name,
                        description=description,
                        language=lang,
                        model=model or "unknown"
                    )

        human_file = task / "solutions" / "human" / "human_solution.json"
        if human_file.exists():
            human_recs = json.loads(human_file.read_text())
            recs_by_lang = {}
            for rec in human_recs:
                lang = rec.get("language", "unknown").lower()
                if lang not in recs_by_lang:
                    recs_by_lang[lang] = []
                recs_by_lang[lang].append(rec)

            for lang, recs_in_lang in recs_by_lang.items():
                if supported_languages and lang not in supported_languages:
                    continue
                if dataset_source == "codenet" and lang == "python":
                    continue
                chosen_rec = recs_in_lang[0]
                yield Solution(
                    code=chosen_rec["code"],
                    label=0,
                    task_name=task.name,
                    description=description,
                    language=lang,
                    model="Human"
                )

def get_all_solutions(root_path, process_desc_func=None, supported_languages=None) -> list[Solution]:
    return list(iter_solutions(Path(root_path), process_desc_func=process_desc_func, supported_languages=supported_languages))

def infer_dataset_source_from_task(task_name: str) -> str:
    if not task_name:
        return "unknown"
    prefix = task_name.split('_', 1)[0].lower() if '_' in task_name else "unknown"
    return prefix if prefix in ("apps", "codenet") else "unknown"


def pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.00%"
    return f"{(part / whole) * 100:.2f}%"


def print_solution_sample_distribution(solutions: list[Solution]):
    total = len(solutions)
    print("\n" + "-" * 16 + " Sample Distribution (solutions-level) " + "-" * 16)
    print(f"Total solutions (after sampling): {total}")

    from collections import defaultdict
    dataset_counts = defaultdict(int)
    language_counts = defaultdict(int)
    nested = defaultdict(lambda: defaultdict(lambda: {"count": 0, "orig_models": defaultdict(int)}))

    for s in solutions:
        ds = infer_dataset_source_from_task(s.task_name)
        lang = s.language or "unknown"
        model = s.model or "unknown"
        dataset_counts[ds] += 1
        language_counts[lang] += 1
        nested[ds][lang]["count"] += 1
        nested[ds][lang]["orig_models"][model] += 1

    print("\n- Dataset proportions:")
    for ds, cnt in sorted(dataset_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {ds:8s}  {cnt:6d}  ({pct(cnt, total)})")

    lang_total = sum(language_counts.values())
    print("\n- Language proportions:")
    for lang, cnt in sorted(language_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {lang:8s}  {cnt:6d}  ({pct(cnt, lang_total)})")

    print("\n- Per (dataset × language): original model distribution:")
    for ds in sorted(nested.keys()):
        for lang in sorted(nested[ds].keys()):
            group = nested[ds][lang]
            group_count = group["count"]
            print(f"\n  [{ds} | {lang}]  solutions={group_count}")
            orig_models = group["orig_models"]
            if orig_models:
                for model_name, cnt in sorted(orig_models.items(), key=lambda x: (-x[1], x[0])):
                    print(f"    - {model_name:24s} count={cnt:5d} ({pct(cnt, group_count)})")
            else:
                print("    (no original model info)")


def sample_solutions_by_task(solutions: list[Solution], sample_per_dataset: int = 100, random_state: int = 42) -> list[Solution]:

    tasks_by_dataset = {"apps": set(), "codenet": set()}
    for s in solutions:
        ds = infer_dataset_source_from_task(s.task_name)
        if ds in tasks_by_dataset:
            tasks_by_dataset[ds].add(s.task_name)

    selected_per_ds = {}
    for ds, tasks in tasks_by_dataset.items():
        tasks_list = sorted(list(tasks))
        selected = set(tasks_list[: int(sample_per_dataset)])
        selected_per_ds[ds] = selected

    selected_task_set = set().union(*selected_per_ds.values())
    before_count = len(solutions)
    filtered = [s for s in solutions if s.task_name in selected_task_set]
    after_count = len(filtered)

    print(f"[Sampling] Task sampling (name-ordered) applied: apps={len(selected_per_ds['apps'])}, codenet={len(selected_per_ds['codenet'])} | solutions: {before_count} -> {after_count}")

    return filtered
