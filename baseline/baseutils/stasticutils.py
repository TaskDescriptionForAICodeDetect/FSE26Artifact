import sys, os, json, argparse, random, pickle
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from baseutils.ioutils import Solution

MAX_SOL_PER_TYPE = 1
LANGUAGE_FILTER  = "python"
random.seed(0)


def search_best_threshold(labels: np.ndarray, scores: np.ndarray, step: float = 0.01):
    min_s, max_s = scores.min(), scores.max()
    thresholds = np.arange(min_s, max_s + step, step)
    best_thr, best_f1 = None, -1.0

    for thr in thresholds:
        preds = (scores < thr).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1


def evaluate_within_across_threshold(
    solutions: List[Solution],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    step: float = 0.01,
    test_size: float = 0.3,
    random_state: int = 42,
):
    languages = sorted(list(set(s.language for s in solutions)))
    models = sorted(list(set(s.model for s in solutions if s.model != "Human")))
    rs = np.random.RandomState(random_state)

    model_within_preds = {m: {"y_true": [], "y_pred": []} for m in models}
    model_across_preds = {m: {"y_true": [], "y_pred": []} for m in models}

    overall_within_model_f1s = []
    overall_across_model_f1s = []

    print("\n" + "="*24 + " Per-Language Results (Task-based Split) " + "="*24)
    overall_within_lang_avgs = []
    overall_across_lang_avgs = []
    for lang in languages:
        lang_indices = [i for i, s in enumerate(solutions) if s.language == lang]
        if len(lang_indices) < 2: continue

        unique_tasks_in_lang = sorted(list(set(solutions[i].task_name for i in lang_indices)))
        if len(unique_tasks_in_lang) < 2:
            print(f"\n--- Language: {lang} --- \n  > Skip: Not enough unique tasks.")
            continue
        
        train_tasks, test_tasks = train_test_split(
            unique_tasks_in_lang, test_size=test_size, random_state=random_state
        )

        task_to_indices = defaultdict(list)
        for i in lang_indices:
            task_to_indices[solutions[i].task_name].append(i)

        train_indices_pool = [i for task in train_tasks for i in task_to_indices[task]]
        test_indices_pool = [i for task in test_tasks for i in task_to_indices[task]]
        
        lang_models = sorted(list(set(solutions[i].model for i in lang_indices if solutions[i].model != "Human")))
        if not lang_models: continue

        print(f"\n--- Language: {lang} ---")

        within_f1_list = []
        print("  > Within (per model)")
        for m in lang_models:
            train_idx = [i for i in train_indices_pool if solutions[i].model == m or solutions[i].label == 0]
            test_idx = [i for i in test_indices_pool if solutions[i].model == m or solutions[i].label == 0]

            if len(train_idx) < 2 or len(test_idx) < 2 or len(set(labels[train_idx])) < 2:
                print(f"    - Skip model {m}: not enough samples/classes in train/test tasks.")
                continue

            X_train, y_train = scores[train_idx], labels[train_idx]
            X_test, y_test = scores[test_idx], labels[test_idx]
            thr, _ = search_best_threshold(y_train, X_train, step=step)
            y_pred = (X_test < thr).astype(int)
            within_f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            print(f"    [Within] Model: {m}")
            print(classification_report(y_test, y_pred, digits=4, zero_division=0))
            model_within_preds[m]["y_true"].extend(list(y_test))
            model_within_preds[m]["y_pred"].extend(list(y_pred))

        if within_f1_list:
            overall_within_model_f1s.extend(within_f1_list)
            lang_within_avg = np.mean(within_f1_list)
            print(f"  > Within (avg macro F1 over models): {lang_within_avg:.4f}")
            overall_within_lang_avgs.append(lang_within_avg)

        across_f1_list = []
        print("  > Across (per model)")
        for m in lang_models:
            human_train_idx = [i for i in train_indices_pool if solutions[i].label == 0]
            other_ai_train_idx_pool = [i for i in train_indices_pool if solutions[i].label == 1 and solutions[i].model != m]
            
            human_test_idx = [i for i in test_indices_pool if solutions[i].label == 0]
            target_ai_test_idx_pool = [i for i in test_indices_pool if solutions[i].model == m]

            if not all([human_train_idx, other_ai_train_idx_pool, human_test_idx, target_ai_test_idx_pool]):
                print(f"    - Skip model {m}: insufficient data in task pools for across evaluation.")
                continue

            train_ai_size = len(human_train_idx)
            sampled_other_ai_train = list(rs.choice(other_ai_train_idx_pool, size=min(train_ai_size, len(other_ai_train_idx_pool)), replace=False))
            train_idx = human_train_idx + sampled_other_ai_train

            test_ai_size = len(human_test_idx)
            sampled_target_ai_test = list(rs.choice(target_ai_test_idx_pool, size=min(test_ai_size, len(target_ai_test_idx_pool)), replace=False))
            test_idx = human_test_idx + sampled_target_ai_test

            X_train, y_train = scores[train_idx], labels[train_idx]
            X_test, y_test = scores[test_idx], labels[test_idx]

            if len(set(y_train)) < 2:
                print(f"    - Skip model {m}: not enough classes in sampled training set.")
                continue

            thr, _ = search_best_threshold(y_train, X_train, step=step)
            y_pred = (X_test < thr).astype(int)
            across_f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            print(f"    [Across] Model: {m}")
            print(classification_report(y_test, y_pred, digits=4, zero_division=0))
            model_across_preds[m]["y_true"].extend(list(y_test))
            model_across_preds[m]["y_pred"].extend(list(y_pred))

        if across_f1_list:
            overall_across_model_f1s.extend(across_f1_list)
            lang_across_avg = np.mean(across_f1_list)
            print(f"  > Across (avg macro F1 over models): {lang_across_avg:.4f}")
            overall_across_lang_avgs.append(lang_across_avg)

    if overall_within_lang_avgs or overall_across_lang_avgs:
        print("\n" + "="*24 + " Overall Results (Averaged over Languages) " + "="*24)
        if overall_within_lang_avgs:
            print(f"  > Overall Within (avg of per-language means): {np.mean(overall_within_lang_avgs):.4f}")
        else:
            print("  > Overall Within: No data.")
        if overall_across_lang_avgs:
            print(f"  > Overall Across (avg of per-language means): {np.mean(overall_across_lang_avgs):.4f}")
        else:
            print("  > Overall Across: No data.")

    if overall_within_model_f1s or overall_across_model_f1s:
        print("\n" + "="*24 + " Overall Results (Averaged over All Models) " + "="*24)
        if overall_within_model_f1s:
            print(f"  > Overall Within (avg over all models): {np.mean(overall_within_model_f1s):.4f}")
        else:
            print("  > Overall Within: No data.")
        if overall_across_model_f1s:
            print(f"  > Overall Across (avg over all models): {np.mean(overall_across_model_f1s):.4f}")
        else:
            print("  > Overall Across: No data.")

    print("\n" + "="*26 + " Per-Model Results " + "="*26)
    for m in models:
        y_true_w = np.array(model_within_preds[m]["y_true"]) if model_within_preds[m]["y_true"] else None
        y_pred_w = np.array(model_within_preds[m]["y_pred"]) if model_within_preds[m]["y_pred"] else None
        if y_true_w is not None and len(y_true_w) > 0:
            print(f"\n--- Model: {m} | Within (aggregated over languages) ---")
            print(classification_report(y_true_w, y_pred_w, digits=4, zero_division=0))
        else:
            print(f"\n--- Model: {m} | Within: No data.")

        y_true_a = np.array(model_across_preds[m]["y_true"]) if model_across_preds[m]["y_true"] else None
        y_pred_a = np.array(model_across_preds[m]["y_pred"]) if model_across_preds[m]["y_pred"] else None
        if y_true_a is not None and len(y_true_a) > 0:
            print(f"\n--- Model: {m} | Across (aggregated over languages) ---")
            print(classification_report(y_true_a, y_pred_a, digits=4, zero_division=0))
        else:
            print(f"\n--- Model: {m} | Across: No data.")


def evaluate_within_across_classifier(
    solutions: List[Solution],
    X: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float = 0.9,
    random_state: int = 42,
    rf_params: Dict = None,
):
    from sklearn.ensemble import RandomForestClassifier

    languages = sorted(list(set(s.language for s in solutions)))
    models = sorted(list(set(s.model for s in solutions if s.model != "Human")))
    rs = np.random.RandomState(random_state)

    model_within_preds = {m: {"y_true": [], "y_pred": []} for m in models}
    model_across_preds = {m: {"y_true": [], "y_pred": []} for m in models}
    clf_params = rf_params or {}

    print("\n" + "="*24 + " Per-Language Results (Task-based Split) " + "="*24)
    overall_within_lang_avgs = []
    overall_across_lang_avgs = []
    for lang in languages:
        lang_indices = [i for i, s in enumerate(solutions) if s.language == lang]
        if len(lang_indices) < 2: continue
        
        unique_tasks_in_lang = sorted(list(set(solutions[i].task_name for i in lang_indices)))
        if len(unique_tasks_in_lang) < 2:
            print(f"\n--- Language: {lang} --- \n  > Skip: Not enough unique tasks.")
            continue

        train_tasks, test_tasks = train_test_split(
            unique_tasks_in_lang, test_size=test_size, random_state=random_state
        )

        task_to_indices = defaultdict(list)
        for i in lang_indices:
            task_to_indices[solutions[i].task_name].append(i)

        train_indices_pool = [i for task in train_tasks for i in task_to_indices[task]]
        test_indices_pool = [i for task in test_tasks for i in task_to_indices[task]]

        lang_models = sorted(list(set(solutions[i].model for i in lang_indices if solutions[i].model != "Human")))
        if not lang_models: continue
        
        print(f"\n--- Language: {lang} ---")

        within_f1_list = []
        print("  > Within (per model)")
        for m in lang_models:
            train_idx = [i for i in train_indices_pool if solutions[i].model == m or solutions[i].label == 0]
            test_idx = [i for i in test_indices_pool if solutions[i].model == m or solutions[i].label == 0]

            if len(train_idx) < 2 or len(test_idx) < 2 or len(set(labels[train_idx])) < 2:
                print(f"    - Skip model {m}: not enough samples/classes in train/test tasks.")
                continue

            X_train, y_train = X[train_idx], labels[train_idx]
            X_test, y_test = X[test_idx], labels[test_idx]
            clf = RandomForestClassifier(**clf_params).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            within_f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            print(classification_report(y_test, y_pred, digits=4, zero_division=0))

            model_within_preds[m]["y_true"].extend(list(y_test))
            model_within_preds[m]["y_pred"].extend(list(y_pred))

        if within_f1_list:
            lang_within_avg = np.mean(within_f1_list)
            print(f"  > Within (avg macro F1 over models): {lang_within_avg:.4f}")
            overall_within_lang_avgs.append(lang_within_avg)

        across_f1_list = []
        print("  > Across (per model)")
        for m in lang_models:
            human_train_idx = [i for i in train_indices_pool if solutions[i].label == 0]
            other_ai_train_idx_pool = [i for i in train_indices_pool if solutions[i].label == 1 and solutions[i].model != m]
            
            human_test_idx = [i for i in test_indices_pool if solutions[i].label == 0]
            target_ai_test_idx_pool = [i for i in test_indices_pool if solutions[i].model == m]
            
            if not all([human_train_idx, other_ai_train_idx_pool, human_test_idx, target_ai_test_idx_pool]):
                print(f"    - Skip model {m}: insufficient data in task pools for across evaluation.")
                continue

            train_ai_size = len(human_train_idx)
            sampled_other_ai_train = list(rs.choice(other_ai_train_idx_pool, size=min(train_ai_size, len(other_ai_train_idx_pool)), replace=False))
            train_idx = human_train_idx + sampled_other_ai_train

            test_ai_size = len(human_test_idx)
            sampled_target_ai_test = list(rs.choice(target_ai_test_idx_pool, size=min(test_ai_size, len(target_ai_test_idx_pool)), replace=False))
            test_idx = human_test_idx + sampled_target_ai_test

            X_train, y_train = X[train_idx], labels[train_idx]
            X_test, y_test = X[test_idx], labels[test_idx]

            if len(set(y_train)) < 2:
                print(f"    - Skip model {m}: not enough classes in sampled training set.")
                continue

            clf = RandomForestClassifier(**clf_params).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            across_f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            print(classification_report(y_test, y_pred, digits=4, zero_division=0))

            model_across_preds[m]["y_true"].extend(list(y_test))
            model_across_preds[m]["y_pred"].extend(list(y_pred))

        if across_f1_list:
            lang_across_avg = np.mean(across_f1_list)
            print(f"  > Across (avg macro F1 over models): {lang_across_avg:.4f}")
            overall_across_lang_avgs.append(lang_across_avg)

    if overall_within_lang_avgs or overall_across_lang_avgs:
        print("\n" + "="*24 + " Overall Results (Averaged over Languages) " + "="*24)
        if overall_within_lang_avgs:
            print(f"  > Overall Within (avg of per-language means): {np.mean(overall_within_lang_avgs):.4f}")
        else:
            print("  > Overall Within: No data.")
        if overall_across_lang_avgs:
            print(f"  > Overall Across (avg of per-language means): {np.mean(overall_across_lang_avgs):.4f}")
        else:
            print("  > Overall Across: No data.")

    print("\n" + "="*26 + " Per-Model Results " + "="*26)
    for m in models:
        y_true_w = np.array(model_within_preds[m]["y_true"]) if model_within_preds[m]["y_true"] else None
        y_pred_w = np.array(model_within_preds[m]["y_pred"]) if model_within_preds[m]["y_pred"] else None
        if y_true_w is not None and len(y_true_w) > 0:
            print(f"\n--- Model: {m} | Within (aggregated over languages) ---")
            print(classification_report(y_true_w, y_pred_w, digits=4, zero_division=0))
        else:
            print(f"\n--- Model: {m} | Within: No data.")

        y_true_a = np.array(model_across_preds[m]["y_true"]) if model_across_preds[m]["y_true"] else None
        y_pred_a = np.array(model_across_preds[m]["y_pred"]) if model_across_preds[m]["y_pred"] else None
        if y_true_a is not None and len(y_true_a) > 0:
            print(f"\n--- Model: {m} | Across (aggregated over languages) ---")
            print(classification_report(y_true_a, y_pred_a, digits=4, zero_division=0))
        else:
            print(f"\n--- Model: {m} | Across: No data.")
