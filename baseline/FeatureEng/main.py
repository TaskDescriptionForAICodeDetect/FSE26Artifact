import sys, os, json, pickle
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from baseutils.ioutils import get_all_solutions
from baseutils.stasticutils import evaluate_within_across_classifier
from FeatureEng.codeUtil import extract_features
def main(**config) -> Dict:
    print("\n" + "="*50)
    print("--- Starting Feature Engineering Experiment ---")
    for key, val in config.items():
        print(f"  - {key}: {val}")
    print("="*50 + "\n")
    print("Loading all solutions...")
    all_solutions = get_all_solutions(config['root_path'])
    supported_languages = {'python', 'java', 'c', 'c++'}
    solutions = [s for s in all_solutions if s.language in supported_languages]
    print(f"Loaded {len(all_solutions)} total solutions, filtered down to {len(solutions)} supported solutions (Python, Java, C, C++).")
    if not solutions:
        print("No solutions found for the supported languages. Exiting.")
        return { "config": config, "classification_report": None, "confusion_matrix": None }
    print("Computing features for all solutions...")
    feature_list = []
    for s in tqdm(solutions, desc="Extracting features for solutions"):
        feats = extract_features(s.code, s.language, s.task_name)
        feature_list.append(feats)
    features = np.vstack(feature_list)
    labels = np.array([s.label for s in solutions])
    evaluate_within_across_classifier(
        solutions=solutions,
        X=features,
        labels=labels,
        test_size=config.get('test_size', 0.3),
        random_state=config.get('random_state', 42),
        rf_params=config.get('rf_params', {}),
    )
    return {
        "config": config,
    }
def run_all_experiments():
    base_path = "../splitDescription/split_benchmark/"
    base_config = {
        "root_path": base_path,
        "test_size": 0.3,
        "random_state": 42,
        "rf_params": dict(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42),
    }
    experiment_configs = [
        base_config
    ]
    all_results = []
    for exp_config in experiment_configs:
        result = main(**exp_config)
        all_results.append(result)
    summary_path = "feature_eng_experiment_summary.json"
    print(f"\n{'='*30}\nExperiment summary saved to -> {summary_path}\n{'='*30}")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    run_all_experiments()
