"""
Table 10: Comprehensive NSGA-II evaluation.

Compare: Baseline GA vs Adaptive GA vs NSGA-II
Metrics: Accuracy, F1, Confusion Matrix, feature count, time complexity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from config import set_seed, DATA_DIR, RESULTS_DIR
from data_loader import load_dataset, get_splits
from models.backbone import build_feature_extractor, extract_features_cached
from feature_selection.adaptive_ga import AdaptiveGA
from feature_selection.baseline_ga import BaselineGA
from feature_selection.nsga2 import NSGA2Selector
from classifiers import train_and_evaluate
from sklearn.neighbors import KNeighborsClassifier
from config import KNN_K, KNN_WEIGHTS
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


def run():
    set_seed()
    print("=" * 70)
    print("  Table 10: Comprehensive NSGA-II Evaluation")
    print("=" * 70)

    # 1. Load and split
    print("\n[1] Loading dataset ...")
    X, y = load_dataset(DATA_DIR)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    # 2. Extract features
    print("\n[2] Extracting features ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")
    F_train_val = extract_features_cached(model, X_train_val, "t10_trainval_dense_se")
    F_test = extract_features_cached(model, X_test, "t10_test_dense_se")
    F_train = F_train_val[:len(X_train)]
    n_feat = F_train_val.shape[1]

    results = []
    methods = {
        "Baseline GA": BaselineGA(n_features=n_feat),
        "Adaptive GA": AdaptiveGA(n_features=n_feat),
        "NSGA-II": NSGA2Selector(n_features=n_feat),
    }

    for name, selector in methods.items():
        print(f"\n[3] Running {name} ...")
        t0 = time.time()
        result = selector.run(F_train_val, y_train_val)
        elapsed = time.time() - t0

        if len(result) == 3:  # NSGA-II
            sel_idx = result[0]
        else:
            sel_idx = result[0]

        # Evaluate
        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(
            knn, F_train[:, sel_idx], y_train, F_test[:, sel_idx], y_test
        )
        metrics["method"] = name
        metrics["n_features"] = len(sel_idx)
        metrics["reduction_pct"] = f"{(1 - len(sel_idx)/n_feat)*100:.1f}%"
        metrics["time_seconds"] = round(elapsed, 1)
        results.append(metrics)

        print(f"  {name}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Features={len(sel_idx)}, Time={elapsed:.1f}s")

    # Print
    print_table(results, "Table 10: Comprehensive Feature Selection Comparison")

    # Save
    save_results_json(results, os.path.join(RESULTS_DIR, "table_10.json"))

    csv_rows = [{
        "Method": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Features": r["n_features"],
        "Reduction": r["reduction_pct"],
        "Time (s)": r["time_seconds"],
    } for r in results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_10.csv"))

    latex10 = generate_latex_table(
        csv_rows,
        caption="Comprehensive comparison of feature selection methods: Baseline GA, Adaptive GA, and NSGA-II",
        label="tab:table10",
        columns=[
            ("Method", "Method"),
            ("Accuracy", "Accuracy"),
            ("F1", "F1-Score"),
            ("Features", "\\# Features"),
            ("Reduction", "Reduction \\%"),
            ("Time (s)", "Time (s)"),
        ]
    )
    save_latex_table(latex10, os.path.join(RESULTS_DIR, "table_10.tex"))

    # Also save confusion matrices
    cm_data = {r["method"]: r["confusion_matrix"] for r in results}
    save_results_json(cm_data, os.path.join(RESULTS_DIR, "table_10_cm.json"))

    print("\nTable 10 complete.")
    return results


if __name__ == "__main__":
    run()
