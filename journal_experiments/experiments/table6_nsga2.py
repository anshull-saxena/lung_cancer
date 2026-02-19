"""
Table 6: NSGA-II integration.

1. Extract features (DenseNet121 + SE)
2. Run NSGA-II feature selection
3. Report: Pareto front, best solution metrics, feature count
4. Compare with Adaptive GA results
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES
from data_loader import load_dataset, get_splits
from models.backbone import build_feature_extractor, extract_features_cached
from feature_selection.adaptive_ga import AdaptiveGA
from feature_selection.nsga2 import NSGA2Selector
from classifiers import get_classifiers, train_and_evaluate
from sklearn.neighbors import KNeighborsClassifier
from config import KNN_K, KNN_WEIGHTS
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


def run():
    set_seed()
    print("=" * 70)
    print("  Table 6: NSGA-II Integration")
    print("=" * 70)

    # 1. Load and split
    print("\n[1] Loading dataset ...")
    X, y = load_dataset(DATA_DIR)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    # 2. Extract features
    print("\n[2] Building feature extractor and extracting features ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")
    F_train_val = extract_features_cached(model, X_train_val, "t6_trainval_dense_se")
    F_test = extract_features_cached(model, X_test, "t6_test_dense_se")
    F_train = F_train_val[:len(X_train)]

    # 3. Run Adaptive GA (for comparison)
    print("\n[3] Running Adaptive GA ...")
    ga = AdaptiveGA(n_features=F_train_val.shape[1])
    ga_sel_idx, ga_history = ga.run(F_train_val, y_train_val)

    # Evaluate GA result
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
    ga_metrics = train_and_evaluate(
        knn, F_train[:, ga_sel_idx], y_train, F_test[:, ga_sel_idx], y_test
    )
    ga_metrics["method"] = "Adaptive GA"
    ga_metrics["n_features"] = len(ga_sel_idx)

    # 4. Run NSGA-II
    print("\n[4] Running NSGA-II ...")
    nsga = NSGA2Selector(n_features=F_train_val.shape[1])
    nsga_sel_idx, pareto_front, nsga_history = nsga.run(F_train_val, y_train_val)

    # Evaluate NSGA-II result
    knn2 = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
    nsga_metrics = train_and_evaluate(
        knn2, F_train[:, nsga_sel_idx], y_train, F_test[:, nsga_sel_idx], y_test
    )
    nsga_metrics["method"] = "NSGA-II"
    nsga_metrics["n_features"] = len(nsga_sel_idx)

    # 5. Comparison table
    table6_results = [ga_metrics, nsga_metrics]
    print_table(table6_results, "Table 6: Adaptive GA vs NSGA-II")

    # 6. Pareto front data
    pareto_data = {
        "pareto_front": [{"accuracy": a, "n_features": int(n)}
                         for a, n in pareto_front],
        "comparison": table6_results,
        "ga_history": ga_history,
        "nsga_history": nsga_history,
    }

    save_results_json(pareto_data, os.path.join(RESULTS_DIR, "table_6.json"))

    csv_rows = [{
        "Method": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Features": r["n_features"],
    } for r in table6_results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_6.csv"))

    latex6 = generate_latex_table(
        csv_rows,
        caption="Comparison of Adaptive GA and NSGA-II feature selection on DenseNet121 + SE features",
        label="tab:table6",
        columns=[
            ("Method", "Method"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Specificity", "Specificity"),
            ("Features", "\\# Features"),
        ]
    )
    save_latex_table(latex6, os.path.join(RESULTS_DIR, "table_6.tex"))

    print("\nTable 6 complete.")
    return pareto_data


if __name__ == "__main__":
    run()
