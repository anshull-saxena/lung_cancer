"""
Tables 8 & 9: Combined GA + Grouping comparisons.

Table 8: Adaptive GA + Grouping vs Baseline GA (no adaptation, no grouping)
Table 9: NSGA-II + Grouping vs Baseline NSGA-II (no grouping)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    print("  Tables 8 & 9: GA+Grouping & NSGA-II+Grouping Comparisons")
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
    F_train_val = extract_features_cached(model, X_train_val, "t89_trainval_dense_se")
    F_test = extract_features_cached(model, X_test, "t89_test_dense_se")
    F_train = F_train_val[:len(X_train)]
    n_feat = F_train_val.shape[1]

    # Feature groups
    quarter = n_feat // 4
    groups = [
        (0, quarter),
        (quarter, 2 * quarter),
        (2 * quarter, 3 * quarter),
        (3 * quarter, n_feat),
    ]

    def eval_selector(selector, label):
        sel_idx = None
        if hasattr(selector, 'run'):
            result = selector.run(F_train_val, y_train_val)
            if len(result) == 3:  # NSGA-II returns (indices, pareto, history)
                sel_idx = result[0]
            else:
                sel_idx = result[0]
        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(
            knn, F_train[:, sel_idx], y_train, F_test[:, sel_idx], y_test
        )
        metrics["method"] = label
        metrics["n_features"] = len(sel_idx)
        metrics["reduction"] = f"{len(sel_idx)/n_feat:.1%}"
        return metrics

    # ═══════════════════════════════════════════════════════════════════════════
    # Table 8: Adaptive GA + Grouping vs Baseline GA
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3] Table 8: Adaptive GA + Grouping vs Baseline GA ...")

    # Baseline GA (fixed rates, no grouping)
    print("  Running Baseline GA ...")
    baseline_ga = BaselineGA(n_features=n_feat)
    m_baseline = eval_selector(baseline_ga, "Baseline GA")

    # Adaptive GA (adaptive rates, no grouping)
    print("  Running Adaptive GA (no grouping) ...")
    adapt_ga = AdaptiveGA(n_features=n_feat)
    m_adapt = eval_selector(adapt_ga, "Adaptive GA")

    # Adaptive GA + Grouping
    print("  Running Adaptive GA + Grouping ...")
    adapt_ga_g = AdaptiveGA(n_features=n_feat, seed=42)
    adapt_ga_g.set_groups(groups)
    m_adapt_g = eval_selector(adapt_ga_g, "Adaptive GA + Grouping")

    table8_results = [m_baseline, m_adapt, m_adapt_g]
    print_table(table8_results, "Table 8: GA Variants Comparison")

    save_results_json(table8_results, os.path.join(RESULTS_DIR, "table_8.json"))
    csv8 = [{
        "Method": r["method"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "Specificity": r["specificity"],
        "Features": r["n_features"], "Reduction": r["reduction"],
    } for r in table8_results]
    save_results_csv(csv8, os.path.join(RESULTS_DIR, "table_8.csv"))
    latex8 = generate_latex_table(
        csv8,
        caption="Comparison of Baseline GA, Adaptive GA, and Adaptive GA with grouping operator",
        label="tab:table8",
        columns=[
            ("Method", "Method"), ("Accuracy", "Accuracy"),
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-Score"), ("Features", "\\# Features"),
            ("Reduction", "Reduction"),
        ]
    )
    save_latex_table(latex8, os.path.join(RESULTS_DIR, "table_8.tex"))

    # ═══════════════════════════════════════════════════════════════════════════
    # Table 9: NSGA-II + Grouping vs Baseline NSGA-II
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4] Table 9: NSGA-II + Grouping vs Baseline NSGA-II ...")

    # NSGA-II without grouping
    print("  Running NSGA-II (no grouping) ...")
    nsga_no = NSGA2Selector(n_features=n_feat)
    m_nsga_no = eval_selector(nsga_no, "NSGA-II")

    # NSGA-II with grouping
    print("  Running NSGA-II + Grouping ...")
    nsga_g = NSGA2Selector(n_features=n_feat, seed=42)
    nsga_g.set_groups(groups)
    m_nsga_g = eval_selector(nsga_g, "NSGA-II + Grouping")

    table9_results = [m_nsga_no, m_nsga_g]
    print_table(table9_results, "Table 9: NSGA-II Grouping Comparison")

    save_results_json(table9_results, os.path.join(RESULTS_DIR, "table_9.json"))
    csv9 = [{
        "Method": r["method"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "Specificity": r["specificity"],
        "Features": r["n_features"], "Reduction": r["reduction"],
    } for r in table9_results]
    save_results_csv(csv9, os.path.join(RESULTS_DIR, "table_9.csv"))
    latex9 = generate_latex_table(
        csv9,
        caption="Effect of grouping operator on NSGA-II feature selection",
        label="tab:table9",
        columns=[
            ("Method", "Method"), ("Accuracy", "Accuracy"),
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-Score"), ("Features", "\\# Features"),
            ("Reduction", "Reduction"),
        ]
    )
    save_latex_table(latex9, os.path.join(RESULTS_DIR, "table_9.tex"))

    print("\nTables 8 & 9 complete.")
    return table8_results, table9_results


if __name__ == "__main__":
    run()
