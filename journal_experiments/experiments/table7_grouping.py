"""
Table 7: Grouping operator effect.

Compare Adaptive GA without vs with the grouping operator.
Since our base uses single DenseNet121 (1024 features), we split
features into synthetic groups (quartiles) to demonstrate the operator.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import set_seed, DATA_DIR, RESULTS_DIR
from data_loader import load_dataset, get_splits
from models.backbone import build_feature_extractor, extract_features_cached
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import train_and_evaluate
from sklearn.neighbors import KNeighborsClassifier
from config import KNN_K, KNN_WEIGHTS
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


def run():
    set_seed()
    print("=" * 70)
    print("  Table 7: Grouping Operator Effect")
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
    F_train_val = extract_features_cached(model, X_train_val, "t7_trainval_dense_se")
    F_test = extract_features_cached(model, X_test, "t7_test_dense_se")
    F_train = F_train_val[:len(X_train)]
    n_feat = F_train_val.shape[1]

    # Define feature groups (quartiles of the feature dimension)
    quarter = n_feat // 4
    groups = [
        (0, quarter),
        (quarter, 2 * quarter),
        (2 * quarter, 3 * quarter),
        (3 * quarter, n_feat),
    ]
    print(f"  Feature groups: {groups}")

    # 3. Adaptive GA WITHOUT grouping
    print("\n[3] Running Adaptive GA WITHOUT grouping ...")
    ga_no_group = AdaptiveGA(n_features=n_feat)
    sel_no_group, hist_no = ga_no_group.run(F_train_val, y_train_val)

    knn1 = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
    metrics_no = train_and_evaluate(
        knn1, F_train[:, sel_no_group], y_train, F_test[:, sel_no_group], y_test
    )
    metrics_no["method"] = "Adaptive GA (no grouping)"
    metrics_no["n_features"] = len(sel_no_group)

    # 4. Adaptive GA WITH grouping
    print("\n[4] Running Adaptive GA WITH grouping ...")
    ga_with_group = AdaptiveGA(n_features=n_feat, seed=42)
    ga_with_group.set_groups(groups)
    sel_with_group, hist_with = ga_with_group.run(F_train_val, y_train_val)

    knn2 = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
    metrics_with = train_and_evaluate(
        knn2, F_train[:, sel_with_group], y_train, F_test[:, sel_with_group], y_test
    )
    metrics_with["method"] = "Adaptive GA (with grouping)"
    metrics_with["n_features"] = len(sel_with_group)

    # 5. Comparison
    table7_results = [metrics_no, metrics_with]
    print_table(table7_results, "Table 7: Grouping Operator Effect")

    # Feature reduction comparison
    print(f"  Without grouping: {len(sel_no_group)}/{n_feat} "
          f"({len(sel_no_group)/n_feat:.1%})")
    print(f"  With grouping:    {len(sel_with_group)}/{n_feat} "
          f"({len(sel_with_group)/n_feat:.1%})")

    save_data = {
        "results": table7_results,
        "groups": groups,
        "no_grouping_history": hist_no,
        "with_grouping_history": hist_with,
    }
    save_results_json(save_data, os.path.join(RESULTS_DIR, "table_7.json"))

    csv_rows = [{
        "Method": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Features": r["n_features"],
        "Reduction": f"{r['n_features']/n_feat:.1%}",
    } for r in table7_results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_7.csv"))

    latex7 = generate_latex_table(
        csv_rows,
        caption="Effect of grouping operator on Adaptive GA feature selection",
        label="tab:table7",
        columns=[
            ("Method", "Method"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Features", "\\# Features"),
            ("Reduction", "Reduction \\%"),
        ]
    )
    save_latex_table(latex7, os.path.join(RESULTS_DIR, "table_7.tex"))

    print("\nTable 7 complete.")
    return save_data


if __name__ == "__main__":
    run()
