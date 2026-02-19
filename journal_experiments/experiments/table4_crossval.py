"""
Table 4: 10-fold cross-validation.

For each fold:
  - Extract features (DenseNet121 + SE)
  - Run Adaptive GA
  - Train KNN, evaluate
Report fold-wise metrics + mean ± std.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import set_seed, DATA_DIR, RESULTS_DIR, K_FOLDS, CLASS_NAMES
from data_loader import load_dataset, get_kfold_splits
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
    print("  Table 4: 10-Fold Cross-Validation")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1] Loading dataset ...")
    X, y = load_dataset(DATA_DIR)

    # 2. Build feature extractor
    print("\n[2] Building DenseNet121 + SE feature extractor ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")

    # 3. Extract ALL features once (cache)
    print("\n[3] Extracting features for full dataset ...")
    F_all = extract_features_cached(model, X, "t4_full_dense_se")

    # 4. K-fold cross validation
    print(f"\n[4] Running {K_FOLDS}-fold cross-validation ...")
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(get_kfold_splits(X, y, K_FOLDS)):
        print(f"\n--- Fold {fold_i + 1}/{K_FOLDS} ---")
        F_train, y_train = F_all[train_idx], y[train_idx]
        F_test, y_test = F_all[test_idx], y[test_idx]

        # Run Adaptive GA
        ga = AdaptiveGA(n_features=F_train.shape[1], n_gen=30, seed=42 + fold_i)
        sel_idx, _ = ga.run(F_train, y_train)

        # Train KNN
        F_train_sel = F_train[:, sel_idx]
        F_test_sel = F_test[:, sel_idx]

        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(knn, F_train_sel, y_train, F_test_sel, y_test)
        metrics["fold"] = fold_i + 1
        metrics["n_features"] = len(sel_idx)
        fold_results.append(metrics)

        print(f"  Fold {fold_i+1}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, Features={len(sel_idx)}")

    # Compute summary (mean ± std)
    metric_keys = ["accuracy", "precision", "recall", "f1", "specificity"]
    summary = {"fold": "Mean ± Std"}
    for key in metric_keys:
        vals = [r[key] for r in fold_results]
        summary[key] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    summary["n_features"] = f"{np.mean([r['n_features'] for r in fold_results]):.0f}"

    # Print
    print_table(fold_results + [summary], "Table 4: 10-Fold Cross-Validation Results")

    # Prepare serializable results
    table4_data = {
        "folds": fold_results,
        "summary": {
            key: {"mean": float(np.mean([r[key] for r in fold_results])),
                  "std": float(np.std([r[key] for r in fold_results]))}
            for key in metric_keys
        },
        "avg_features": float(np.mean([r["n_features"] for r in fold_results])),
    }

    save_results_json(table4_data, os.path.join(RESULTS_DIR, "table_4.json"))

    # Build rows for CSV/LaTeX
    csv_rows = []
    for r in fold_results:
        csv_rows.append({
            "Fold": r["fold"], "Accuracy": r["accuracy"],
            "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "Specificity": r["specificity"],
            "Features": r["n_features"],
        })
    csv_rows.append({
        "Fold": "Mean",
        "Accuracy": np.mean([r["accuracy"] for r in fold_results]),
        "Precision": np.mean([r["precision"] for r in fold_results]),
        "Recall": np.mean([r["recall"] for r in fold_results]),
        "F1": np.mean([r["f1"] for r in fold_results]),
        "Specificity": np.mean([r["specificity"] for r in fold_results]),
        "Features": np.mean([r["n_features"] for r in fold_results]),
    })
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_4.csv"))

    latex4 = generate_latex_table(
        csv_rows,
        caption="10-fold cross-validation results with DenseNet121 + SE + Adaptive GA + KNN",
        label="tab:table4",
        columns=[
            ("Fold", "Fold"), ("Accuracy", "Accuracy"),
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-Score"), ("Specificity", "Specificity"),
            ("Features", "\\# Features"),
        ]
    )
    save_latex_table(latex4, os.path.join(RESULTS_DIR, "table_4.tex"))

    print("\nTable 4 complete.")
    return table4_data


if __name__ == "__main__":
    run()
