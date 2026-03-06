"""
Table 15: Ensemble fusion strategy comparison.

Trains all 5 classifiers (KNN, SVM, RF, LR, XGBoost) on DenseNet121+SE features
after Adaptive GA selection, then compares 6 ensemble strategies:
  1. Majority Vote
  2. Weighted Probability Fusion
  3. GP1 (Max)
  4. GP2 (Algebraic Product)
  5. GP3 (Ratio-Based)
  6. GP4 (Weighted Sum)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES
from data_loader import load_dataset, get_splits
from models.backbone import build_feature_extractor, extract_features_cached
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import get_all_classifiers, train_and_evaluate
from ensemble_fusion import evaluate_all_ensemble_methods
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


def run():
    set_seed()
    print("=" * 70)
    print("  Table 15: Ensemble Fusion Strategy Comparison")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1] Loading dataset ...")
    X, y = load_dataset(DATA_DIR)

    # 2. Split 70/10/20
    print("\n[2] Splitting data ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 3. Build feature extractor
    print("\n[3] Building DenseNet121 + SE feature extractor ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")

    # 4. Extract features
    print("\n[4] Extracting features ...")
    F_train_val = extract_features_cached(model, X_train_val, "t15_trainval_dense_se")
    F_test = extract_features_cached(model, X_test, "t15_test_dense_se")
    F_train = F_train_val[:len(X_train)]
    F_val = F_train_val[len(X_train):]

    # 5. Run Adaptive GA on train+val features
    print("\n[5] Running Adaptive GA ...")
    ga = AdaptiveGA(n_features=F_train_val.shape[1])
    sel_idx, ga_history = ga.run(F_train_val, y_train_val)
    print(f"  Selected {len(sel_idx)} / {F_train_val.shape[1]} features")

    # 6. Apply feature selection
    F_train_sel = F_train[:, sel_idx]
    F_val_sel = F_val[:, sel_idx]
    F_test_sel = F_test[:, sel_idx]

    # 7. Train all 5 classifiers and collect probabilities
    print("\n[6] Training all 5 classifiers ...")
    classifiers = get_all_classifiers()
    clf_names = list(classifiers.keys())

    individual_results = []
    probas_dict = {}
    val_acc_dict = {}

    for name, clf in classifiers.items():
        print(f"  Training {name} ...")
        # Train on train set
        clf.fit(F_train_sel, y_train)

        # Validation accuracy (for epsilon weight computation)
        val_pred = clf.predict(F_val_sel)
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(y_val, val_pred)
        val_acc_dict[name] = val_acc
        print(f"    Val accuracy: {val_acc:.4f}")

        # Test predictions and probabilities
        test_metrics = train_and_evaluate(clf, F_train_sel, y_train,
                                          F_test_sel, y_test)
        test_metrics["method"] = name
        test_metrics["n_features"] = len(sel_idx)
        individual_results.append(test_metrics)

        # Get probability estimates for ensemble
        probas = clf.predict_proba(F_test_sel)
        probas_dict[name] = probas

    # Print individual classifier results
    print_table(individual_results,
                "Table 15a: Individual Classifier Performance (5 classifiers)")

    # 8. Run all ensemble methods
    print("\n[7] Evaluating ensemble fusion strategies ...")
    ensemble_results = evaluate_all_ensemble_methods(
        clf_names, probas_dict, val_acc_dict, y_test
    )
    for r in ensemble_results:
        r["n_features"] = len(sel_idx)

    print_table(ensemble_results,
                "Table 15b: Ensemble Fusion Strategy Comparison")

    # 9. Save everything
    all_results = {
        "individual_classifiers": individual_results,
        "ensemble_comparison": ensemble_results,
        "val_accuracies": val_acc_dict,
        "n_features_selected": len(sel_idx),
    }
    save_results_json(all_results, os.path.join(RESULTS_DIR, "table_15.json"))

    # CSV for ensemble comparison
    csv_rows = [{
        "Method": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
    } for r in ensemble_results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_15.csv"))

    # LaTeX
    latex15 = generate_latex_table(
        csv_rows,
        caption="Comparison of ensemble fusion strategies with 5 classifiers "
                "(DenseNet121 + SE + Adaptive GA)",
        label="tab:table15",
        columns=[
            ("Method", "Method"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Specificity", "Specificity"),
        ]
    )
    save_latex_table(latex15, os.path.join(RESULTS_DIR, "table_15.tex"))

    # Also save individual classifier CSV
    csv_ind = [{
        "Classifier": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Features": r["n_features"],
    } for r in individual_results]
    save_results_csv(csv_ind, os.path.join(RESULTS_DIR, "table_15_individual.csv"))

    latex15_ind = generate_latex_table(
        csv_ind,
        caption="Individual classifier performance with 5 classifiers "
                "(DenseNet121 + SE + Adaptive GA)",
        label="tab:table15_ind",
        columns=[
            ("Classifier", "Classifier"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Specificity", "Specificity"),
        ]
    )
    save_latex_table(latex15_ind, os.path.join(RESULTS_DIR, "table_15_individual.tex"))

    print("\nTable 15 complete.")
    return all_results


if __name__ == "__main__":
    run()
