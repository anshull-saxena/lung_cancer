"""
Table 16: Full proposed pipeline.

Multi-backbone feature extraction (DenseNet121 + ResNet50 + VGG16 + EfficientNetB0)
→ SE attention on each → feature concatenation (~4864 features)
→ NSGA-II multi-objective feature selection (with grouping operator)
→ 5 classifiers (KNN, SVM, RF, LR, XGBoost)
→ GP ensemble fusion (GP1-GP4 + Weighted Fusion + Majority Vote)

This is the complete proposed method from the paper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import tensorflow as tf
from config import set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES
from data_loader import load_dataset, get_splits
from models.backbone import (build_feature_extractor, extract_features_cached,
                              BACKBONE_REGISTRY)
from feature_selection.nsga2 import NSGA2Selector
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import get_all_classifiers, train_and_evaluate
from ensemble_fusion import evaluate_all_ensemble_methods
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)

BACKBONES = ["densenet121", "resnet50", "vgg16", "efficientnetb0"]
ATTENTION = "se"


def run():
    set_seed()
    print("=" * 70)
    print("  Table 16: Full Proposed Pipeline")
    print("  Multi-Backbone + SE + NSGA-II + 5 Classifiers + GP Ensemble")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1] Loading dataset ...")
    X, y = load_dataset(DATA_DIR)

    # 2. Split
    print("\n[2] Splitting data ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 3. Extract features from all backbones
    print("\n[3] Extracting features from all backbones ...")
    backbone_features_trainval = []
    backbone_features_test = []
    backbone_features_train = []
    backbone_features_val = []
    group_boundaries = []
    offset = 0

    for bb_name in BACKBONES:
        print(f"\n  Backbone: {bb_name} + {ATTENTION} ...")
        set_seed()
        tf.keras.backend.clear_session()

        model, feat_dim = build_feature_extractor(bb_name, ATTENTION)
        print(f"    Feature dim: {feat_dim}")

        F_trainval = extract_features_cached(
            model, X_train_val, f"t16_{bb_name}_{ATTENTION}_trainval"
        )
        F_test = extract_features_cached(
            model, X_test, f"t16_{bb_name}_{ATTENTION}_test"
        )
        F_train = F_trainval[:len(X_train)]
        F_val = F_trainval[len(X_train):]

        backbone_features_trainval.append(F_trainval)
        backbone_features_test.append(F_test)
        backbone_features_train.append(F_train)
        backbone_features_val.append(F_val)

        group_boundaries.append((offset, offset + feat_dim))
        offset += feat_dim

        del model

    # Concatenate all backbone features
    F_concat_trainval = np.concatenate(backbone_features_trainval, axis=1)
    F_concat_test = np.concatenate(backbone_features_test, axis=1)
    F_concat_train = np.concatenate(backbone_features_train, axis=1)
    F_concat_val = np.concatenate(backbone_features_val, axis=1)
    total_features = F_concat_trainval.shape[1]
    print(f"\n  Concatenated features: {total_features}")
    print(f"  Group boundaries: {group_boundaries}")

    # 4. NSGA-II feature selection with grouping
    print("\n[4] Running NSGA-II with grouping operator ...")
    t0 = time.time()
    nsga = NSGA2Selector(n_features=total_features)
    nsga.set_groups(group_boundaries)
    sel_idx, pareto_front, nsga_history = nsga.run(F_concat_trainval, y_train_val)
    nsga_time = time.time() - t0
    print(f"  Selected {len(sel_idx)} / {total_features} features "
          f"({len(sel_idx)/total_features:.1%}) in {nsga_time:.1f}s")

    # 5. Apply feature selection
    F_train_sel = F_concat_train[:, sel_idx]
    F_val_sel = F_concat_val[:, sel_idx]
    F_test_sel = F_concat_test[:, sel_idx]

    # 6. Train all 5 classifiers
    print("\n[5] Training all 5 classifiers ...")
    classifiers = get_all_classifiers()
    clf_names = list(classifiers.keys())

    individual_results = []
    probas_dict = {}
    val_acc_dict = {}

    for name, clf in classifiers.items():
        print(f"  Training {name} ...")
        clf.fit(F_train_sel, y_train)

        # Validation accuracy
        val_pred = clf.predict(F_val_sel)
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(y_val, val_pred)
        val_acc_dict[name] = val_acc
        print(f"    Val accuracy: {val_acc:.4f}")

        # Test evaluation
        test_metrics = train_and_evaluate(clf, F_train_sel, y_train,
                                          F_test_sel, y_test)
        test_metrics["method"] = name
        test_metrics["n_features"] = len(sel_idx)
        individual_results.append(test_metrics)

        # Probabilities for ensemble
        probas = clf.predict_proba(F_test_sel)
        probas_dict[name] = probas

    print_table(individual_results,
                "Table 16a: Individual Classifier Performance (Full Pipeline)")

    # 7. All ensemble methods
    print("\n[6] Evaluating ensemble fusion strategies ...")
    ensemble_results = evaluate_all_ensemble_methods(
        clf_names, probas_dict, val_acc_dict, y_test
    )
    for r in ensemble_results:
        r["n_features"] = len(sel_idx)

    print_table(ensemble_results,
                "Table 16b: Ensemble Comparison (Full Pipeline)")

    # 8. Also run Adaptive GA for comparison
    print("\n[7] Running Adaptive GA with grouping (for comparison) ...")
    t0 = time.time()
    aga = AdaptiveGA(n_features=total_features)
    aga.set_groups(group_boundaries)
    aga_sel_idx, aga_history = aga.run(F_concat_trainval, y_train_val)
    aga_time = time.time() - t0

    from sklearn.neighbors import KNeighborsClassifier
    from config import KNN_K, KNN_WEIGHTS
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
    aga_metrics = train_and_evaluate(
        knn, F_concat_train[:, aga_sel_idx], y_train,
        F_concat_test[:, aga_sel_idx], y_test
    )
    aga_metrics["method"] = "Adaptive GA + KNN"
    aga_metrics["n_features"] = len(aga_sel_idx)
    aga_metrics["time_seconds"] = round(aga_time, 1)

    # Best ensemble result
    best_ensemble = max(ensemble_results, key=lambda r: r["accuracy"])
    best_ensemble_row = {
        "method": f"NSGA-II + {best_ensemble['method']}",
        "accuracy": best_ensemble["accuracy"],
        "precision": best_ensemble["precision"],
        "recall": best_ensemble["recall"],
        "f1": best_ensemble["f1"],
        "specificity": best_ensemble["specificity"],
        "n_features": len(sel_idx),
        "time_seconds": round(nsga_time, 1),
    }

    fs_comparison = [aga_metrics, best_ensemble_row]
    print_table(fs_comparison,
                "Table 16c: Feature Selection Method Comparison (Full Pipeline)")

    # 9. Save all results
    all_results = {
        "pipeline": {
            "backbones": BACKBONES,
            "attention": ATTENTION,
            "total_features": total_features,
            "group_boundaries": group_boundaries,
            "nsga_selected": len(sel_idx),
            "nsga_time_seconds": round(nsga_time, 1),
        },
        "individual_classifiers": individual_results,
        "ensemble_comparison": ensemble_results,
        "val_accuracies": val_acc_dict,
        "pareto_front": [{"accuracy": a, "n_features": int(n)}
                         for a, n in pareto_front],
        "nsga_history": nsga_history,
    }
    save_results_json(all_results, os.path.join(RESULTS_DIR, "table_16.json"))

    # CSV — individual classifiers
    csv_ind = [{
        "Classifier": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Features": r["n_features"],
    } for r in individual_results]
    save_results_csv(csv_ind, os.path.join(RESULTS_DIR, "table_16_individual.csv"))

    # CSV — ensemble comparison
    csv_ens = [{
        "Method": r["method"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
    } for r in ensemble_results]
    save_results_csv(csv_ens, os.path.join(RESULTS_DIR, "table_16_ensemble.csv"))

    # LaTeX — individual classifiers
    latex_ind = generate_latex_table(
        csv_ind,
        caption="Individual classifier performance with full multi-backbone pipeline "
                "(4 backbones + SE + NSGA-II feature selection)",
        label="tab:table16_ind",
        columns=[
            ("Classifier", "Classifier"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Specificity", "Specificity"),
            ("Features", "\\# Features"),
        ]
    )
    save_latex_table(latex_ind, os.path.join(RESULTS_DIR, "table_16_individual.tex"))

    # LaTeX — ensemble comparison
    latex_ens = generate_latex_table(
        csv_ens,
        caption="Ensemble fusion strategy comparison with full multi-backbone pipeline",
        label="tab:table16_ens",
        columns=[
            ("Method", "Method"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Specificity", "Specificity"),
        ]
    )
    save_latex_table(latex_ens, os.path.join(RESULTS_DIR, "table_16_ensemble.tex"))

    print("\nTable 16 complete.")
    return all_results


if __name__ == "__main__":
    run()
