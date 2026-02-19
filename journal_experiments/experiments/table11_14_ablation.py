"""
Tables 11-14: Ablation studies.

Table 11: Effect of GA population size (20, 40, 60, 80)
Table 12: Effect of number of generations (10, 25, 50, 100)
Table 13: Effect of KNN k value (3, 5, 7, 9)
Table 14: Effect of DenseNet121 feature layer + comparison with other backbones
          (ResNet50, VGG16, EfficientNetB0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from config import set_seed, DATA_DIR, RESULTS_DIR, N_GEN, POP_SIZE
from data_loader import load_dataset, get_splits
from models.backbone import build_feature_extractor, extract_features_cached
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import train_and_evaluate
from sklearn.neighbors import KNeighborsClassifier
from config import KNN_K, KNN_WEIGHTS
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


def _load_and_prepare():
    """Shared data loading for all ablation tables."""
    X, y = load_dataset(DATA_DIR)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    return X, y, X_train, y_train, X_train_val, y_train_val, X_test, y_test


def run_table11(F_train_val, y_train_val, F_train, y_train, F_test, y_test):
    """Table 11: Effect of GA population size."""
    print("\n" + "=" * 70)
    print("  Table 11: Effect of GA Population Size")
    print("=" * 70)

    pop_sizes = [20, 40, 60, 80]
    results = []
    n_feat = F_train_val.shape[1]

    for pop in pop_sizes:
        print(f"\n  Pop size = {pop} ...")
        set_seed()
        ga = AdaptiveGA(n_features=n_feat, pop_size=pop, n_gen=N_GEN)
        sel_idx, _ = ga.run(F_train_val, y_train_val)

        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(
            knn, F_train[:, sel_idx], y_train, F_test[:, sel_idx], y_test
        )
        metrics["pop_size"] = pop
        metrics["n_features"] = len(sel_idx)
        results.append(metrics)
        print(f"  Pop={pop}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, Features={len(sel_idx)}")

    print_table(results, "Table 11: Effect of GA Population Size")

    save_results_json(results, os.path.join(RESULTS_DIR, "table_11.json"))
    csv_rows = [{
        "Pop Size": r["pop_size"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "Specificity": r["specificity"],
        "Features": r["n_features"],
    } for r in results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_11.csv"))
    latex = generate_latex_table(
        csv_rows,
        caption="Effect of GA population size on classification performance",
        label="tab:table11",
        columns=[
            ("Pop Size", "Pop. Size"), ("Accuracy", "Accuracy"),
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-Score"), ("Features", "\\# Features"),
        ]
    )
    save_latex_table(latex, os.path.join(RESULTS_DIR, "table_11.tex"))
    return results


def run_table12(F_train_val, y_train_val, F_train, y_train, F_test, y_test):
    """Table 12: Effect of number of generations."""
    print("\n" + "=" * 70)
    print("  Table 12: Effect of Number of Generations")
    print("=" * 70)

    gen_values = [10, 25, 50, 100]
    results = []
    n_feat = F_train_val.shape[1]

    for n_gen in gen_values:
        print(f"\n  Generations = {n_gen} ...")
        set_seed()
        ga = AdaptiveGA(n_features=n_feat, pop_size=POP_SIZE, n_gen=n_gen)
        sel_idx, _ = ga.run(F_train_val, y_train_val)

        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(
            knn, F_train[:, sel_idx], y_train, F_test[:, sel_idx], y_test
        )
        metrics["n_gen"] = n_gen
        metrics["n_features"] = len(sel_idx)
        results.append(metrics)
        print(f"  Gen={n_gen}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, Features={len(sel_idx)}")

    print_table(results, "Table 12: Effect of Number of Generations")

    save_results_json(results, os.path.join(RESULTS_DIR, "table_12.json"))
    csv_rows = [{
        "Generations": r["n_gen"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "Specificity": r["specificity"],
        "Features": r["n_features"],
    } for r in results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_12.csv"))
    latex = generate_latex_table(
        csv_rows,
        caption="Effect of number of GA generations on classification performance",
        label="tab:table12",
        columns=[
            ("Generations", "Generations"), ("Accuracy", "Accuracy"),
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-Score"), ("Features", "\\# Features"),
        ]
    )
    save_latex_table(latex, os.path.join(RESULTS_DIR, "table_12.tex"))
    return results


def run_table13(F_train_val, y_train_val, F_train, y_train, F_test, y_test):
    """Table 13: Effect of KNN k value."""
    print("\n" + "=" * 70)
    print("  Table 13: Effect of KNN k Value")
    print("=" * 70)

    # First, run GA once to get selected features
    set_seed()
    n_feat = F_train_val.shape[1]
    ga = AdaptiveGA(n_features=n_feat)
    sel_idx, _ = ga.run(F_train_val, y_train_val)

    F_train_sel = F_train[:, sel_idx]
    F_test_sel = F_test[:, sel_idx]

    k_values = [3, 5, 7, 9]
    results = []

    for k in k_values:
        print(f"\n  k = {k} ...")
        knn = KNeighborsClassifier(n_neighbors=k, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(knn, F_train_sel, y_train, F_test_sel, y_test)
        metrics["k"] = k
        metrics["n_features"] = len(sel_idx)
        results.append(metrics)
        print(f"  k={k}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    print_table(results, "Table 13: Effect of KNN k Value")

    save_results_json(results, os.path.join(RESULTS_DIR, "table_13.json"))
    csv_rows = [{
        "k": r["k"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "Specificity": r["specificity"],
    } for r in results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_13.csv"))
    latex = generate_latex_table(
        csv_rows,
        caption="Effect of KNN k value on classification performance",
        label="tab:table13",
        columns=[
            ("k", "k"), ("Accuracy", "Accuracy"),
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-Score"), ("Specificity", "Specificity"),
        ]
    )
    save_latex_table(latex, os.path.join(RESULTS_DIR, "table_13.tex"))
    return results


def run_table14(X_train_val, y_train_val, X_train, y_train, X_test, y_test):
    """Table 14: Backbone comparison (DenseNet121, ResNet50, VGG16, EfficientNetB0)."""
    print("\n" + "=" * 70)
    print("  Table 14: Backbone Comparison")
    print("=" * 70)

    backbones = ["densenet121", "resnet50", "vgg16", "efficientnetb0"]
    results = []

    for bb_name in backbones:
        print(f"\n  Backbone: {bb_name} ...")
        set_seed()
        tf.keras.backend.clear_session()

        model, feat_dim = build_feature_extractor(bb_name, "se")
        print(f"  Feature dim: {feat_dim}")

        F_train_val = extract_features_cached(
            model, X_train_val, f"t14_{bb_name}_trainval"
        )
        F_test = extract_features_cached(
            model, X_test, f"t14_{bb_name}_test"
        )
        F_train = F_train_val[:len(X_train)]

        # Run Adaptive GA
        ga = AdaptiveGA(n_features=F_train_val.shape[1])
        sel_idx, _ = ga.run(F_train_val, y_train_val)

        # Evaluate
        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(
            knn, F_train[:, sel_idx], y_train, F_test[:, sel_idx], y_test
        )
        metrics["backbone"] = bb_name
        metrics["feature_dim"] = feat_dim
        metrics["n_features"] = len(sel_idx)
        metrics["total_params"] = model.count_params()
        results.append(metrics)

        print(f"  {bb_name}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, Features={len(sel_idx)}/{feat_dim}")

        del model

    print_table(results, "Table 14: Backbone Comparison")

    save_results_json(results, os.path.join(RESULTS_DIR, "table_14.json"))
    csv_rows = [{
        "Backbone": r["backbone"],
        "Feature Dim": r["feature_dim"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Selected": r["n_features"],
        "Total Params": r["total_params"],
    } for r in results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_14.csv"))
    latex = generate_latex_table(
        csv_rows,
        caption="Comparison of CNN backbones with SE attention + Adaptive GA + KNN",
        label="tab:table14",
        columns=[
            ("Backbone", "Backbone"),
            ("Feature Dim", "Feat. Dim"),
            ("Accuracy", "Accuracy"),
            ("F1", "F1-Score"),
            ("Selected", "Selected"),
            ("Total Params", "Params"),
        ]
    )
    save_latex_table(latex, os.path.join(RESULTS_DIR, "table_14.tex"))
    return results


def run():
    set_seed()
    print("=" * 70)
    print("  Tables 11-14: Ablation Studies")
    print("=" * 70)

    # Load data once
    print("\n[1] Loading dataset ...")
    (X, y, X_train, y_train, X_train_val, y_train_val,
     X_test, y_test) = _load_and_prepare()

    # Extract DenseNet121+SE features once for Tables 11-13
    print("\n[2] Extracting DenseNet121 + SE features ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")
    F_train_val = extract_features_cached(model, X_train_val, "t11_14_trainval_dense_se")
    F_test = extract_features_cached(model, X_test, "t11_14_test_dense_se")
    F_train = F_train_val[:len(X_train)]
    del model
    tf.keras.backend.clear_session()

    # Tables 11-13 use the same extracted features
    t11 = run_table11(F_train_val, y_train_val, F_train, y_train, F_test, y_test)
    t12 = run_table12(F_train_val, y_train_val, F_train, y_train, F_test, y_test)
    t13 = run_table13(F_train_val, y_train_val, F_train, y_train, F_test, y_test)

    # Table 14 needs raw images for different backbones
    t14 = run_table14(X_train_val, y_train_val, X_train, y_train, X_test, y_test)

    print("\nTables 11-14 complete.")
    return {"table11": t11, "table12": t12, "table13": t13, "table14": t14}


if __name__ == "__main__":
    run()
