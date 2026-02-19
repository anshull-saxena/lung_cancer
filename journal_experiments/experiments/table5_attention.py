"""
Table 5: Attention mechanism comparison.

For each attention type (SE, ECA, CBAM, Split, Dual, ViT, Swin):
  - Build DenseNet121 + attention
  - Extract features
  - Run Adaptive GA
  - Train KNN, evaluate
Comparative table: rows = attention types, cols = metrics + param count.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES
from data_loader import load_dataset, get_splits
from models.backbone import build_feature_extractor, extract_features_cached
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import train_and_evaluate
from sklearn.neighbors import KNeighborsClassifier
from config import KNN_K, KNN_WEIGHTS
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


ATTENTION_TYPES = ["se", "eca", "cbam", "split", "dual", "vit", "swin"]


def count_attention_params(model, backbone_name="densenet121"):
    """Count parameters added by the attention module (total - backbone)."""
    total = model.count_params()
    # Build a baseline without attention to compare
    from tensorflow.keras.applications import DenseNet121
    base = DenseNet121(include_top=False, weights="imagenet",
                       input_shape=(224, 224, 3))
    base_params = base.count_params()
    # Approximate: total model params = backbone + attention + GAP(0)
    attn_params = total - base_params
    return max(attn_params, 0)


def run():
    set_seed()
    print("=" * 70)
    print("  Table 5: Attention Mechanism Comparison")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1] Loading dataset ...")
    X, y = load_dataset(DATA_DIR)

    # 2. Split
    print("\n[2] Splitting data ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    # 3. For each attention mechanism
    table5_results = []

    for attn_name in ATTENTION_TYPES:
        print(f"\n[3] Processing attention: {attn_name.upper()} ...")

        # Build model
        model, feat_dim = build_feature_extractor("densenet121", attn_name)
        attn_params = count_attention_params(model)
        print(f"  Attention params: {attn_params:,}")

        # Extract features
        cache_name = f"t5_{attn_name}_trainval"
        F_train_val = extract_features_cached(model, X_train_val, cache_name)
        cache_name_test = f"t5_{attn_name}_test"
        F_test = extract_features_cached(model, X_test, cache_name_test)

        F_train = F_train_val[:len(X_train)]

        # Run Adaptive GA
        print(f"  Running Adaptive GA ...")
        ga = AdaptiveGA(n_features=F_train_val.shape[1])
        sel_idx, _ = ga.run(F_train_val, y_train_val)

        # Evaluate with KNN
        F_train_sel = F_train[:, sel_idx]
        F_test_sel = F_test[:, sel_idx]

        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        metrics = train_and_evaluate(knn, F_train_sel, y_train, F_test_sel, y_test)
        metrics["attention"] = attn_name.upper()
        metrics["n_features"] = len(sel_idx)
        metrics["attn_params"] = attn_params
        table5_results.append(metrics)

        print(f"  {attn_name.upper()}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, Features={len(sel_idx)}")

        # Clean up model to free memory
        del model
        import tensorflow as tf
        tf.keras.backend.clear_session()

    # Print
    print_table(table5_results, "Table 5: Attention Mechanism Comparison")

    # Save
    save_results_json(table5_results, os.path.join(RESULTS_DIR, "table_5.json"))

    csv_rows = [{
        "Attention": r["attention"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Specificity": r["specificity"],
        "Features": r["n_features"],
        "Attn Params": r["attn_params"],
    } for r in table5_results]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table_5.csv"))

    latex5 = generate_latex_table(
        csv_rows,
        caption="Comparison of attention mechanisms with DenseNet121 backbone + Adaptive GA + KNN",
        label="tab:table5",
        columns=[
            ("Attention", "Attention"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1-Score"),
            ("Specificity", "Specificity"),
            ("Features", "\\# Features"),
            ("Attn Params", "Attn. Params"),
        ]
    )
    save_latex_table(latex5, os.path.join(RESULTS_DIR, "table_5.tex"))

    print("\nTable 5 complete.")
    return table5_results


if __name__ == "__main__":
    run()
