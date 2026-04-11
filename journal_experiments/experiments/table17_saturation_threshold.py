"""
Table 17: Saturation Threshold (T) Sensitivity Analysis.

Varies the saturation threshold T in {0.6, 0.7, 0.8, 0.9} for GP ensemble
operators. For each T, runs all ensemble methods and reports extended metrics:
Accuracy, Precision, Sensitivity (Recall), Specificity, F1, AUC, MCC.
Also generates per-class ROC curves for each T value.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, roc_curve, auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from config import set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES, SEED
from data_loader import load_dataset_paths, get_splits
from models.backbone import (
    build_feature_extractor,
    extract_features_from_paths_cached,
)
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import get_all_classifiers, train_and_evaluate, compute_specificity
from ensemble_fusion import (
    compute_epsilon_weights, weighted_probability_fusion,
    apply_gp_fusion, gp1_fusion, gp2_fusion, gp3_fusion, gp4_fusion, gp5_fusion,
)
from evaluation import (print_table, save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")


def _extended_metrics(y_true, y_pred, probas, method_name, class_names=CLASS_NAMES):
    """Compute all requested metrics including AUC, MCC, and ROC data."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    sens = recall_score(y_true, y_pred, average="macro", zero_division=0)  # sensitivity = recall
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    spec_per = compute_specificity(y_true, y_pred)
    spec = float(np.mean(spec_per))
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC (macro, one-vs-rest)
    try:
        auc_val = roc_auc_score(y_bin, probas, multi_class="ovr", average="macro")
    except ValueError:
        auc_val = 0.0

    # Per-class ROC curves
    roc_data = {}
    for i, cn in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])
        roc_auc_i = auc(fpr, tpr)
        roc_data[cn] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_i)}

    return {
        "method": method_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "auc": float(auc_val),
        "mcc": float(mcc),
        "roc_data": roc_data,
        "y_pred": y_pred.tolist(),
    }


def _plot_roc_curves(results, threshold_val, class_names=CLASS_NAMES):
    """Plot per-class ROC curves for all methods at a given threshold."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 4.5))
    if len(class_names) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for ax, cn in zip(axes, class_names):
        for idx, r in enumerate(results):
            roc = r.get("roc_data", {}).get(cn)
            if roc is None:
                continue
            ax.plot(roc["fpr"], roc["tpr"],
                    color=colors[idx], linewidth=1.5,
                    label=f'{r["method"]} (AUC={roc["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {cn}")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    fig.suptitle(f"ROC Curves (Saturation Threshold T = {threshold_val})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"roc_T{threshold_val}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC plot saved to {path}")
    return path


def run():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch size for CNN feature extraction")
    ap.add_argument("--force-features", action="store_true",
                    help="Recompute feature cache even if .npy exists")
    args, _ = ap.parse_known_args()

    set_seed()
    print("=" * 70)
    print("  Table 17: Saturation Threshold (T) Sensitivity Analysis")
    print("=" * 70)

    # ---- 1. Load dataset ----
    print("\n[1] Indexing dataset (paths only; low-memory) ...")
    X, y = load_dataset_paths(DATA_DIR)

    # ---- 2. Split ----
    print("\n[2] Splitting data ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # ---- 3. Feature extraction ----
    print("\n[3] Building DenseNet121 + SE feature extractor ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")

    print("\n[4] Extracting features ...")
    F_train_val = extract_features_from_paths_cached(
        model,
        X_train_val,
        "t17_trainval_dense_se",
        force=args.force_features,
        batch_size=args.batch_size,
    )
    F_test = extract_features_from_paths_cached(
        model,
        X_test,
        "t17_test_dense_se",
        force=args.force_features,
        batch_size=args.batch_size,
    )
    F_train = F_train_val[:len(X_train)]
    F_val = F_train_val[len(X_train):]

    # ---- 4. Adaptive GA feature selection ----
    print("\n[5] Running Adaptive GA ...")
    ga = AdaptiveGA(n_features=F_train_val.shape[1])
    sel_idx, _ = ga.run(F_train_val, y_train_val)
    print(f"  Selected {len(sel_idx)} / {F_train_val.shape[1]} features")

    F_train_sel = F_train[:, sel_idx]
    F_val_sel = F_val[:, sel_idx]
    F_test_sel = F_test[:, sel_idx]

    # ---- 5. Train all 5 classifiers (once — threshold only affects fusion) ----
    print("\n[6] Training all 5 classifiers ...")
    classifiers = get_all_classifiers()
    clf_names = list(classifiers.keys())

    probas_dict = {}
    val_acc_dict = {}

    for name, clf in classifiers.items():
        print(f"  Training {name} ...")
        clf.fit(F_train_sel, y_train)
        val_pred = clf.predict(F_val_sel)
        val_acc = accuracy_score(y_val, val_pred)
        val_acc_dict[name] = val_acc
        print(f"    Val accuracy: {val_acc:.4f}")
        probas_dict[name] = clf.predict_proba(F_test_sel)

    # Compute epsilon weights (once)
    accs = np.array([val_acc_dict[n] for n in clf_names])
    epsilon, ranked_idx = compute_epsilon_weights(accs)
    ranked_names = [clf_names[i] for i in ranked_idx]
    ranked_probas = [probas_dict[n] for n in ranked_names]
    top4_probas = ranked_probas[:4]
    top4_eps = epsilon[:4]

    # Also compute individual classifier predictions for majority vote
    from scipy.stats import mode as scipy_mode
    all_preds = [np.argmax(probas_dict[n], axis=1) for n in clf_names]
    stacked = np.stack(all_preds, axis=0)
    mv_result = scipy_mode(stacked, axis=0, keepdims=False)
    mv_pred = np.asarray(mv_result.mode).flatten()
    # Majority vote probas: average of all classifier probas
    mv_probas = np.mean([probas_dict[n] for n in clf_names], axis=0)

    # Weighted fusion probas (no saturation — threshold-independent)
    wpf_proba = weighted_probability_fusion(ranked_probas, epsilon)

    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # ---- 6. Loop over thresholds ----
    all_threshold_results = {}
    combined_csv_rows = []

    gp_operators = [
        ("GP1 (Max+Sat)", gp1_fusion),
        ("GP2 (Noisy-OR+Sat)", gp2_fusion),
        ("GP3 (Soft OR+Sat)", gp3_fusion),
        ("GP4 (Min×Prod+Sat)", gp4_fusion),
        ("GP5 (Norm Max+Sat)", gp5_fusion),
    ]
 
    for T in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f" Threshold T = {T}")
        print(f"{'='*60}")

        results_T = []

        # Majority Vote (threshold-independent)
        results_T.append(_extended_metrics(y_test, mv_pred, mv_probas,
                                           "Majority Vote"))

        # Weighted Fusion (threshold-independent)
        wpf_pred = np.argmax(wpf_proba, axis=1)
        results_T.append(_extended_metrics(y_test, wpf_pred, wpf_proba,
                                           "Weighted Fusion"))

        # GP1–GP5 with current threshold
        for gp_name, gp_func in gp_operators:
            gp_proba = apply_gp_fusion(top4_probas, top4_eps, gp_func,
                                       normalize=True, threshold=T)
            gp_pred = np.argmax(gp_proba, axis=1)
            results_T.append(_extended_metrics(y_test, gp_pred, gp_proba, gp_name))

        # Print table
        print_rows = [{k: v for k, v in r.items() if k not in ("roc_data", "y_pred")}
                      for r in results_T]
        print_table(print_rows, f"Table 17: Ensemble Results (T = {T})")

        # ROC plot
        _plot_roc_curves(results_T, T)

        # Collect for saving
        all_threshold_results[str(T)] = results_T

        # CSV rows (per-threshold table)
        csv_rows_T = [{
            "Method": r["method"],
            "Accuracy": f'{r["accuracy"]:.4f}',
            "Precision": f'{r["precision"]:.4f}',
            "Sensitivity": f'{r["sensitivity"]:.4f}',
            "Specificity": f'{r["specificity"]:.4f}',
            "F1": f'{r["f1"]:.4f}',
            "AUC": f'{r["auc"]:.4f}',
            "MCC": f'{r["mcc"]:.4f}',
        } for r in results_T]
        save_results_csv(csv_rows_T,
                         os.path.join(RESULTS_DIR, f"table_17_T{T}.csv"))

        # LaTeX per threshold
        latex_T = generate_latex_table(
            csv_rows_T,
            caption=f"Ensemble fusion comparison with saturation threshold $T = {T}$",
            label=f"tab:table17_T{str(T).replace('.', '')}",
            columns=[
                ("Method", "Method"),
                ("Accuracy", "Accuracy"),
                ("Precision", "Precision"),
                ("Sensitivity", "Sensitivity"),
                ("Specificity", "Specificity"),
                ("F1", "F1-Score"),
                ("AUC", "AUC"),
                ("MCC", "MCC"),
            ]
        )
        save_latex_table(latex_T, os.path.join(RESULTS_DIR, f"table_17_T{T}.tex"))

        # Combined rows with T column
        for r in csv_rows_T:
            row_with_T = {"T": T}
            row_with_T.update(r)
            combined_csv_rows.append(row_with_T)

    # ---- 7. Save combined results ----
    save_results_csv(combined_csv_rows,
                     os.path.join(RESULTS_DIR, "table_17_combined.csv"))

    # JSON with ROC data
    json_data = {}
    for T_str, results_T in all_threshold_results.items():
        json_data[T_str] = [{k: v for k, v in r.items() if k != "y_pred"}
                            for r in results_T]
    save_results_json(json_data, os.path.join(RESULTS_DIR, "table_17.json"))

    # Combined LaTeX
    combined_latex = generate_latex_table(
        combined_csv_rows,
        caption="Ensemble fusion comparison across saturation thresholds $T \\in \\{0.6, 0.7, 0.8, 0.9\\}$",
        label="tab:table17_combined",
        columns=[
            ("T", "$T$"),
            ("Method", "Method"),
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Sensitivity", "Sensitivity"),
            ("Specificity", "Specificity"),
            ("F1", "F1"),
            ("AUC", "AUC"),
            ("MCC", "MCC"),
        ]
    )
    save_latex_table(combined_latex, os.path.join(RESULTS_DIR, "table_17.tex"))

    print("\n" + "=" * 70)
    print("  Table 17 complete.")
    print("=" * 70)
    return all_threshold_results


if __name__ == "__main__":
    run()
