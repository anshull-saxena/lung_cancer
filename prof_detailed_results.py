"""
prof_detailed_results.py — Complete implementation for Professor Hait's request.

Tasks handled:
1. Additional Evaluation Metrics: Accuracy, Sensitivity, Specificity, AUC, F1-score, MCC.
2. Plots:
   - ROC curve (AUC-ROC) for the best model.
   - Confusion Matrix for the best model.
   - Feature number vs. Accuracy (varying L0 penalty to control feature count).
   - Generation number vs. Accuracy.
   - Mutation probability vs. Accuracy.
   - Crossover probability vs. Accuracy.
3. Attention Module Analysis: Comparison and identification of the best module.

Usage: Run this script to generate all figures and CSV results.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, roc_curve, auc,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# For running inside journal_experiments/experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from journal_experiments.config import (
    set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES, SEED,
    POP_SIZE, N_GEN, CX_PROB, MUT_PROB, INDPB, CV_FOLDS,
    KNN_K, KNN_WEIGHTS
)
from journal_experiments.data_loader import load_dataset_paths, get_splits
from journal_experiments.models.backbone import build_feature_extractor, extract_features_from_paths_cached
from journal_experiments.feature_selection.adaptive_ga import AdaptiveGA
from journal_experiments.ensemble_fusion import (
    compute_epsilon_weights, apply_gp_fusion, gp3_fusion
)
from journal_experiments.classifiers import get_all_classifiers, compute_specificity

# Setup Output Directories
FIGURES_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CLASSES = len(CLASS_NAMES)
BEST_TAU = 0.9  # As identified in previous experiments

# ═══════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def get_extended_metrics(y_true, y_pred, probas=None):
    """Compute all metrics requested by the professor."""
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    spec = float(np.mean(compute_specificity(y_true, y_pred)))
    
    auc_val = 0.0
    if probas is not None:
        y_bin = label_binarize(y_true, classes=range(N_CLASSES))
        auc_val = roc_auc_score(y_bin, probas, multi_class="ovr", average="macro")
        
    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1-score": f1,
        "AUC": auc_val,
        "MCC": mcc
    }

def plot_sensitivity(x_values, y_values, xlabel, title, filename):
    """Generate and save sensitivity plots."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Plot saved: {filename}")

# ═══════════════════════════════════════════════════════════════════════
#  Main Analysis Implementation
# ═══════════════════════════════════════════════════════════════════════

def run_prof_request():
    set_seed()
    print("\n" + "="*70)
    print("  COMMENCING DETAILED ANALYSIS FOR PROFESSOR")
    print("="*70)

    # 1. Prepare Data
    print("\n[1] Preparing features...")
    X_paths, y = load_dataset_paths(DATA_DIR)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = get_splits(X_paths, y)
    X_trval_paths = np.concatenate([X_tr, X_val])
    y_trval = np.concatenate([y_tr, y_val])
    
    # Load DenseNet121+SE features (Default Backbone)
    model, feat_dim = build_feature_extractor("densenet121", "se")
    F_trval = extract_features_from_paths_cached(model, X_trval_paths, "fill_trainval_dense_se")
    F_test = extract_features_from_paths_cached(model, X_te, "fill_test_dense_se")
    
    F_tr = F_trval[:len(X_tr)]
    F_val = F_trval[len(X_tr):]

    # 2. Parameter Sensitivity Experiments (tau=0.9 fixed)
    print("\n[2] Running Parameter Sensitivity Studies (tau=0.9 fixed)...")
    
    # A. Feature Number vs. Accuracy (Vary L0 Penalty)
    print("  Exp A: Feature Number vs. Accuracy")
    PENALTIES = [0.0, 0.001, 0.005, 0.01, 0.02]
    feat_results = []
    for i, p in enumerate(PENALTIES):
        ga = AdaptiveGA(n_features=feat_dim, l0_penalty=p, seed=SEED + i)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        feat_results.append({"n_features": len(sel_idx), "accuracy": acc})
    
    feat_df = pd.DataFrame(feat_results).sort_values("n_features")
    plot_sensitivity(feat_df["n_features"], feat_df["accuracy"], "Number of Selected Features", 
                     "Feature Number vs. Accuracy", "prof_feat_vs_acc.png")

    # B. Generation Number vs. Accuracy
    print("  Exp B: Generations vs. Accuracy")
    GENS = [20, 50, 100, 150]
    gen_accs = []
    for i, g in enumerate(GENS):
        ga = AdaptiveGA(n_features=feat_dim, n_gen=g, seed=SEED + i*7)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        gen_accs.append(acc)
    plot_sensitivity(GENS, gen_accs, "Number of Generations", "Generations vs. Accuracy", "prof_gen_vs_acc.png")

    # C. Mutation Probability vs. Accuracy
    print("  Exp C: Mutation vs. Accuracy")
    MUTS = [0.01, 0.05, 0.1, 0.15, 0.2]
    mut_accs = []
    for i, m in enumerate(MUTS):
        ga = AdaptiveGA(n_features=feat_dim, mut_prob=m, seed=SEED + i*11)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        mut_accs.append(acc)
    plot_sensitivity(MUTS, mut_accs, "Mutation Probability", "Mutation vs. Accuracy", "prof_mut_vs_acc.png")

    # D. Crossover Probability vs. Accuracy
    print("  Exp D: Crossover vs. Accuracy")
    CXS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cx_accs = []
    for i, c in enumerate(CXS):
        ga = AdaptiveGA(n_features=feat_dim, cx_prob=c, seed=SEED + i*13)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        cx_accs.append(acc)
    plot_sensitivity(CXS, cx_accs, "Crossover Probability", "Crossover vs. Accuracy", "prof_cx_vs_acc.png")

    # 3. Best Model Visuals (GP3 Fusion, tau=0.9)
    print("\n[3] Generating Visuals for Best Model (GP3, tau=0.9)...")
    ga = AdaptiveGA(n_features=feat_dim, seed=SEED)
    sel_idx, _ = ga.run(F_trval, y_trval)
    
    clfs = get_all_classifiers()
    probas = {}
    val_accs = []
    clf_names = list(clfs.keys())
    
    for name, clf in clfs.items():
        clf.fit(F_tr[:, sel_idx], y_tr)
        val_accs.append(accuracy_score(y_val, clf.predict(F_val[:, sel_idx])))
        probas[name] = clf.predict_proba(F_test[:, sel_idx])
    
    epsilon, ranked_idx = compute_epsilon_weights(np.array(val_accs))
    top4_probas = [probas[clf_names[i]] for i in ranked_idx[:4]]
    top4_eps = epsilon[:4]
    
    final_proba = apply_gp_fusion(top4_probas, top4_eps, gp3_fusion, threshold=BEST_TAU)
    final_pred = np.argmax(final_proba, axis=1)
    
    # Metrics
    metrics = get_extended_metrics(y_te, final_pred, final_proba)
    print("\nFinal Model Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_te, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (GP3 Fusion, tau=0.9)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(FIGURES_DIR, "prof_confusion_matrix.png"), dpi=300)
    plt.close()

    # ROC Curve
    y_te_bin = label_binarize(y_te, classes=range(N_CLASSES))
    plt.figure(figsize=(8, 6))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], final_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title("Multi-class ROC Curve (GP3 Fusion, tau=0.9)", fontsize=14, fontweight='bold')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_DIR, "prof_roc_curve.png"), dpi=300)
    plt.close()

    # 4. Attention Module Comparison
    print("\n[4] Attention Module Comparison...")
    ATTNS = ["se", "eca", "cbam", "split", "dual", "vit", "swin"]
    attn_results = []
    for attn in ATTNS:
        print(f"  Processing {attn.upper()}...")
        m_extractor, _ = build_feature_extractor("densenet121", attn)
        # Assuming cache exists for user's convenience
        f_trval_attn = extract_features_from_paths_cached(m_extractor, X_trval_paths, f"t5_{attn}_trainval")
        f_test_attn = extract_features_from_paths_cached(m_extractor, X_te, f"t5_{attn}_test")
        
        # Fast GA for attention comparison
        ga_attn = AdaptiveGA(n_features=f_trval_attn.shape[1], n_gen=50, seed=SEED)
        sel_idx_attn, _ = ga_attn.run(f_trval_attn, y_trval)
        
        clf_attn = KNeighborsClassifier(n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1)
        clf_attn.fit(f_trval_attn[:len(X_tr), sel_idx_attn], y_tr)
        acc_attn = accuracy_score(y_te, clf_attn.predict(f_test_attn[:, sel_idx_attn]))
        
        attn_results.append({"Attention": attn.upper(), "Accuracy": acc_attn})
    
    attn_df = pd.DataFrame(attn_results)
    print("\nAttention Comparison:")
    print(attn_df)
    attn_df.to_csv(os.path.join(RESULTS_DIR, "prof_attention_comparison.csv"), index=False)
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE. Figures in 'figures/', CSVs in 'results/'.")
    print("="*70)

if __name__ == "__main__":
    run_prof_request()
