"""
generate_detailed_analysis.py — Generate detailed plots for Prof. Hait's request.

1. Feature number vs. accuracy (vary l0_penalty to get different counts)
2. Generation number vs. accuracy (50, 100, 150)
3. Mutation vs. accuracy (0.01, 0.05, 0.1, 0.2)
4. Crossover vs. accuracy (0.4, 0.5, 0.6, 0.7, 0.8)
5. Best model (GP3, tau=0.9): ROC Curve & Confusion Matrix
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.getcwd(), "journal_experiments"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier

from journal_experiments.config import (
    set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES, SEED,
    POP_SIZE, N_GEN, CX_PROB, MUT_PROB, INDPB, CV_FOLDS
)
from journal_experiments.data_loader import load_dataset_paths, get_splits
from journal_experiments.models.backbone import build_feature_extractor, extract_features_from_paths_cached
from journal_experiments.feature_selection.adaptive_ga import AdaptiveGA
from journal_experiments.ensemble_fusion import (
    compute_epsilon_weights, apply_gp_fusion, gp3_fusion
)

# Setup directories
FIGURES_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CLASSES = len(CLASS_NAMES)

def plot_and_save(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved plot: {path}")

def run_analysis():
    set_seed()
    
    # 1. Load data and features
    print("\n[1] Preparing data and features...")
    X_paths, y = load_dataset_paths(DATA_DIR)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = get_splits(X_paths, y)
    X_trval_paths = np.concatenate([X_tr, X_val])
    y_trval = np.concatenate([y_tr, y_val])
    
    model, feat_dim = build_feature_extractor("densenet121", "se")
    
    # Use existing cache if possible
    F_trval = extract_features_from_paths_cached(model, X_trval_paths, "fill_trainval_dense_se")
    F_test = extract_features_from_paths_cached(model, X_te, "fill_test_dense_se")
    
    F_tr = F_trval[:len(X_tr)]
    F_val = F_trval[len(X_tr):]
    
    # Best parameters from previous run
    BEST_TAU = 0.9
    
    # --- Experiment 1: Feature Number vs Accuracy ---
    print("\n[Exp 1] Feature Number vs. Accuracy (varying L0 penalty)")
    PENALTIES = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    feat_counts = []
    feat_accs = []
    
    for i, p in enumerate(PENALTIES):
        print(f"  Penalty={p}")
        ga = AdaptiveGA(n_features=feat_dim, l0_penalty=p, seed=SEED + i*13)
        sel_idx, _ = ga.run(F_trval, y_trval)
        
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        
        feat_counts.append(len(sel_idx))
        feat_accs.append(acc)
    
    # Sort by feature count for plotting
    sorted_idx = np.argsort(feat_counts)
    plot_and_save(np.array(feat_counts)[sorted_idx], np.array(feat_accs)[sorted_idx], 
                  "Number of Selected Features", "Accuracy", 
                  "Feature Number vs. Accuracy (tau=0.9)", "feat_num_vs_acc.png")

    # --- Experiment 2: Generation Number vs Accuracy ---
    print("\n[Exp 2] Generation Number vs. Accuracy")
    GEN_VALUES = [25, 50, 75, 100, 125, 150]
    gen_accs = []
    for i, g in enumerate(GEN_VALUES):
        print(f"  Gens={g}")
        ga = AdaptiveGA(n_features=feat_dim, n_gen=g, seed=SEED + i*17)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        gen_accs.append(acc)
    
    plot_and_save(GEN_VALUES, gen_accs, "Number of Generations", "Accuracy",
                  "Generation Number vs. Accuracy (tau=0.9)", "gen_vs_acc.png")

    # --- Experiment 3: Mutation Probability vs Accuracy ---
    print("\n[Exp 3] Mutation vs. Accuracy")
    MUT_VALUES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    mut_accs = []
    for i, m in enumerate(MUT_VALUES):
        print(f"  Mutation={m}")
        ga = AdaptiveGA(n_features=feat_dim, mut_prob=m, seed=SEED + i*19)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        mut_accs.append(acc)
    
    plot_and_save(MUT_VALUES, mut_accs, "Mutation Probability", "Accuracy",
                  "Mutation vs. Accuracy (tau=0.9)", "mut_vs_acc.png")

    # --- Experiment 4: Crossover Probability vs Accuracy ---
    print("\n[Exp 4] Crossover vs. Accuracy")
    CX_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cx_accs = []
    for i, c in enumerate(CX_VALUES):
        print(f"  Crossover={c}")
        ga = AdaptiveGA(n_features=feat_dim, cx_prob=c, seed=SEED + i*23)
        sel_idx, _ = ga.run(F_trval, y_trval)
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        acc = accuracy_score(y_te, clf.predict(F_test[:, sel_idx]))
        cx_accs.append(acc)
    
    plot_and_save(CX_VALUES, cx_accs, "Crossover Probability", "Accuracy",
                  "Crossover vs. Accuracy (tau=0.9)", "cx_vs_acc.png")

    # --- Best Model Plots: ROC & Confusion Matrix ---
    print("\n[Final] Generating ROC and Confusion Matrix for best model (GP3, tau=0.9)")
    # Re-run best setup (or use saved if possible, but re-run for fresh data)
    ga = AdaptiveGA(n_features=feat_dim, seed=SEED)
    sel_idx, _ = ga.run(F_trval, y_trval)
    
    # Train 5 classifiers to get top-4 for fusion
    from journal_experiments.classifiers import get_all_classifiers
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
    
    # GP3 Fusion with tau=0.9
    final_proba = apply_gp_fusion(top4_probas, top4_eps, gp3_fusion, threshold=BEST_TAU)
    final_pred = np.argmax(final_proba, axis=1)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_te, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Best Model: GP3, tau=0.9)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_best.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Saved Confusion Matrix: {cm_path}")
    
    # 2. ROC Curve
    y_te_bin = label_binarize(y_te, classes=range(N_CLASSES))
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], final_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve (Best Model: GP3, tau=0.9)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve_best.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  Saved ROC Curve: {roc_path}")

if __name__ == "__main__":
    run_analysis()
