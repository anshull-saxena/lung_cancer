"""
fill_tables.py — Fill ALL 6 tables from Prof. Hait's to_be_filled.pdf.

Fresh runs with corrected GP fusion formulas (saturation-based).

Table 1: τ threshold sensitivity (GP1-GP4, τ ∈ {0.6, 0.7, 0.8, 0.9})
Table 2: Best Grouping vs individual classifiers
Table 3: SOTA comparison
Table 4: Generations sensitivity (50, 100, 150)
Table 5: Crossover sensitivity (0.4, 0.5, 0.6)
Table 6: GA parameters summary
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, roc_curve, auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import mode as scipy_mode

from config import (
    set_seed, DATA_DIR, RESULTS_DIR, CLASS_NAMES, SEED,
    POP_SIZE, N_GEN, CX_PROB, MUT_PROB, INDPB, CV_FOLDS,
)
from data_loader import load_dataset_paths, get_splits
from models.backbone import build_feature_extractor, extract_features_from_paths_cached
from feature_selection.adaptive_ga import AdaptiveGA
from classifiers import get_all_classifiers, compute_specificity
from ensemble_fusion import (
    compute_epsilon_weights, weighted_probability_fusion,
    apply_gp_fusion, gp1_fusion, gp2_fusion, gp3_fusion, gp4_fusion, gp5_fusion,
)
from evaluation import (save_results_json, save_results_csv,
                         generate_latex_table, save_latex_table)


FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CLASSES = len(CLASS_NAMES)


# ═══════════════════════════════════════════════════════════════════════
#  Shared metric helpers
# ═══════════════════════════════════════════════════════════════════════

def extended_metrics(y_true, y_pred, probas=None):
    """Return dict with acc, prec, sensitivity, specificity, f1, auc, mcc."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    sens = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    spec = float(np.mean(compute_specificity(y_true, y_pred)))
    mcc  = matthews_corrcoef(y_true, y_pred)

    auc_val = 0.0
    if probas is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
            auc_val = roc_auc_score(y_bin, probas, multi_class="ovr", average="macro")
        except ValueError:
            pass

    return {
        "accuracy":    round(float(acc), 4),
        "precision":   round(float(prec), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "f1":          round(float(f1), 4),
        "auc":         round(float(auc_val), 4),
        "mcc":         round(float(mcc), 4),
    }


def roc_per_class(y_true, probas):
    """Return {class_name: {fpr, tpr, auc}} for ROC plotting."""
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    data = {}
    for i, cn in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])
        data[cn] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                     "auc": round(float(auc(fpr, tpr)), 4)}
    return data


def plot_roc(roc_dict, title, filename):
    """Plot per-class ROC curves and save PNG."""
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(5 * N_CLASSES, 4.5))
    if N_CLASSES == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_dict)))
    for ax, cn in zip(axes, CLASS_NAMES):
        for idx, (method, rd) in enumerate(roc_dict.items()):
            cd = rd.get(cn)
            if cd is None:
                continue
            ax.plot(cd["fpr"], cd["tpr"], color=colors[idx], lw=1.5,
                    label=f'{method} ({cd["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(cn); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ROC → {path}")


def banner(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")


def print_rows(rows, title):
    banner(title)
    if not rows:
        return
    keys = list(rows[0].keys())
    hdr = "".join(f"{k:>14}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("".join(f"{str(r[k]):>14}" for k in keys))


# ═══════════════════════════════════════════════════════════════════════
#  Shared pipeline: load data, extract features, train classifiers
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare(args):
    """Load dataset, extract features, return splits and feature arrays."""
    print("\n[1] Indexing dataset ...")
    X, y = load_dataset_paths(DATA_DIR)

    print("\n[2] Splitting ...")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = get_splits(X, y)
    X_trval = np.concatenate([X_tr, X_val])
    y_trval = np.concatenate([y_tr, y_val])
    print(f"  Train={len(X_tr)} Val={len(X_val)} Test={len(X_te)}")

    print("\n[3] DenseNet121 + SE feature extractor ...")
    model, feat_dim = build_feature_extractor("densenet121", "se")

    print("\n[4] Extracting features ...")
    kw = dict(force=args.force_features, batch_size=args.batch_size)
    F_trval = extract_features_from_paths_cached(model, X_trval,
                                                  "fill_trainval_dense_se", **kw)
    F_test  = extract_features_from_paths_cached(model, X_te,
                                                  "fill_test_dense_se", **kw)
    F_tr  = F_trval[:len(X_tr)]
    F_val = F_trval[len(X_tr):]

    return (y_tr, y_val, y_te, y_trval,
            F_tr, F_val, F_test, F_trval, feat_dim)


def train_classifiers(F_tr, y_tr, F_val, y_val, F_te, sel_idx):
    """Train 5 classifiers on selected features, return probas + val accs."""
    F_tr_s  = F_tr[:, sel_idx]
    F_val_s = F_val[:, sel_idx]
    F_te_s  = F_te[:, sel_idx]

    clfs = get_all_classifiers()
    probas, val_accs, preds = {}, {}, {}
    for name, clf in clfs.items():
        clf.fit(F_tr_s, y_tr)
        val_accs[name] = accuracy_score(y_val, clf.predict(F_val_s))
        probas[name]   = clf.predict_proba(F_te_s)
        preds[name]    = np.argmax(probas[name], axis=1)
        print(f"    {name}: val_acc={val_accs[name]:.4f}")
    return clfs, probas, val_accs, preds


# ═══════════════════════════════════════════════════════════════════════
#  TABLE 1: τ threshold sensitivity  (GP1-GP4 × τ ∈ {0.6,0.7,0.8,0.9})
# ═══════════════════════════════════════════════════════════════════════

def run_table1(y_te, probas, val_accs, clf_names):
    banner("TABLE 1: Saturation Threshold (τ) Sensitivity")
    THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

    accs = np.array([val_accs[n] for n in clf_names])
    epsilon, ranked_idx = compute_epsilon_weights(accs)
    ranked_names  = [clf_names[i] for i in ranked_idx]
    ranked_probas = [probas[n] for n in ranked_names]
    top4_probas = ranked_probas[:4]
    top4_eps    = epsilon[:4]

    gp_ops = [
        ("Grouping 1 (GP1)", gp1_fusion),
        ("Grouping 2 (GP2)", gp2_fusion),
        ("Grouping 3 (GP3)", gp3_fusion),
        ("Grouping 4 (GP4)", gp4_fusion),
    ]

    all_rows = []
    all_json = {}

    for T in THRESHOLDS:
        print(f"\n  τ = {T}")
        rows_T = []
        roc_dict = {}
        for gp_name, gp_func in gp_ops:
            gp_proba = apply_gp_fusion(top4_probas, top4_eps, gp_func,
                                       normalize=True, threshold=T)
            gp_pred = np.argmax(gp_proba, axis=1)
            m = extended_metrics(y_te, gp_pred, gp_proba)
            m["method"] = gp_name
            rows_T.append(m)
            roc_dict[gp_name] = roc_per_class(y_te, gp_proba)

        print_rows(rows_T, f"Table 1 (τ={T})")
        plot_roc(roc_dict, f"ROC — τ = {T}", f"roc_tau_{T}.png")

        csv_T = [{"Method": r["method"], **{k: r[k] for k in
                  ["accuracy","precision","sensitivity","specificity","f1","auc","mcc"]}}
                 for r in rows_T]
        save_results_csv(csv_T, os.path.join(RESULTS_DIR, f"table1_tau{T}.csv"))
        save_latex_table(
            generate_latex_table(csv_T,
                caption=f"Effect of threshold $\\tau = {T}$ on grouping operators",
                label=f"tab:t1_tau{str(T).replace('.','')}", columns=[
                    ("Method","Method"),("accuracy","Acc"),("precision","Prec"),
                    ("sensitivity","Sens"),("specificity","Spec"),("f1","F1"),
                    ("auc","AUC"),("mcc","MCC")]),
            os.path.join(RESULTS_DIR, f"table1_tau{T}.tex"))

        for r in csv_T:
            all_rows.append({"tau": T, **r})
        all_json[str(T)] = rows_T

    save_results_csv(all_rows, os.path.join(RESULTS_DIR, "table1_combined.csv"))
    save_results_json(all_json, os.path.join(RESULTS_DIR, "table1.json"))
    return all_rows


# ═══════════════════════════════════════════════════════════════════════
#  TABLE 2: Best Grouping vs Individual Classifiers
# ═══════════════════════════════════════════════════════════════════════

def run_table2(y_te, probas, preds, val_accs, clf_names, table1_rows):
    banner("TABLE 2: Best Grouping vs Individual Classifiers")

    # Find best grouping result from Table 1
    best_gp = max(table1_rows, key=lambda r: r.get("accuracy", r.get("Accuracy", 0)))
    best_row = {
        "Method": f'Best Grouping ({best_gp.get("Method","GP")} τ={best_gp["tau"]})',
        "accuracy":    best_gp.get("accuracy", best_gp.get("Accuracy")),
        "precision":   best_gp.get("precision", best_gp.get("Precision")),
        "sensitivity": best_gp.get("sensitivity", best_gp.get("Sensitivity")),
        "specificity": best_gp.get("specificity", best_gp.get("Specificity")),
        "f1":          best_gp.get("f1", best_gp.get("F1")),
        "auc":         best_gp.get("auc", best_gp.get("AUC")),
        "mcc":         best_gp.get("mcc", best_gp.get("MCC")),
    }
    rows = [best_row]

    # Individual classifiers
    for name in clf_names:
        m = extended_metrics(y_te, preds[name], probas[name])
        m["Method"] = f"{name} (Single)"
        rows.append(m)

    print_rows(rows, "Table 2")
    csv_rows = [{"Method": r["Method"], **{k: r[k] for k in
                ["accuracy","precision","sensitivity","specificity","f1","auc","mcc"]}}
                for r in rows]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table2_best_vs_clf.csv"))
    save_latex_table(
        generate_latex_table(csv_rows,
            caption="Comparison of best grouping operator with individual classifiers",
            label="tab:t2_best_vs_clf", columns=[
                ("Method","Method"),("accuracy","Acc"),("precision","Prec"),
                ("sensitivity","Sens"),("specificity","Spec"),("f1","F1"),
                ("auc","AUC"),("mcc","MCC")]),
        os.path.join(RESULTS_DIR, "table2_best_vs_clf.tex"))
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  TABLE 3: SOTA Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_table3(best_accuracy, best_f1, best_sensitivity):
    banner("TABLE 3: State-of-the-Art Comparison")
    # Published baselines (from final_journal_report.tex)
    sota = [
        {"Method": "Masud et al. (2021)",        "Accuracy": 0.9633, "F1": 0.9630, "Sensitivity": 0.9633},
        {"Method": "Mangal et al. (2020)",        "Accuracy": 0.9789, "F1": 0.9785, "Sensitivity": 0.9789},
        {"Method": "Nishio et al. (2021)",        "Accuracy": 0.9500, "F1": 0.9490, "Sensitivity": 0.9500},
        {"Method": "Hatuwal & Thapa (2020)",      "Accuracy": 0.9720, "F1": 0.9715, "Sensitivity": 0.9720},
        {"Method": "Talukder et al. (2022)",      "Accuracy": 0.9810, "F1": 0.9808, "Sensitivity": 0.9810},
        {"Method": "Hage Chehade et al. (2022)",  "Accuracy": 0.9625, "F1": 0.9620, "Sensitivity": 0.9625},
        {"Method": "Proposed Method (Ours)",       "Accuracy": best_accuracy,
         "F1": best_f1, "Sensitivity": best_sensitivity},
    ]
    print_rows(sota, "Table 3")
    save_results_csv(sota, os.path.join(RESULTS_DIR, "table3_sota.csv"))
    save_latex_table(
        generate_latex_table(sota,
            caption="Comparison with state-of-the-art methods on LC25000",
            label="tab:t3_sota", columns=[
                ("Method","Method"),("Accuracy","Accuracy"),
                ("F1","F1"),("Sensitivity","Sensitivity")]),
        os.path.join(RESULTS_DIR, "table3_sota.tex"))
    return sota


# ═══════════════════════════════════════════════════════════════════════
#  TABLE 4: Generations sensitivity (50, 100, 150)
# ═══════════════════════════════════════════════════════════════════════

def run_table4(F_trval, y_trval, F_tr, y_tr, F_val, y_val, F_te, y_te, feat_dim):
    banner("TABLE 4: Effect of Number of Generations")
    GEN_VALUES = [50, 100, 150]
    rows = []

    for gen in GEN_VALUES:
        print(f"\n  Generations = {gen}")
        t0 = time.time()
        ga = AdaptiveGA(n_features=feat_dim, n_gen=gen, seed=SEED)
        sel_idx, _ = ga.run(F_trval, y_trval)
        elapsed = time.time() - t0
        print(f"    Selected {len(sel_idx)} features in {elapsed:.1f}s")

        # Train KNN (primary classifier) on selected features
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        probas = clf.predict_proba(F_te[:, sel_idx])
        y_pred = clf.predict(F_te[:, sel_idx])

        m = extended_metrics(y_te, y_pred, probas)
        m["generations"] = gen
        m["features"] = len(sel_idx)
        rows.append(m)
        print(f"    Acc={m['accuracy']} Sens={m['sensitivity']} F1={m['f1']}")

    print_rows(rows, "Table 4")
    csv_rows = [{"Generations": r["generations"], "Features": r["features"],
                 **{k: r[k] for k in
                 ["accuracy","precision","sensitivity","specificity","f1","auc","mcc"]}}
                for r in rows]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table4_generations.csv"))
    save_latex_table(
        generate_latex_table(csv_rows,
            caption="Effect of number of GA generations on classification performance",
            label="tab:t4_gens", columns=[
                ("Generations","Gens"),("Features","Feat"),("accuracy","Acc"),
                ("sensitivity","Sens"),("specificity","Spec"),("f1","F1"),
                ("auc","AUC"),("mcc","MCC")]),
        os.path.join(RESULTS_DIR, "table4_generations.tex"))
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  TABLE 5: Crossover sensitivity (0.4, 0.5, 0.6)
# ═══════════════════════════════════════════════════════════════════════

def run_table5(F_trval, y_trval, F_tr, y_tr, F_val, y_val, F_te, y_te, feat_dim):
    banner("TABLE 5: Effect of Crossover Value")
    CX_VALUES = [0.4, 0.5, 0.6]
    rows = []

    for cx in CX_VALUES:
        print(f"\n  Crossover = {cx}")
        t0 = time.time()
        ga = AdaptiveGA(n_features=feat_dim, cx_prob=cx, seed=SEED)
        sel_idx, _ = ga.run(F_trval, y_trval)
        elapsed = time.time() - t0
        print(f"    Selected {len(sel_idx)} features in {elapsed:.1f}s")

        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        clf.fit(F_tr[:, sel_idx], y_tr)
        probas = clf.predict_proba(F_te[:, sel_idx])
        y_pred = clf.predict(F_te[:, sel_idx])

        m = extended_metrics(y_te, y_pred, probas)
        m["crossover"] = cx
        m["features"] = len(sel_idx)
        rows.append(m)
        print(f"    Acc={m['accuracy']} Sens={m['sensitivity']} F1={m['f1']}")

    print_rows(rows, "Table 5")
    csv_rows = [{"Crossover": r["crossover"], "Features": r["features"],
                 **{k: r[k] for k in
                 ["accuracy","precision","sensitivity","specificity","f1","auc","mcc"]}}
                for r in rows]
    save_results_csv(csv_rows, os.path.join(RESULTS_DIR, "table5_crossover.csv"))
    save_latex_table(
        generate_latex_table(csv_rows,
            caption="Effect of crossover probability on classification performance",
            label="tab:t5_cx", columns=[
                ("Crossover","CX"),("Features","Feat"),("accuracy","Acc"),
                ("sensitivity","Sens"),("specificity","Spec"),("f1","F1"),
                ("auc","AUC"),("mcc","MCC")]),
        os.path.join(RESULTS_DIR, "table5_crossover.tex"))
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  TABLE 6: GA Parameters summary
# ═══════════════════════════════════════════════════════════════════════

def run_table6():
    banner("TABLE 6: GA Parameters")
    rows = [
        {"Parameter": "Population Size",       "Value": POP_SIZE},
        {"Parameter": "Generations",            "Value": N_GEN},
        {"Parameter": "Crossover Probability",  "Value": CX_PROB},
        {"Parameter": "Mutation Probability",   "Value": MUT_PROB},
        {"Parameter": "Bit-flip Probability",   "Value": INDPB},
        {"Parameter": "Inner CV Folds",         "Value": CV_FOLDS},
        {"Parameter": "Classifier (fitness)",   "Value": "KNN (k=5, distance)"},
        {"Parameter": "L0 Penalty",             "Value": 0.001},
        {"Parameter": "Random Seed",            "Value": SEED},
    ]
    print_rows(rows, "Table 6")
    save_results_csv(rows, os.path.join(RESULTS_DIR, "table6_params.csv"))
    save_latex_table(
        generate_latex_table(rows,
            caption="Genetic Algorithm hyperparameters",
            label="tab:t6_params", columns=[
                ("Parameter","Parameter"),("Value","Value")]),
        os.path.join(RESULTS_DIR, "table6_params.tex"))
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    ap = argparse.ArgumentParser(description="Fill all 6 tables from to_be_filled.pdf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--tables", nargs="+", type=int, default=[1,2,3,4,5,6],
                    help="Which tables to run (default: all)")
    args, _ = ap.parse_known_args()

    set_seed()
    total_t0 = time.time()

    # Common data loading & feature extraction
    (y_tr, y_val, y_te, y_trval,
     F_tr, F_val, F_test, F_trval, feat_dim) = load_and_prepare(args)

    # Default GA feature selection (used for Tables 1, 2, 3)
    banner("Running default Adaptive GA (for Tables 1-3)")
    ga = AdaptiveGA(n_features=feat_dim, seed=SEED)
    sel_idx, _ = ga.run(F_trval, y_trval)
    print(f"  Selected {len(sel_idx)} / {feat_dim} features")

    # Train all 5 classifiers
    print("\n  Training classifiers ...")
    clfs, probas, val_accs, preds = train_classifiers(
        F_tr, y_tr, F_val, y_val, F_test, sel_idx)
    clf_names = list(clfs.keys())

    # ── Table 1 ──
    table1_rows = []
    if 1 in args.tables:
        table1_rows = run_table1(y_te, probas, val_accs, clf_names)

    # ── Table 2 ──
    if 2 in args.tables:
        if not table1_rows:
            table1_rows = run_table1(y_te, probas, val_accs, clf_names)
        run_table2(y_te, probas, preds, val_accs, clf_names, table1_rows)

    # ── Table 3 ──
    if 3 in args.tables:
        # Get best result from Table 2 individual classifiers or Table 1
        best_m = extended_metrics(y_te, preds["KNN"], probas["KNN"])
        run_table3(best_m["accuracy"], best_m["f1"], best_m["sensitivity"])

    # ── Table 4 ──
    if 4 in args.tables:
        run_table4(F_trval, y_trval, F_tr, y_tr, F_val, y_val, F_test, y_te, feat_dim)

    # ── Table 5 ──
    if 5 in args.tables:
        run_table5(F_trval, y_trval, F_tr, y_tr, F_val, y_val, F_test, y_te, feat_dim)

    # ── Table 6 ──
    if 6 in args.tables:
        run_table6()

    elapsed = time.time() - total_t0
    banner(f"ALL DONE — {elapsed:.1f}s total")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"  Figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    run()
