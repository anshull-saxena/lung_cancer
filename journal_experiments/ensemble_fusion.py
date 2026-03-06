"""
Ensemble fusion operators: GP1-GP4, weighted probability fusion, and epsilon weights.
Based on: "A grouping-based ensemble methods with NSGA-II" methodology.
"""
import numpy as np
from sklearn.metrics import accuracy_score

from classifiers import compute_specificity
from config import CLASS_NAMES


# ============================================================================
# WEIGHT CALCULATION
# ============================================================================

def compute_epsilon_weights(accuracies):
    """
    Compute cumulative performance weights from classifier accuracies.

    Classifiers are ranked by accuracy (descending). Weights decay
    multiplicatively so that higher-ranked classifiers receive more weight.

    Args:
        accuracies: array-like of classifier validation accuracies

    Returns:
        epsilon: np.array of normalized weights (same order as ranked)
        ranked_indices: np.array — indices into the original list, sorted desc
    """
    accuracies = np.asarray(accuracies, dtype=float)
    ranked_indices = np.argsort(accuracies)[::-1]
    ranked_accs = accuracies[ranked_indices]

    T = [1.0]
    for j in range(1, len(ranked_accs)):
        T.append(T[-1] * ranked_accs[j - 1])

    T = np.array(T)
    epsilon = T / T.sum()
    return epsilon, ranked_indices


# ============================================================================
# GP FUSION OPERATORS
# ============================================================================

def gp1_fusion(z1, z2, z3, z4):
    """GP1: Maximum operator — 1 - min(1-z1, 1-z2, 1-z3, 1-z4) = max(z)."""
    return 1.0 - np.minimum.reduce([1 - z1, 1 - z2, 1 - z3, 1 - z4])


def gp2_fusion(z1, z2, z3, z4):
    """GP2: Algebraic product — 1 - (1-z1)(1-z2)(1-z3)(1-z4)."""
    return 1.0 - (1 - z1) * (1 - z2) * (1 - z3) * (1 - z4)


def gp3_fusion(z1, z2, z3, z4):
    """GP3: Ratio-based — 1 - (1 + z1*z2*z3*z4) / ((1-z1)(1-z2)(1-z3)(1-z4))."""
    numerator = 1.0 + z1 * z2 * z3 * z4
    denominator = (1 - z1) * (1 - z2) * (1 - z3) * (1 - z4)
    denominator = np.maximum(denominator, 1e-10)
    return 1.0 - numerator / denominator


def gp4_fusion(z1, z2, z3, z4):
    """GP4: Weighted sum — (z1 + z2 + z3 + z4) / (1 + z1*z2*z3*z4)."""
    numerator = z1 + z2 + z3 + z4
    denominator = 1.0 + z1 * z2 * z3 * z4
    return numerator / denominator


# ============================================================================
# WEIGHTED PROBABILITY FUSION (uses all 5 classifiers)
# ============================================================================

def weighted_probability_fusion(probas_list, weights):
    """
    Weighted fusion with product dampening (uses all classifiers).

    Formula per class c:
        P_ens(c) = sum(w_i * P_i(c)) / (1 + prod(w_i * P_i(c)))

    Args:
        probas_list: list of K arrays, each (n_samples, n_classes)
        weights: array of K floats (epsilon values, same order as probas_list)

    Returns:
        ensemble_proba: (n_samples, n_classes)
    """
    n_samples, n_classes = probas_list[0].shape
    K = len(probas_list)
    ensemble_proba = np.zeros((n_samples, n_classes))

    for c in range(n_classes):
        numerator = sum(weights[i] * probas_list[i][:, c] for i in range(K))
        product = np.prod(
            [weights[i] * probas_list[i][:, c] for i in range(K)], axis=0
        )
        ensemble_proba[:, c] = numerator / (1.0 + product)

    return ensemble_proba


# ============================================================================
# GP FUSION APPLICATOR (uses top 4 classifiers)
# ============================================================================

def apply_gp_fusion(probas_list, weights, fusion_func, normalize=True):
    """
    Apply a GP fusion operator to the top-4 classifiers' weighted probabilities.

    Args:
        probas_list: list of 4 arrays (top classifiers, ranked), each (n_samples, n_classes)
        weights: array of 4 floats (epsilon weights for the top-4)
        fusion_func: one of {gp1_fusion, gp2_fusion, gp3_fusion, gp4_fusion}
        normalize: whether to normalize output to valid probability distribution

    Returns:
        fused_proba: (n_samples, n_classes)
    """
    n_samples, n_classes = probas_list[0].shape
    fused_proba = np.zeros((n_samples, n_classes))

    for c in range(n_classes):
        z1 = weights[0] * probas_list[0][:, c]
        z2 = weights[1] * probas_list[1][:, c]
        z3 = weights[2] * probas_list[2][:, c]
        z4 = weights[3] * probas_list[3][:, c]
        fused_proba[:, c] = fusion_func(z1, z2, z3, z4)

    if normalize:
        row_sums = fused_proba.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        fused_proba = fused_proba / row_sums

    return fused_proba


# ============================================================================
# HIGH-LEVEL: run all ensemble methods and return comparison metrics
# ============================================================================

def evaluate_all_ensemble_methods(clf_names, probas_dict, val_acc_dict,
                                  y_test, class_names=CLASS_NAMES):
    """
    Run all 6 ensemble strategies and return a list of result dicts.

    Strategies: Majority Vote, Weighted Fusion, GP1, GP2, GP3, GP4.

    Args:
        clf_names: list of classifier names in the order they were trained
        probas_dict: dict {clf_name: (n_samples, n_classes) probability array}
        val_acc_dict: dict {clf_name: float validation accuracy}
        y_test: ground truth labels (n_samples,)

    Returns:
        list of dicts, each with method, accuracy, precision, recall, f1,
        specificity, y_pred
    """
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 confusion_matrix)
    from scipy.stats import mode as scipy_mode

    names = list(clf_names)
    accs = np.array([val_acc_dict[n] for n in names])
    epsilon, ranked_idx = compute_epsilon_weights(accs)
    ranked_names = [names[i] for i in ranked_idx]

    # Ordered probability arrays (ranked by val accuracy, descending)
    ranked_probas = [probas_dict[n] for n in ranked_names]

    results = []

    def _metrics_from_pred(y_pred, method_name):
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        spec_per = compute_specificity(y_test, y_pred)
        spec = np.mean(spec_per)
        cm = confusion_matrix(y_test, y_pred)
        return {
            "method": method_name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "specificity": float(spec),
            "per_class_specificity": [float(v) for v in spec_per],
            "confusion_matrix": cm.tolist(),
            "y_pred": y_pred.tolist(),
        }

    def _metrics_from_proba(proba, method_name):
        y_pred = np.argmax(proba, axis=1)
        return _metrics_from_pred(y_pred, method_name)

    # 1. Majority Vote
    all_preds = [np.argmax(probas_dict[n], axis=1) for n in names]
    stacked = np.stack(all_preds, axis=0)
    mv_result = scipy_mode(stacked, axis=0, keepdims=False)
    mv_pred = np.asarray(mv_result.mode).flatten()
    results.append(_metrics_from_pred(mv_pred, "Majority Vote"))

    # 2. Weighted Probability Fusion (all classifiers)
    wpf_proba = weighted_probability_fusion(ranked_probas, epsilon)
    results.append(_metrics_from_proba(wpf_proba, "Weighted Fusion"))

    # 3-6. GP1 through GP4 (top 4 classifiers)
    top4_probas = ranked_probas[:4]
    top4_eps = epsilon[:4]
    # Re-normalize top4 weights so they sum to the same proportion
    top4_eps_norm = top4_eps / top4_eps.sum() * epsilon[:4].sum()

    gp_operators = [
        ("GP1 (Max)", gp1_fusion),
        ("GP2 (Product)", gp2_fusion),
        ("GP3 (Ratio)", gp3_fusion),
        ("GP4 (Weighted Sum)", gp4_fusion),
    ]
    for gp_name, gp_func in gp_operators:
        gp_proba = apply_gp_fusion(top4_probas, top4_eps, gp_func, normalize=True)
        results.append(_metrics_from_proba(gp_proba, gp_name))

    # Print epsilon weights for reference
    print(f"  Epsilon weights (ranked): {dict(zip(ranked_names, epsilon.round(4)))}")

    return results
