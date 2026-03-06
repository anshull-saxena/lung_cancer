"""
Classifier setup, training, evaluation, and ensemble voting.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from scipy.stats import mode as scipy_mode

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from config import (KNN_K, KNN_WEIGHTS, SVM_KERNEL, SVM_C, SVM_GAMMA,
                     RF_N_ESTIMATORS, SEED, CLASS_NAMES, NUM_CLASSES)


def get_classifiers():
    """Return a dict of the original 3 classifiers (for backward compat)."""
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1
        ),
        "SVM": SVC(
            kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
            probability=True, random_state=SEED
        ),
        "RF": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=SEED, n_jobs=-1
        ),
    }


def get_all_classifiers():
    """Return all 5 classifiers: KNN, SVM, RF, LR, XGBoost."""
    clfs = {
        "KNN": KNeighborsClassifier(
            n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1
        ),
        "SVM": SVC(
            kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
            probability=True, random_state=SEED
        ),
        "RF": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=SEED, n_jobs=-1
        ),
        "LR": LogisticRegression(
            max_iter=1000, random_state=SEED, n_jobs=-1
        ),
    }
    if _HAS_XGB:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=SEED, n_jobs=-1
        )
    else:
        import warnings
        warnings.warn("xgboost not installed; XGBoost classifier unavailable.")
    return clfs


def compute_specificity(y_true, y_pred, num_classes=None):
    """Compute per-class specificity from confusion matrix."""
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    spec = []
    for i in range(num_classes):
        # True negatives for class i
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec.append(specificity)
    return spec


def train_and_evaluate(clf, X_train, y_train, X_test, y_test,
                       class_names=CLASS_NAMES):
    """
    Train a classifier and compute comprehensive metrics.

    Returns:
        dict with keys: accuracy, precision, recall, f1, specificity,
        per_class_{precision,recall,f1,specificity}, confusion_matrix
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)
    spec_per = compute_specificity(y_test, y_pred)
    spec_macro = np.mean(spec_per)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec_macro),
        "recall": float(rec_macro),
        "f1": float(f1_macro),
        "specificity": float(spec_macro),
        "per_class_precision": [float(v) for v in prec_per],
        "per_class_recall": [float(v) for v in rec_per],
        "per_class_f1": [float(v) for v in f1_per],
        "per_class_specificity": [float(v) for v in spec_per],
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred.tolist(),
    }


def majority_vote_ensemble(predictions_list, y_test, class_names=CLASS_NAMES):
    """
    Majority voting ensemble from a list of prediction arrays.

    Args:
        predictions_list: list of np.arrays, each (N,)
        y_test: ground truth

    Returns:
        metrics dict (same format as train_and_evaluate)
    """
    stacked = np.stack(predictions_list, axis=0)
    result = scipy_mode(stacked, axis=0, keepdims=False)
    ensemble_pred = np.asarray(result.mode).flatten()

    acc = accuracy_score(y_test, ensemble_pred)
    prec_macro = precision_score(y_test, ensemble_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, ensemble_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, ensemble_pred, average="macro", zero_division=0)

    prec_per = precision_score(y_test, ensemble_pred, average=None, zero_division=0)
    rec_per = recall_score(y_test, ensemble_pred, average=None, zero_division=0)
    f1_per = f1_score(y_test, ensemble_pred, average=None, zero_division=0)
    spec_per = compute_specificity(y_test, ensemble_pred)
    spec_macro = np.mean(spec_per)

    cm = confusion_matrix(y_test, ensemble_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec_macro),
        "recall": float(rec_macro),
        "f1": float(f1_macro),
        "specificity": float(spec_macro),
        "per_class_precision": [float(v) for v in prec_per],
        "per_class_recall": [float(v) for v in rec_per],
        "per_class_f1": [float(v) for v in f1_per],
        "per_class_specificity": [float(v) for v in spec_per],
        "confusion_matrix": cm.tolist(),
        "y_pred": ensemble_pred.tolist(),
    }
