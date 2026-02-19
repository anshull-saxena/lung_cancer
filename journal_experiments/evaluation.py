"""
Metrics computation, result persistence, and table generation.
"""
import json
import csv
import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from config import RESULTS_DIR, CLASS_NAMES


def _compute_specificity(y_true, y_pred, num_classes=None):
    """Compute per-class specificity from confusion matrix."""
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    spec = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec.append(specificity)
    return spec


def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """
    Compute comprehensive classification metrics.

    Returns:
        dict with accuracy, precision, recall, f1, specificity (macro + per-class)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    spec_per = _compute_specificity(y_true, y_pred)
    spec = np.mean(spec_per)

    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "per_class_precision": {cn: float(v) for cn, v in zip(class_names, prec_per)},
        "per_class_recall": {cn: float(v) for cn, v in zip(class_names, rec_per)},
        "per_class_f1": {cn: float(v) for cn, v in zip(class_names, f1_per)},
        "per_class_specificity": {cn: float(v) for cn, v in zip(class_names, spec_per)},
        "confusion_matrix": cm.tolist(),
    }


def save_results_json(results_dict, filepath):
    """Save results dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2, default=convert)
    print(f"Results saved to {filepath}")


def save_results_csv(results_dict, filepath):
    """Save a flat results dict (or list of dicts) to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(results_dict, list):
        rows = results_dict
    else:
        rows = [results_dict]

    if not rows:
        return

    keys = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            # Convert non-serializable values to strings
            clean = {}
            for k, v in row.items():
                if isinstance(v, (list, dict, np.ndarray)):
                    clean[k] = str(v)
                elif isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            writer.writerow(clean)
    print(f"CSV saved to {filepath}")


def print_table(results, title="Results"):
    """Print a formatted table to console."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    if isinstance(results, list):
        if not results:
            return
        # Identify the label key and metric keys
        label_key = None
        for candidate in ("method", "name", "Method", "Name"):
            if candidate in results[0]:
                label_key = candidate
                break
        exclude = {"confusion_matrix", "y_pred", "per_class_precision",
                   "per_class_recall", "per_class_f1", "per_class_specificity"}
        if label_key:
            exclude.add(label_key)
        keys = [k for k in results[0].keys() if k not in exclude]
        # Header
        label_header = label_key.title() if label_key else "Item"
        header = f"{label_header:<25}"
        for k in keys:
            header += f"{k:>12}"
        print(header)
        print("-" * len(header))
        for row in results:
            label_val = row.get(label_key, "") if label_key else ""
            line = f"{str(label_val):<25}"
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    line += f"{v:>12.4f}"
                else:
                    line += f"{str(v):>12}"
            print(line)
    elif isinstance(results, dict):
        for k, v in results.items():
            if k in ("confusion_matrix", "y_pred"):
                continue
            if isinstance(v, float):
                print(f"  {k:<25}: {v:.4f}")
            else:
                print(f"  {k:<25}: {v}")
    print(f"{'='*70}\n")


def generate_latex_table(results, caption, label, columns=None):
    """
    Generate a LaTeX table string from results.

    Args:
        results: list of dicts (rows)
        caption: table caption
        label: table label
        columns: list of (key, header) tuples to include

    Returns:
        LaTeX string
    """
    if columns is None:
        # Auto-detect from first row, excluding large fields
        exclude = {"confusion_matrix", "y_pred", "per_class_precision",
                    "per_class_recall", "per_class_f1", "per_class_specificity"}
        columns = [(k, k.replace("_", " ").title())
                    for k in results[0].keys() if k not in exclude]

    n_cols = len(columns)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join(h for _, h in columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for row in results:
        cells = []
        for key, _ in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_latex_table(latex_str, filepath):
    """Save LaTeX table string to a .tex file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {filepath}")
