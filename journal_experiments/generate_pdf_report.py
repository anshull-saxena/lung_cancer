"""
Generate a single PDF report with all 6 tables and ROC plots.
Uses matplotlib for table rendering and image embedding.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
OUTPUT_PDF = os.path.join(BASE_DIR, "experiment_results.pdf")


def read_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def make_table_page(pdf, title, headers, rows, col_widths=None, highlight_best=None):
    """Render a formatted table on a PDF page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))  # landscape-ish
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    n_cols = len(headers)
    n_rows = len(rows)

    cell_text = []
    for row in rows:
        cell_text.append([str(row.get(h, "")) for h in headers])

    # Display headers
    display_headers = []
    header_map = {
        "accuracy": "Acc", "precision": "Prec", "sensitivity": "Sens",
        "specificity": "Spec", "f1": "F1", "auc": "AUC", "mcc": "MCC",
        "Method": "Method", "Accuracy": "Accuracy", "F1": "F1",
        "Sensitivity": "Sensitivity", "Generations": "Gens",
        "Features": "Feat", "Crossover": "CX", "Parameter": "Parameter",
        "Value": "Value", "tau": "τ",
    }
    for h in headers:
        display_headers.append(header_map.get(h, h))

    if col_widths is None:
        col_widths = [0.22 if i == 0 else (0.78 / (n_cols - 1)) for i in range(n_cols)]

    table = ax.table(
        cellText=cell_text,
        colLabels=display_headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("#ffffff")

    # Highlight best row if specified
    if highlight_best is not None:
        for j in range(n_cols):
            cell = table[highlight_best + 1, j]
            cell.set_facecolor("#d5f5e3")
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_image_page(pdf, image_path, title):
    """Embed an image (ROC plot) as a full page."""
    if not os.path.exists(image_path):
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def find_best_row(rows, key="accuracy"):
    """Find index of row with highest value for key."""
    best_idx = 0
    best_val = 0
    for i, r in enumerate(rows):
        val = float(r.get(key, 0))
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def main():
    with PdfPages(OUTPUT_PDF) as pdf:
        # ── Title page ──
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.65,
                "Adaptive GA Deep Feature Selector\nfor Lung Histopathological Image Classification",
                ha="center", va="center", fontsize=20, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.45,
                "Experiment Results Report",
                ha="center", va="center", fontsize=16, color="#2c3e50",
                transform=ax.transAxes)
        ax.text(0.5, 0.30,
                "DenseNet121 + SE Attention + Adaptive GA + GP Ensemble Fusion\n"
                "Dataset: LC25000 (15,000 images, 3 classes)",
                ha="center", va="center", fontsize=12, color="#555555",
                transform=ax.transAxes)
        ax.text(0.5, 0.15,
                "Anshul Saxena, Advik Kashi Vishwanath, Kritii Gupta, Chandrima Debnath\n"
                "Supervisor: Prof. Swati Hait, BITS Pilani Hyderabad",
                ha="center", va="center", fontsize=11, color="#777777",
                transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 1: τ sensitivity (4 sub-tables) ──
        metrics_cols = ["Method", "accuracy", "precision", "sensitivity",
                        "specificity", "f1", "auc", "mcc"]
        col_w = [0.20] + [0.80 / 7] * 7

        for tau in ["0.6", "0.7", "0.8", "0.9"]:
            rows, _ = read_csv(f"table1_tau{tau}.csv")
            best = find_best_row(rows, "accuracy")
            make_table_page(pdf,
                            f"Table 1: Effect of Saturation Threshold τ = {tau} on Grouping Operators",
                            metrics_cols, rows, col_widths=col_w, highlight_best=best)

            # ROC plot for this τ
            roc_path = os.path.join(FIGURES_DIR, f"roc_tau_{tau}.png")
            make_image_page(pdf, roc_path, f"ROC Curves — τ = {tau}")

        # ── Table 1 combined summary ──
        combined_rows, _ = read_csv("table1_combined.csv")
        combined_cols = ["tau", "Method", "accuracy", "precision", "sensitivity",
                         "specificity", "f1", "auc", "mcc"]
        col_w_comb = [0.06, 0.18] + [0.76 / 7] * 7
        best_comb = find_best_row(combined_rows, "accuracy")
        make_table_page(pdf,
                        "Table 1 (Combined): All τ Values — Grouping Operator Performance",
                        combined_cols, combined_rows, col_widths=col_w_comb,
                        highlight_best=best_comb)

        # ── Table 2: Best grouping vs classifiers ──
        rows2, _ = read_csv("table2_best_vs_clf.csv")
        make_table_page(pdf,
                        "Table 2: Best Grouping Operator vs Individual Classifiers",
                        metrics_cols, rows2, col_widths=col_w, highlight_best=0)

        # ── Table 3: SOTA ──
        rows3, _ = read_csv("table3_sota.csv")
        sota_cols = ["Method", "Accuracy", "F1", "Sensitivity"]
        col_w3 = [0.35, 0.22, 0.22, 0.21]
        best3 = find_best_row(rows3, "Accuracy")
        make_table_page(pdf, "Table 3: Comparison with State-of-the-Art Methods on LC25000",
                        sota_cols, rows3, col_widths=col_w3, highlight_best=best3)

        # ── Table 4: Generations ──
        rows4, _ = read_csv("table4_generations.csv")
        gen_cols = ["Generations", "Features", "accuracy", "precision", "sensitivity",
                    "specificity", "f1", "auc", "mcc"]
        col_w4 = [0.08, 0.07] + [0.85 / 7] * 7
        best4 = find_best_row(rows4, "accuracy")
        make_table_page(pdf, "Table 4: Effect of Number of GA Generations",
                        gen_cols, rows4, col_widths=col_w4, highlight_best=best4)

        # ── Table 5: Crossover ──
        rows5, _ = read_csv("table5_crossover.csv")
        cx_cols = ["Crossover", "Features", "accuracy", "precision", "sensitivity",
                   "specificity", "f1", "auc", "mcc"]
        col_w5 = [0.08, 0.07] + [0.85 / 7] * 7
        best5 = find_best_row(rows5, "accuracy")
        make_table_page(pdf, "Table 5: Effect of Crossover Probability",
                        cx_cols, rows5, col_widths=col_w5, highlight_best=best5)

        # ── Table 6: GA Parameters ──
        rows6, _ = read_csv("table6_params.csv")
        param_cols = ["Parameter", "Value"]
        col_w6 = [0.50, 0.50]
        make_table_page(pdf, "Table 6: Genetic Algorithm Hyperparameters",
                        param_cols, rows6, col_widths=col_w6)

    print(f"PDF saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
