"""
Publication-quality architecture diagram for the Adaptive GA Deep Feature
Selector for Lung Histopathological Image Classification.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colors ──
C_INPUT    = "#E8F5E9"   # light green
C_BACKBONE = "#BBDEFB"   # light blue
C_ATTN     = "#E1BEE7"   # light purple
C_GA       = "#FFF9C4"   # light yellow
C_CLF      = "#FFCCBC"   # light orange
C_ENSEMBLE = "#B2EBF2"   # light cyan
C_OUTPUT   = "#F8BBD0"   # light pink
C_ARROW    = "#37474F"
C_BORDER   = "#455A64"
C_TEXT     = "#212121"
C_HEADER   = "#FFFFFF"
C_GP       = "#80DEEA"
C_SAT      = "#FFE082"


def rounded_box(ax, x, y, w, h, text, color, fontsize=8, bold=False,
                border_color=C_BORDER, text_color=C_TEXT, linewidth=1.2,
                alpha=1.0, subtext=None, subsize=6.5):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor=border_color,
                          linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ty = y + h/2 if subtext is None else y + h*0.62
    ax.text(x + w/2, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color)
    if subtext:
        ax.text(x + w/2, y + h*0.3, subtext, ha="center", va="center",
                fontsize=subsize, color="#555555", style="italic")


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def bracket_arrow(ax, x1, y1, x_mid, y_mid, x2, y2, color=C_ARROW, lw=1.0):
    """Draw an L-shaped connector."""
    ax.plot([x1, x_mid], [y1, y1], color=color, lw=lw, solid_capstyle="round")
    ax.plot([x_mid, x_mid], [y1, y2], color=color, lw=lw, solid_capstyle="round")
    arrow(ax, x_mid, y2 + 0.02, x2, y2, color=color, lw=lw)


def section_label(ax, x, y, text, fontsize=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                      edgecolor="#1565C0", linewidth=1.0))


def create_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(18, 26))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 26)
    ax.axis("off")

    # Title
    ax.text(9, 25.5, "Adaptive GA Deep Feature Selector for Lung Histopathological Image Classification",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0D47A1")
    ax.text(9, 25.15, "End-to-End Architecture",
            ha="center", va="center", fontsize=10, color="#1565C0")

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: INPUT
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 24.5, "Stage 1: Input")

    rounded_box(ax, 5.5, 24.0, 7, 0.7,
                "LC25000 Lung Histopathology Dataset",
                C_INPUT, fontsize=10, bold=True,
                subtext="15,000 images | 3 classes (ACA, Normal, SCC) | 224 x 224 RGB",
                subsize=7.5)

    # Split boxes
    arrow(ax, 9, 24.0, 9, 23.65, lw=1.5)
    rounded_box(ax, 4.0, 23.0, 10, 0.6,
                "Stratified Split: Train 70% (10,499) | Val 10% (1,501) | Test 20% (3,000)",
                "#E8EAF6", fontsize=8, bold=False)

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 22.4, "Stage 2: Feature Extraction")

    arrow(ax, 9, 23.0, 9, 22.05, lw=1.5)

    # Backbone header
    rounded_box(ax, 2.5, 21.3, 13, 0.65,
                "CNN Backbone (ImageNet Pre-trained, Frozen Weights)",
                C_BACKBONE, fontsize=9, bold=True)

    # 4 backbones
    bw = 2.8
    bx_start = 2.85
    bx_gap = 0.27
    by = 20.4

    backbones = [
        ("DenseNet121", "(7x7x1024)"),
        ("ResNet50", "(7x7x2048)"),
        ("VGG16", "(7x7x512)"),
        ("EfficientNetB0", "(7x7x1280)"),
    ]

    for i, (name, dim) in enumerate(backbones):
        bx = bx_start + i * (bw + bx_gap)
        rounded_box(ax, bx, by, bw, 0.65, name, "#90CAF9", fontsize=8,
                    bold=True, subtext=dim, subsize=7)
        arrow(ax, bx + bw/2, 21.3, bx + bw/2, by + 0.65, lw=1.0)

    # ═══════════════════════════════════════════════════════════
    # STAGE 3: ATTENTION MECHANISM
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 19.85, "Stage 3: Attention")

    # Attention header
    rounded_box(ax, 2.5, 19.05, 13, 0.55,
                "Pluggable Attention Mechanism (Applied to Feature Maps)",
                C_ATTN, fontsize=8.5, bold=True)

    # Connect backbones to attention
    for i in range(4):
        bx = bx_start + i * (bw + bx_gap)
        arrow(ax, bx + bw/2, 20.4, bx + bw/2, 19.6, lw=0.8)

    # 7 attention mechanisms
    aw = 1.65
    ax_gap = 0.08
    ax_start = 1.4
    ay = 18.15

    attentions = [
        ("SE", "Squeeze-\nExcite"),
        ("ECA", "Efficient\nChannel"),
        ("CBAM", "Channel+\nSpatial"),
        ("Split", "ResNeSt\n4-group"),
        ("Dual", "Chan+\nPosition"),
        ("ViT", "Self-Attn\n8 heads"),
        ("Swin", "Window\n7x7, 8h"),
    ]

    for i, (name, desc) in enumerate(attentions):
        ax_pos = ax_start + i * (aw + ax_gap)
        rounded_box(ax, ax_pos, ay, aw, 0.7,
                    name, "#CE93D8", fontsize=8, bold=True,
                    subtext=desc, subsize=5.5)
        arrow(ax, ax_pos + aw/2, 19.05, ax_pos + aw/2, ay + 0.7, lw=0.7)

    # GAP
    arrow(ax, 9, 18.15, 9, 17.75, lw=1.5)
    rounded_box(ax, 5.5, 17.15, 7, 0.55,
                "Global Average Pooling  \u2192  Feature Vector (N, D)",
                "#C5E1A5", fontsize=8, bold=False,
                subtext="D \u2208 {512, 1024, 1280, 2048} depending on backbone")

    # ═══════════════════════════════════════════════════════════
    # STAGE 4: ADAPTIVE GA FEATURE SELECTION
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 16.55, "Stage 4: Feature Selection")

    arrow(ax, 9, 17.15, 9, 16.25, lw=1.5)

    # GA main box
    ga_y = 14.2
    ga_h = 2.0
    rounded_box(ax, 2.0, ga_y, 14, ga_h,
                "", C_GA, fontsize=1, linewidth=1.5)

    ax.text(9, ga_y + ga_h - 0.25,
            "Adaptive Genetic Algorithm with Grouping Operator",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#F57F17")

    # GA sub-components (left side)
    ga_sub_y = ga_y + 0.25
    ga_sub_h = 0.45
    ga_sub_w = 3.2

    # Population
    rounded_box(ax, 2.3, ga_sub_y + 0.95, ga_sub_w, ga_sub_h,
                "Population: 40", "#FFF59D", fontsize=7, bold=True,
                subtext="Binary masks (1=select)", subsize=5.5)

    # Fitness
    rounded_box(ax, 2.3, ga_sub_y + 0.3, ga_sub_w, ga_sub_h,
                "Fitness: 3-fold CV (KNN)", "#FFF59D", fontsize=7, bold=True,
                subtext="acc - 0.001 x (sel/total)", subsize=5.5)

    # Adaptive rates (center)
    rounded_box(ax, 5.8, ga_sub_y + 0.95, 3.5, ga_sub_h,
                "Adaptive Rates", "#FFF59D", fontsize=7, bold=True,
                subtext="CX: [0.4-0.95]  MUT: [0.01-0.3]", subsize=5.5)

    # Grouping operator (center)
    rounded_box(ax, 5.8, ga_sub_y + 0.3, 3.5, ga_sub_h,
                "Grouping Operator", "#FFF59D", fontsize=7, bold=True,
                subtext="Crossover/mutation respect groups", subsize=5.5)

    # Generations + output (right)
    rounded_box(ax, 9.6, ga_sub_y + 0.95, 3.2, ga_sub_h,
                "Generations: 50", "#FFF59D", fontsize=7, bold=True,
                subtext="Selection \u2192 Crossover \u2192 Mutation", subsize=5.5)

    rounded_box(ax, 9.6, ga_sub_y + 0.3, 3.2, ga_sub_h,
                "Output: ~521 / 1024", "#FFF59D", fontsize=7, bold=True,
                subtext="Selected feature indices", subsize=5.5)

    # Internal arrows
    arrow(ax, 5.5, ga_sub_y + 1.17, 5.8, ga_sub_y + 1.17, lw=0.6)
    arrow(ax, 9.3, ga_sub_y + 1.17, 9.6, ga_sub_y + 1.17, lw=0.6)
    arrow(ax, 5.5, ga_sub_y + 0.52, 5.8, ga_sub_y + 0.52, lw=0.6)
    arrow(ax, 9.3, ga_sub_y + 0.52, 9.6, ga_sub_y + 0.52, lw=0.6)

    # ═══════════════════════════════════════════════════════════
    # STAGE 5: MULTI-CLASSIFIER TRAINING
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 13.6, "Stage 5: Classification")

    arrow(ax, 9, 14.2, 9, 13.25, lw=1.5)

    rounded_box(ax, 3.5, 12.55, 11, 0.55,
                "Train 5 Classifiers on Selected Features (Train Set)",
                C_CLF, fontsize=8.5, bold=True)

    # 5 classifiers
    cw = 2.4
    cx_gap = 0.2
    cx_start = 2.15
    cy = 11.5

    classifiers = [
        ("KNN", "K=5, distance\nweighted"),
        ("SVM", "RBF kernel\nC=1.0, \u03b3=scale"),
        ("RF", "300 trees\nn_jobs=-1"),
        ("LR", "max_iter=1000\nL2 reg."),
        ("XGBoost", "300 trees\ndepth=6, lr=0.1"),
    ]

    for i, (name, params) in enumerate(classifiers):
        cx = cx_start + i * (cw + cx_gap)
        rounded_box(ax, cx, cy, cw, 0.7, name, "#FF8A65", fontsize=8.5,
                    bold=True, subtext=params, subsize=5.5)
        arrow(ax, cx + cw/2, 12.55, cx + cw/2, cy + 0.7, lw=0.8)

    # Probabilities output
    arrow(ax, 9, 11.5, 9, 11.15, lw=1.5)
    rounded_box(ax, 3.5, 10.55, 11, 0.55,
                "Class Probability Outputs: P_i(c) for each classifier i, class c",
                "#FFAB91", fontsize=7.5, bold=False)

    # ═══════════════════════════════════════════════════════════
    # STAGE 6: EPSILON WEIGHT COMPUTATION
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 10.0, "Stage 6: Ensemble Fusion")

    arrow(ax, 9, 10.55, 9, 9.7, lw=1.5)

    rounded_box(ax, 3.0, 9.1, 12, 0.55,
                "Epsilon Weight Computation (from Validation Accuracy)",
                "#B2EBF2", fontsize=8, bold=True,
                subtext="\u03b5: T[j] = T[j-1] \u00d7 acc[j-1], normalized  |  Rank classifiers descending",
                subsize=6.5)

    # ═══════════════════════════════════════════════════════════
    # STAGE 6b: ENSEMBLE METHODS
    # ═══════════════════════════════════════════════════════════
    arrow(ax, 6, 9.1, 6, 8.65, lw=1.0)
    arrow(ax, 12, 9.1, 12, 8.65, lw=1.0)

    # Left branch: Majority Vote + Weighted Fusion (all 5)
    rounded_box(ax, 2.0, 7.9, 5.5, 0.65,
                "All 5 Classifiers", "#80DEEA", fontsize=8, bold=True)

    mv_y = 7.0
    rounded_box(ax, 2.0, mv_y, 2.5, 0.65,
                "Majority Vote", "#4DD0E1", fontsize=7.5, bold=True,
                subtext="mode(predictions)", subsize=5.5)

    rounded_box(ax, 4.8, mv_y, 2.7, 0.65,
                "Weighted Fusion", "#4DD0E1", fontsize=7.5, bold=True,
                subtext="\u03a3(w\u2c7cP) / (1+\u03a0(w\u2c7cP))", subsize=5.5)

    arrow(ax, 3.25, 7.9, 3.25, mv_y + 0.65, lw=0.8)
    arrow(ax, 6.15, 7.9, 6.15, mv_y + 0.65, lw=0.8)

    # Right branch: GP operators (top 4)
    rounded_box(ax, 8.5, 7.9, 7.5, 0.65,
                "Top-4 Classifiers (by \u03b5 weight)", "#80DEEA", fontsize=8, bold=True)

    # GP operators
    gp_w = 2.3
    gp_gap = 0.15
    gp_x_start = 8.3
    gp_y = 5.5

    gps = [
        ("GP1", "Max + Sat", "max(z_i)\nif \u2265 T \u2192 1"),
        ("GP2", "Noisy-OR", "1-\u03a0(1-z_i)\nif \u2265 T \u2192 1"),
        ("GP3", "Soft OR", "1-\u221a\u03a0(1-z_i)\nif \u2265 T \u2192 1"),
        ("GP4", "Min\u00d7Prod", "1-\u221a(min\u00b7\u03a0)\nif \u2265 T \u2192 1"),
    ]

    # Saturation threshold box
    rounded_box(ax, 8.5, 6.55, 7.5, 0.55,
                "Saturation Threshold: \u03c4 \u2208 {0.6, 0.7, 0.8, 0.9}",
                C_SAT, fontsize=8, bold=True,
                subtext="saturate(x, \u03c4) = 1 if x \u2265 \u03c4, else x", subsize=6)

    arrow(ax, 12.25, 7.9, 12.25, 7.1, lw=0.8)

    for i, (name, op, formula) in enumerate(gps):
        gx = gp_x_start + i * (gp_w + gp_gap - 0.38)
        rounded_box(ax, gx, gp_y, 1.85, 0.8,
                    f"{name}: {op}", "#26C6DA", fontsize=6.5, bold=True,
                    subtext=formula, subsize=5)
        arrow(ax, gx + 0.925, 6.55, gx + 0.925, gp_y + 0.8, lw=0.7)

    # ═══════════════════════════════════════════════════════════
    # STAGE 7: OUTPUT
    # ═══════════════════════════════════════════════════════════
    section_label(ax, 2.0, 4.95, "Stage 7: Output")

    # Converge arrows to final prediction
    arrow(ax, 3.25, 7.0, 3.25, 4.5, lw=1.0)
    arrow(ax, 6.15, 7.0, 6.15, 4.5, lw=1.0)
    for i in range(4):
        gx = gp_x_start + i * (gp_w + gp_gap - 0.38)
        arrow(ax, gx + 0.925, 5.5, gx + 0.925, 4.5, lw=0.7)

    rounded_box(ax, 2.0, 3.8, 14, 0.6,
                "Normalized Probability Distribution  \u2192  argmax  \u2192  Final Prediction",
                C_OUTPUT, fontsize=9, bold=True)

    # Metrics box
    arrow(ax, 9, 3.8, 9, 3.45, lw=1.5)
    rounded_box(ax, 2.5, 2.7, 13, 0.65,
                "Evaluation Metrics", C_OUTPUT, fontsize=9, bold=True,
                subtext="Accuracy | Precision | Sensitivity | Specificity | F1 | AUC (macro OVR) | MCC",
                subsize=7)

    # Best result callout
    rounded_box(ax, 4.5, 1.8, 9, 0.6,
                "Best: GP3 (Soft OR + Saturation) at \u03c4 = 0.9  \u2192  Accuracy = 99.70%",
                "#C8E6C9", fontsize=9, bold=True, border_color="#2E7D32")

    arrow(ax, 9, 2.7, 9, 2.4, lw=1.2)

    # ═══════════════════════════════════════════════════════════
    # Legend
    # ═══════════════════════════════════════════════════════════
    leg_y = 0.8
    leg_items = [
        (C_INPUT, "Input Data"),
        (C_BACKBONE, "CNN Backbone"),
        (C_ATTN, "Attention"),
        (C_GA, "Feature Selection"),
        (C_CLF, "Classifiers"),
        (C_ENSEMBLE, "Ensemble"),
        (C_OUTPUT, "Output"),
    ]
    leg_x_start = 2.0
    for i, (color, label) in enumerate(leg_items):
        lx = leg_x_start + i * 2.1
        box = FancyBboxPatch((lx, leg_y), 0.35, 0.25, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor=C_BORDER, linewidth=0.8)
        ax.add_patch(box)
        ax.text(lx + 0.5, leg_y + 0.12, label, ha="left", va="center",
                fontsize=6.5, color=C_TEXT)

    fig.tight_layout(pad=0.5)
    out_path = "/Users/anshul/Documents/lung_cancer/architecture_diagram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Architecture saved: {out_path}")

    # Also save PDF version
    out_pdf = "/Users/anshul/Documents/lung_cancer/architecture_diagram.pdf"
    fig2, ax2 = plt.subplots(1, 1, figsize=(18, 26))
    ax2.set_xlim(0, 18)
    ax2.set_ylim(0, 26)
    ax2.axis("off")
    # Re-render for PDF
    plt.close(fig2)

    return out_path


if __name__ == "__main__":
    create_architecture()
