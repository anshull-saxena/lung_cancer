"""
IEEE-Quality Horizontal Architecture Diagram
Fixed spacing, no overlaps, correct labels
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Figure setup - Wide Landscape
fig, ax = plt.subplots(figsize=(28, 12), dpi=300)
ax.set_xlim(0, 29)
ax.set_ylim(0, 13)
ax.axis('off')

# Colors (Consistent with vertical flowchart)
c = {
    'input': '#F5F5F5', 'process': '#E3F2FD', 'cnn': '#BBDEFB',
    'att': '#FFCCBC', 'feature': '#C8E6C9', 'opt': '#FFF9C4', 
    'clf': '#F8BBD0', 'fusion': '#E1BEE7', 'output': '#CFD8DC'
}

def box(x, y, w, h, txt, col, fs=10, fw='normal'):
    """Draw box centered at (x,y)"""
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05",
                       ec='#424242', fc=col, lw=1.5)
    ax.add_patch(b)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, fontweight=fw, wrap=True)
    return {'x': x, 'y': y, 'w': w, 'h': h, 
            'right': x+w/2, 'left': x-w/2, 'top': y+h/2, 'bottom': y-h/2}

def arrow(x1, y1, x2, y2):
    """Straight arrow with shrink to avoid overlap"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), 
                                 arrowstyle='->,head_width=0.4,head_length=0.5',
                                 color='#424242', lw=1.8, shrinkB=5))

def elbow_arrow(x1, y1, x2, y2, x_vertical):
    """Elbow connector: Horizontal -> Vertical -> Horizontal"""
    # 1. Horizontal to x_vertical
    ax.plot([x1, x_vertical], [y1, y1], 'k-', lw=1.5, solid_capstyle='round')
    # 2. Vertical to y2
    ax.plot([x_vertical, x_vertical], [y1, y2], 'k-', lw=1.5, solid_capstyle='round')
    # 3. Horizontal to x2 (with arrow)
    ax.add_patch(FancyArrowPatch((x_vertical, y2), (x2, y2),
                                 arrowstyle='->,head_width=0.3,head_length=0.4',
                                 color='#424242', lw=1.5, shrinkB=5))

def reverse_elbow_arrow(x1, y1, x2, y2, x_vertical):
    """Reverse Elbow: Horizontal -> Vertical -> Horizontal (converging)"""
    # 1. Horizontal from x1 to x_vertical
    ax.plot([x1, x_vertical], [y1, y1], 'k-', lw=1.5, solid_capstyle='round')
    # 2. Vertical from y1 to y2
    ax.plot([x_vertical, x_vertical], [y1, y2], 'k-', lw=1.5, solid_capstyle='round')
    # 3. Horizontal to x2
    ax.add_patch(FancyArrowPatch((x_vertical, y2), (x2, y2),
                                 arrowstyle='->,head_width=0.3,head_length=0.4',
                                 color='#424242', lw=1.5, shrinkB=5))

# --- Layout ---
Y_CTR = 6.5
Y_GAP = 2.2
branch_ys = [Y_CTR + (2-i)*Y_GAP for i in range(5)] # Top to bottom

# X Coordinates - Adjusted
X = {
    'input': 1.2,
    'aug': 3.8,
    'dist1': 5.5,
    'cnn': 7.2,
    'att': 9.8,
    'gap': 11.8,
    'concat': 13.5,
    'nsga': 16.0,
    'sel': 18.5,
    'dist2': 19.8,
    'clf': 21.5,
    'dist3': 24.2, 
    'fusion': 26.0
}

# Title
ax.text(13, 12.5, 'Proposed NSGA-II Multi-CNN Ensemble Framework', 
        ha='center', fontsize=22, fontweight='bold', color='#263238')

# 1. Input
b1 = box(X['input'], Y_CTR, 2.0, 2.0, 'Input Image\n(224×224)', c['input'], fw='bold', fs=10)
arrow(b1['right'], Y_CTR, X['aug']-1.0, Y_CTR)

# 2. Augmentation
b2 = box(X['aug'], Y_CTR, 2.0, 2.5, 'Data\nAugmentation\n(Flip, Rotate)', c['process'], fs=10)

# Fan-out to CNNs
for y in branch_ys:
    elbow_arrow(b2['right'], Y_CTR, X['cnn']-1.25, y, X['dist1'])

# 3. CNN Branches
cnn_names = ['DenseNet121', 'ResNet50', 'VGG16', 'MobileNetV2', 'InceptionV3']
dims = ['1280', '2048', '512', '1280', '2048']

gap_outputs = []

for i, y in enumerate(branch_ys):
    # CNN
    b_cnn = box(X['cnn'], y, 2.5, 1.2, cnn_names[i], c['cnn'], fs=9)
    
    # Attention
    b_att = box(X['att'], y, 1.4, 1.2, 'Channel\nAttention', c['att'], fs=8)
    arrow(b_cnn['right'], y, b_att['left'], y)
    
    # GAP
    b_gap = box(X['gap'], y, 1.6, 1.2, f'Global Avg\nPooling\n{dims[i]}', c['cnn'], fs=8)
    arrow(b_att['right'], y, b_gap['left'], y)
    
    gap_outputs.append((b_gap['right'], y))

# 4. Concatenation
b_concat = box(X['concat'], Y_CTR, 1.2, 10.0, '', c['feature'])
ax.text(X['concat'], Y_CTR, 'Concatenation (6400 Features)', 
        ha='center', va='center', rotation=90, fontweight='bold', fontsize=11)

for x_src, y_src in gap_outputs:
    arrow(x_src, y_src, b_concat['left'], y_src)

# 5. NSGA-II
b_nsga = box(X['nsga'], Y_CTR, 2.8, 4.0, 'NSGA-II\nFeature Selection', c['opt'], fw='bold', fs=10)
ax.text(X['nsga'], Y_CTR-2.5, 'Obj 1: Max Acc\nObj 2: Min Feat', ha='center', fontsize=9, style='italic')

arrow(b_concat['right'], Y_CTR, b_nsga['left'], Y_CTR)

# 6. Selected
b_sel = box(X['sel'], Y_CTR, 1.2, 4.0, '', c['feature'])
ax.text(X['sel'], Y_CTR, 'Selected (169)', 
        ha='center', va='center', rotation=90, fontweight='bold', fontsize=11)

arrow(b_nsga['right'], Y_CTR, b_sel['left'], Y_CTR)

# 7. Classifiers
clfs = ['K-Nearest\nNeighbors', 'Support Vector\nMachine', 'Random\nForest', 'Logistic\nRegression', 'XGBoost']
clf_outputs = []

for i, y in enumerate(branch_ys):
    elbow_arrow(b_sel['right'], Y_CTR, X['clf']-1.1, y, X['dist2'])
    b_clf = box(X['clf'], y, 2.2, 1.0, clfs[i], c['clf'], fs=9)
    clf_outputs.append((b_clf['right'], y))

# 8. Fusion
b_fusion = box(X['fusion'], Y_CTR, 2.5, 3.0, 'GP Fusion\nEnsemble', c['fusion'], fw='bold', fs=10)

for x_src, y_src in clf_outputs:
    reverse_elbow_arrow(x_src, y_src, b_fusion['left'], Y_CTR, X['dist3'])

# 9. Output (Vertical below Fusion)
b_out = box(X['fusion'], 2.0, 3.5, 2.0, 'Output Class\n\n• Adenocarcinoma\n• Normal\n• Squamous Cell', c['output'], fw='bold', fs=10)

# Connect Fusion to Output - ensuring NO overlap
ax.add_patch(FancyArrowPatch((X['fusion'], b_fusion['bottom']), (X['fusion'], b_out['top']), 
                             arrowstyle='->,head_width=0.4,head_length=0.5',
                             color='#424242', lw=1.8, shrinkB=5))

# Backgrounds
rect_feat = Rectangle((X['cnn']-1.5, branch_ys[-1]-1.0), 
                      (X['gap'] - X['cnn'] + 3.5), 11.0,
                      fill=False, ec='#90A4AE', ls='--', lw=1)
ax.add_patch(rect_feat)
ax.text(X['att'], 12.3, 'Feature Extraction Stage', ha='center', color='#546E7A')

plt.tight_layout()
plt.savefig('../figures/methodology_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Fixed horizontal architecture diagram saved")
plt.close()
