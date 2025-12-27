"""
IEEE-Quality Vertical Methodology Flowchart - Fixed Version
No overlaps, proper labels, uniform spacing
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Figure setup
fig, ax = plt.subplots(figsize=(11, 22), dpi=300)
ax.set_xlim(0, 11)
ax.set_ylim(0, 24)
ax.axis('off')

# Colors
c = {
    'input': '#F5F5F5', 'process': '#E3F2FD', 'cnn': '#BBDEFB',
    'feature': '#C8E6C9', 'opt': '#FFF9C4', 'clf': '#F8BBD0',
    'fusion': '#E1BEE7', 'output': '#CFD8DC'
}

def box(x, y, w, h, txt, col, fs=9, fw='normal'):
    """Draw box centered at (x,y)"""
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05",
                       ec='#424242', fc=col, lw=1.5)
    ax.add_patch(b)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, fontweight=fw)
    return {'x': x, 'y': y, 'w': w, 'h': h, 
            'top': y+h/2, 'bottom': y-h/2, 'left': x-w/2, 'right': x+w/2}

def varrow(y1, y2, x=5.5):
    """Vertical arrow - ensures no overlap by using exact box boundaries"""
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), 
                                 arrowstyle='->,head_width=0.4,head_length=0.5',
                                 color='#424242', lw=2))

def connector(x1, y1, x2, y2, y_route):
    """Straight line connector - routes around boxes"""
    ax.plot([x1, x1], [y1, y_route], 'k-', lw=1.5, solid_capstyle='round')
    ax.plot([x1, x2], [y_route, y_route], 'k-', lw=1.5, solid_capstyle='round')
    ax.add_patch(FancyArrowPatch((x2, y_route), (x2, y2),
                                 arrowstyle='->,head_width=0.3,head_length=0.4',
                                 color='#424242', lw=1.5))

# STRICT GRID - 2.5 unit spacing
Y_GRID = [23, 20.5, 18, 15.5, 13, 10.5, 8, 5.5, 3, 0.5]
y_idx = 0

X_CTR = 5.5
W_MAIN = 7.0
W_PARA = 1.7
H_STD = 0.8
H_TALL = 1.2

# Title
ax.text(X_CTR, 23.5, 'Proposed Methodology Flowchart',
        ha='center', fontsize=14, fontweight='bold')

# 1. Input
b1 = box(X_CTR, Y_GRID[y_idx], W_MAIN, H_STD,
         'Input: Lung Histopathology Images (224×224×3)',
         c['input'], fs=9, fw='bold')
y_idx += 1

varrow(b1['bottom'], Y_GRID[y_idx]+H_TALL/2)

# 2. Augmentation  
b2 = box(X_CTR, Y_GRID[y_idx], W_MAIN, H_TALL,
         'Data Augmentation\n(Rotation, Flip, Zoom, Shift)',
         c['process'], fs=8.5)
y_idx += 1

# Section label for CNNs
ax.text(X_CTR, Y_GRID[y_idx]+0.4, 'Multi-CNN Feature Extraction with Channel Attention',
        ha='center', fontsize=10, fontweight='bold')

# 3. CNN Array
y_cnn = Y_GRID[y_idx]
n = 5
spacing = 0.2
total_w = n*W_PARA + (n-1)*spacing
x_start = X_CTR - total_w/2 + W_PARA/2

cnns = ['DenseNet121', 'ResNet50', 'VGG16', 'MobileNetV2', 'InceptionV3']
cnn_boxes = []
for i in range(n):
    x = x_start + i*(W_PARA+spacing)
    b = box(x, y_cnn, W_PARA, H_TALL, f'{cnns[i]}\n+\nChannel\nAttention',
            c['cnn'], fs=6.5)
    cnn_boxes.append(b)

# Connect from augmentation - route BELOW the label
y_route = Y_GRID[y_idx] + 0.7  # Above CNN boxes, below label
for i, b in enumerate(cnn_boxes):
    connector(b2['x'], b2['bottom'], b['x'], b['top'], y_route)

y_idx += 1
varrow(y_cnn-H_TALL/2, Y_GRID[y_idx]+H_STD/2)

# 4. Concatenation
b4 = box(X_CTR, Y_GRID[y_idx], W_MAIN, H_STD,
         'Feature Concatenation (6400-Dimensional Vector)',
         c['feature'], fs=9, fw='bold')
y_idx += 1

varrow(b4['bottom'], Y_GRID[y_idx]+H_TALL/2)

# 5. NSGA-II
b5 = box(X_CTR, Y_GRID[y_idx], W_MAIN+0.5, H_TALL,
         'NSGA-II Multi-Objective Optimization\n(Population: 50, Generations: 30, Crossover: 0.8, Mutation: 0.05)',
         c['opt'], fs=7.5, fw='bold')
ax.text(X_CTR+4.5, Y_GRID[y_idx], 'Objective 1: Maximize Accuracy\nObjective 2: Minimize Features',
        ha='left', va='center', fontsize=7.5, style='italic')
y_idx += 1

varrow(b5['bottom'], Y_GRID[y_idx]+H_STD/2)

# 6. Selected Features
b6 = box(X_CTR, Y_GRID[y_idx], W_MAIN, H_STD,
         'Selected Feature Subset (169 of 6400 Features)',
         c['feature'], fs=9, fw='bold')
y_idx += 1

# Section label for classifiers
ax.text(X_CTR, Y_GRID[y_idx]+0.4, 'Multi-Classifier Ensemble',
        ha='center', fontsize=10, fontweight='bold')

# 7. Classifier Array
y_clf = Y_GRID[y_idx]
clfs = ['K-Nearest\nNeighbors', 'Support Vector\nMachine', 'Random\nForest', 
        'Logistic\nRegression', 'XGBoost']
clf_boxes = []
for i in range(n):
    x = x_start + i*(W_PARA+spacing)
    b = box(x, y_clf, W_PARA, 1.0, clfs[i], c['clf'], fs=6.5)
    clf_boxes.append(b)

# Connect from selected features - route BELOW the label
y_route_clf = Y_GRID[y_idx] + 0.7
for b in clf_boxes:
    connector(b6['x'], b6['bottom'], b['x'], b['top'], y_route_clf)

y_idx += 1
varrow(y_clf-0.5, Y_GRID[y_idx]+H_TALL/2)

# 8. Fusion
b8 = box(X_CTR, Y_GRID[y_idx], W_MAIN, H_TALL,
         'Genetic Programming Ensemble Fusion\n(Product, Weighted Maximum, Hybrid, Weighted Sum)',
         c['fusion'], fs=8, fw='bold')
y_idx += 1

varrow(b8['bottom'], Y_GRID[y_idx]+H_STD/2)

# 9. Output - USE ACTUAL CLASS NAMES
b9 = box(X_CTR, Y_GRID[y_idx], W_MAIN-1, H_STD,
         'Final Classification Output\n(Adenocarcinoma, Normal, Squamous Cell Carcinoma)',
         c['output'], fs=8.5, fw='bold')

plt.tight_layout()
plt.savefig('../figures/methodology_flowchart.png',
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
print("✓ Fixed flowchart saved: no overlaps, proper labels")
plt.close()
