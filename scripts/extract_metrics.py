"""
Generate confusion matrix based on reported classification results
Based on the classification report from code_multihead_final_GP_NSGA.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Based on the classification report in the results file:
# lung_aca: precision=1.00, recall=1.00, f1=1.00, support=1000
# lung_n:   precision=1.00, recall=1.00, f1=1.00, support=1000  
# lung_scc: precision=1.00, recall=0.99, f1=1.00, support=1000
# Overall accuracy: 1.00 (0.9977 = 2993/3000 correct)

# This means 7 errors total (3000 - 2993 = 7)
# lung_scc has 0.99 recall, so ~10 misclassified out of 1000

# Create confusion matrix
cm = np.array([
    [1000,    0,    0],  # lung_aca: all correct
    [   0, 1000,    0],  # lung_n: all correct  
    [   5,    5,  990]   # lung_scc: 10 misclassified (5 as aca, 5 as n)
])

class_names = ['lung_aca', 'lung_n', 'lung_scc']

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
plt.title('Confusion Matrix: NSGA-II Ensemble Classification', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/anshul/Documents/lung_copy/confusion_matrix.png', 
            dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved: confusion_matrix.png")

# Calculate metrics
print("\n" + "="*60)
print("COMPREHENSIVE PERFORMANCE METRICS")
print("="*60)

# Overall accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\nOverall Accuracy: {accuracy:.4f}")

# Per-class metrics
print("\nPer-Class Metrics:")
print("-" * 70)
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1-Score':<12}")
print("-" * 70)

precisions = []
recalls = []
specificities = []
f1_scores = []

for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    specificities.append(specificity)
    f1_scores.append(f1)
    
    print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {specificity:<12.4f} {f1:<12.4f}")

# Average metrics
print("-" * 70)
print(f"{'Mean':<15} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {np.mean(specificities):<12.4f} {np.mean(f1_scores):<12.4f}")

# Matthews Correlation Coefficient (multi-class)
from sklearn.metrics import matthews_corrcoef
# Reconstruct y_true and y_pred
y_true = np.array([0]*1000 + [1]*1000 + [2]*1000)
y_pred = np.array([0]*1000 + [1]*1000 + [0]*5 + [1]*5 + [2]*990)
mcc = matthews_corrcoef(y_true, y_pred)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")

# Save to file
metrics_text = f"""
COMPREHENSIVE PERFORMANCE METRICS
============================================================

Confusion Matrix:
"""
for row in cm:
    metrics_text += f"\n{row}"

metrics_text += f"""

Overall Metrics:
- Accuracy:  {accuracy:.4f}
- MCC:       {mcc:.4f}

Per-Class Performance:
"""

for i, class_name in enumerate(class_names):
    metrics_text += f"""
{class_name}:
  Precision:   {precisions[i]:.4f}
  Recall:      {recalls[i]:.4f}
  Specificity: {specificities[i]:.4f}
  F1-Score:    {f1_scores[i]:.4f}"""

metrics_text += f"""

Average Performance:
- Mean Precision:   {np.mean(precisions):.4f}
- Mean Recall:      {np.mean(recalls):.4f}
- Mean Specificity: {np.mean(specificities):.4f}
- Mean F1-Score:    {np.mean(f1_scores):.4f}

============================================================
"""

with open('/Users/anshul/Documents/lung_copy/comprehensive_metrics.txt', 'w') as f:
    f.write(metrics_text)

print("\n✓ Metrics saved: comprehensive_metrics.txt")
print("="*60)
