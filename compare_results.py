import re
import os

def parse_results(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    # Extract classifier accuracies (existing ones)
    acc_pattern = re.compile(r"KNN:\s*([0-9.]+).*?SVM:\s*([0-9.]+).*?RF\s*:\s*([0-9.]+)", re.DOTALL | re.IGNORECASE)
    ensemble_pattern = re.compile(r"Ensemble.*?([0-9.]+)", re.DOTALL | re.IGNORECASE)
    features_pattern = re.compile(r"Features selected by GA: ([0-9]+) / ([0-9]+) \((.*?)%\)", re.IGNORECASE)

    # New patterns for Logistic Regression and XGBoost/XGB
    logistic_pattern = re.compile(r"Logistic(?:\s+Reg)?:\s*([0-9.]+)", re.IGNORECASE)
    xgb_pattern = re.compile(r"(?:XGBoost|XGB):\s*([0-9.]+)", re.IGNORECASE)
    
    # Pattern for Extra Info and Summary
    extra_info_pattern = re.compile(r"Extra Info:\s*(.*)", re.IGNORECASE)
    summary_pattern = re.compile(r"Summary:\s*(.*)", re.IGNORECASE)

    acc_match = acc_pattern.search(content)
    ensemble_match = ensemble_pattern.search(content)
    features_match = features_pattern.search(content)
    logistic_match = logistic_pattern.search(content)
    xgb_match = xgb_pattern.search(content)
    extra_info_match = extra_info_pattern.search(content)
    summary_match = summary_pattern.search(content)

    results = {}
    if acc_match:
        results['KNN'] = float(acc_match.group(1))
        results['SVM'] = float(acc_match.group(2))
        results['RF'] = float(acc_match.group(3))
    if logistic_match:
        results['Logistic'] = float(logistic_match.group(1))
    if xgb_match:
        results['XGBoost'] = float(xgb_match.group(1))
    if ensemble_match:
        results['Ensemble'] = float(ensemble_match.group(1))
    if features_match:
        results['GA_selected'] = int(features_match.group(1))
        results['GA_total'] = int(features_match.group(2))
        results['GA_percent'] = float(features_match.group(3))
    if extra_info_match:
        results['Extra_Info'] = extra_info_match.group(1).strip()
    if summary_match:
        results['Summary'] = summary_match.group(1).strip()
    return results

def _ensure_package(pkg):
    try:
        __import__(pkg)
    except ImportError:
        import subprocess, sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip3", "install", pkg])

def save_table_image(all_results, file_paths):
    # ensure dependencies
    _ensure_package('pandas')
    _ensure_package('matplotlib')
    from PIL import Image
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_results)
    # Keep File as first column (not index) for visibility in the table
    # Determine figure size based on table shape
    rows, cols = df.shape
    fig_width = max(12, cols * 2.5)
    fig_height = max(3 + rows * 0.6, 3.5)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    # Adjust font size based on table size
    table.set_fontsize(11 if rows < 15 else max(7, int(200/rows)))
    table.scale(1, 1.8)
    
    # Style the header row
    for i in range(cols):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Style the data rows with alternating colors
    for i in range(1, rows + 1):
        for j in range(cols):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E9F5')
            else:
                cell.set_facecolor('#F5F5F5')
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
            # Bold the first column (File names)
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=11)

    plt.title("Model Performance Comparison", fontsize=16, weight='bold', pad=20, color='#2C3E50')

    # Construct output name from combined file basenames.
    basenames = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
    if len(basenames) >= 2:
        out_name = f"{basenames[0]}_{basenames[1]}_comparison.jpg"
    elif len(basenames) == 1:
        out_name = f"{basenames[0]}_comparison.jpg"
    else:
        out_name = "comparison.jpg"

    out_path = os.path.join(os.getcwd(), out_name)
    # Save high-res to ensure good quality jpg
    fig.savefig(out_path, bbox_inches='tight', dpi=300, format='jpg')
    plt.close(fig)

    # Confirm image creation
    print(f"Saved table image to: {out_path}")
    return out_path

def extract_file_label(path):
    """Extract simplified label from file path - everything after 'code' excluding underscores"""
    basename = os.path.splitext(os.path.basename(path))[0]
    # Find 'code' in the filename (case insensitive)
    lower_basename = basename.lower()
    if 'code' in lower_basename:
        idx = lower_basename.index('code') + 4
        label = basename[idx:].replace('_', ' ').strip()
        return label if label else basename
    return basename

def compare_files(file_paths):
    # ensure tabulate for text output
    _ensure_package('tabulate')
    from tabulate import tabulate

    all_results = []
    for path in file_paths:
        res = parse_results(path)
        all_results.append({
            'File': extract_file_label(path),
            'KNN': f"{res.get('KNN'):.4f}" if 'KNN' in res else '-',
            'SVM': f"{res.get('SVM'):.4f}" if 'SVM' in res else '-',
            'RF': f"{res.get('RF'):.4f}" if 'RF' in res else '-',
            'Logistic': f"{res.get('Logistic'):.4f}" if 'Logistic' in res else '-',
            'XGBoost': f"{res.get('XGBoost'):.4f}" if 'XGBoost' in res else '-',
            'Ensemble': f"{res.get('Ensemble'):.4f}" if 'Ensemble' in res else '-',
            'GA_Selected(%)': f"{res.get('GA_selected', '-')}/{res.get('GA_total', '-')} ({res.get('GA_percent', '-')})" if 'GA_selected' in res else '-',
            'Extra Info': res.get('Extra_Info', '-'),
            'Summary': res.get('Summary', '-')
        })

    print("\nComparison of Results:")
    print(tabulate(all_results, headers="keys", tablefmt="grid"))

    def ensemble_value(row):
        try:
            return float(row['Ensemble'])
        except Exception:
            return -1.0

    # Handle case with no ensembles gracefully
    try:
        best_ensemble = max(all_results, key=ensemble_value)
        print(f"\nBest Ensemble Accuracy: {best_ensemble['Ensemble']} in {best_ensemble['File']}")
    except ValueError:
        print("\nNo results to evaluate best ensemble.")

    # Create and save table image
    try:
        save_table_image(all_results, file_paths)
    except Exception as e:
        print(f"Failed to create/save table image: {e}")

def main():
    n = int(input("Enter number of files to compare: "))
    file_paths = []
    for i in range(n):
        path = input(f"Enter path for file {i+1}: ").strip().strip("'\"")
        file_paths.append(path)
    compare_files(file_paths)

if __name__ == "__main__":
    main()
