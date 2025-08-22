import re

def parse_results(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    # Extract classifier accuracies
    acc_pattern = re.compile(r"KNN:\s+([0-9.]+).*?SVM:\s+([0-9.]+).*?RF\s*:\s*([0-9.]+)", re.DOTALL)
    ensemble_pattern = re.compile(r"Ensemble.*?([0-9.]+)", re.DOTALL)
    features_pattern = re.compile(r"Features selected by GA: ([0-9]+) / ([0-9]+) \((.*?)%\)")
    acc_match = acc_pattern.search(content)
    ensemble_match = ensemble_pattern.search(content)
    features_match = features_pattern.search(content)
    results = {}
    if acc_match:
        results['KNN'] = float(acc_match.group(1))
        results['SVM'] = float(acc_match.group(2))
        results['RF'] = float(acc_match.group(3))
    if ensemble_match:
        results['Ensemble'] = float(ensemble_match.group(1))
    if features_match:
        results['GA_selected'] = int(features_match.group(1))
        results['GA_total'] = int(features_match.group(2))
        results['GA_percent'] = float(features_match.group(3))
    return results

def compare_files(file_paths):
    try:
        from tabulate import tabulate
    except ImportError:
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip3", "install", "tabulate"])
        from tabulate import tabulate
    all_results = []
    for path in file_paths:
        res = parse_results(path)
        all_results.append({
            'File': path,
            'KNN': f"{res.get('KNN', '-'):.4f}" if 'KNN' in res else '-',
            'SVM': f"{res.get('SVM', '-'):.4f}" if 'SVM' in res else '-',
            'RF': f"{res.get('RF', '-'):.4f}" if 'RF' in res else '-',
            'Ensemble': f"{res.get('Ensemble', '-'):.4f}" if 'Ensemble' in res else '-',
            'GA_selected/GA_total (%)': f"{res.get('GA_selected', '-')}/{res.get('GA_total', '-')} ({res.get('GA_percent', '-')})" if 'GA_selected' in res else '-'
        })
    print("\nComparison of Results:")
    print(tabulate(all_results, headers="keys", tablefmt="grid"))
    # Optionally, highlight best ensemble
    best_ensemble = max(all_results, key=lambda x: float(x['Ensemble']) if x['Ensemble'] != '-' else 0)
    print(f"\nBest Ensemble Accuracy: {best_ensemble['Ensemble']} in {best_ensemble['File']}")

def main():
    n = int(input("Enter number of files to compare: "))
    file_paths = []
    for i in range(n):
        path = input(f"Enter path for file {i+1}: ")
        file_paths.append(path)
    compare_files(file_paths)

if __name__ == "__main__":
    main()
