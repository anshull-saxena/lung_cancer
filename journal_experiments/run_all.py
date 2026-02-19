"""
Master runner script for all journal experiments.

Usage:
    python run_all.py --all                    # Run all experiments
    python run_all.py --tables 2 3             # Run Tables 2 & 3 only
    python run_all.py --tables 6               # Run Table 6 (NSGA-II) only
    python run_all.py --tables 11 12 13 14     # Run ablation studies
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import set_seed, RESULTS_DIR

# Map table numbers to experiment modules
TABLE_MAP = {
    2: "experiments.table2_3_base",
    3: "experiments.table2_3_base",       # same module as Table 2
    4: "experiments.table4_crossval",
    5: "experiments.table5_attention",
    6: "experiments.table6_nsga2",
    7: "experiments.table7_grouping",
    8: "experiments.table8_9_combined",
    9: "experiments.table8_9_combined",   # same module as Table 8
    10: "experiments.table10_nsga2_full",
    11: "experiments.table11_14_ablation",
    12: "experiments.table11_14_ablation",  # same module
    13: "experiments.table11_14_ablation",
    14: "experiments.table11_14_ablation",
}

# Execution order (module name → display label)
ORDERED_MODULES = [
    ("experiments.table2_3_base",       "Tables 2 & 3"),
    ("experiments.table6_nsga2",        "Table 6"),
    ("experiments.table4_crossval",     "Table 4"),
    ("experiments.table5_attention",    "Table 5"),
    ("experiments.table7_grouping",     "Table 7"),
    ("experiments.table8_9_combined",   "Tables 8 & 9"),
    ("experiments.table10_nsga2_full",  "Table 10"),
    ("experiments.table11_14_ablation", "Tables 11-14"),
]


def run_module(module_name, label):
    """Import and run a specific experiment module."""
    print(f"\n{'#' * 70}")
    print(f"# Running: {label}")
    print(f"{'#' * 70}")
    t0 = time.time()

    mod = __import__(module_name, fromlist=["run"])
    result = mod.run()

    elapsed = time.time() - t0
    print(f"\n  {label} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run journal experiments for Adaptive GA Deep Feature Selector"
    )
    parser.add_argument(
        "--tables", nargs="+", type=int,
        help="Table numbers to run (e.g., --tables 2 3 6)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments in recommended order"
    )
    args = parser.parse_args()

    if not args.all and not args.tables:
        parser.print_help()
        print("\nExample: python run_all.py --tables 2 3")
        print("         python run_all.py --all")
        sys.exit(0)

    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_start = time.time()

    if args.all:
        # Run all in recommended order
        already_run = set()
        for module_name, label in ORDERED_MODULES:
            if module_name not in already_run:
                run_module(module_name, label)
                already_run.add(module_name)
    else:
        # Run specific tables
        already_run = set()
        for table_num in args.tables:
            if table_num not in TABLE_MAP:
                print(f"Warning: Table {table_num} not recognized, skipping.")
                continue
            module_name = TABLE_MAP[table_num]
            if module_name not in already_run:
                # Find label
                label = next(
                    (lbl for mod, lbl in ORDERED_MODULES if mod == module_name),
                    f"Table {table_num}"
                )
                run_module(module_name, label)
                already_run.add(module_name)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  All requested experiments completed in {total_elapsed:.1f}s")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")

    # Generate combined LaTeX output
    _generate_combined_latex()


def _generate_combined_latex():
    """Combine all .tex files in results/ into one document."""
    tex_files = sorted(
        f for f in os.listdir(RESULTS_DIR) if f.endswith(".tex")
    )
    if not tex_files:
        return

    combined_path = os.path.join(RESULTS_DIR, "all_tables.tex")
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\title{Adaptive GA Deep Feature Selector -- Journal Experiments}",
        r"\maketitle",
        "",
    ]

    for tf_name in tex_files:
        if tf_name == "all_tables.tex":
            continue
        tf_path = os.path.join(RESULTS_DIR, tf_name)
        with open(tf_path, "r") as f:
            lines.append(f"% --- {tf_name} ---")
            lines.append(f.read())
            lines.append("")

    lines.append(r"\end{document}")

    with open(combined_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Combined LaTeX saved to: {combined_path}")


if __name__ == "__main__":
    main()
