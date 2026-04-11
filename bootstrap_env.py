from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Bootstrap a fresh machine for this repo: create a venv, install dependencies, "
            "run sanity-check imports, and verify dataset structure."
        )
    )
    ap.add_argument("--venv", default=".venv", help="Virtualenv folder to create/use")
    ap.add_argument("--repo-root", default=".", help="Path to repo root")
    ap.add_argument("--check-only", action="store_true", help="Skip installs; only run checks")
    ap.add_argument(
        "--kernel",
        default="",
        help="Optional: install a Jupyter kernel name (e.g. lung_cancer)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    req = repo_root / "requirements.txt"
    if not req.exists():
        raise SystemExit(f"Could not find requirements.txt at: {req}")

    vdir = repo_root / args.venv
    py = venv_python(vdir)

    print(f"OS: {platform.system()} {platform.machine()}")
    print(f"Host Python: {sys.version.split()[0]}")
    print(f"Repo root: {repo_root}")
    print(f"Venv dir: {vdir}")

    if not vdir.exists():
        print("\nCreating virtual environment...")
        venv.EnvBuilder(with_pip=True).create(str(vdir))

    if not py.exists():
        raise SystemExit(f"Venv python not found at: {py}")

    if not args.check_only:
        run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        run([str(py), "-m", "pip", "install", "-r", str(req)])

        # requirements.txt uses platform markers for Linux (tf[and-cuda]) and macOS (tensorflow-macos).
        # On Windows those markers won't match, so install TensorFlow explicitly.
        if platform.system() == "Windows":
            run([str(py), "-m", "pip", "install", "tensorflow==2.16.1"])

        if args.kernel:
            run(
                [
                    str(py),
                    "-m",
                    "ipykernel",
                    "install",
                    "--user",
                    "--name",
                    args.kernel,
                    "--display-name",
                    args.kernel,
                ]
            )

    # Import + version + GPU checks inside the venv
    check_code = r"""
import importlib, platform, sys
mods = ["numpy","pandas","scipy","sklearn","deap","PIL","matplotlib","seaborn","psutil","tqdm","xgboost"]
print("Python (venv):", sys.version.split()[0])
print("Platform:", platform.platform())
for m in mods:
    importlib.import_module(m)
print("Core imports: OK")

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print("TF GPUs:", [d.name for d in gpus] if gpus else [])
    except Exception as e:
        print("GPU query failed:", e)
except Exception as e:
    print("TensorFlow import failed:", e)
    raise
"""
    run([str(py), "-c", check_code])

    # Dataset structure check (matches journal_experiments/config.py defaults)
    dataset_dir = repo_root / "dataset" / "lung_image_sets"
    print("\nDataset dir:", dataset_dir)
    expected = ["lung_aca", "lung_n", "lung_scc"]
    missing = [c for c in expected if not (dataset_dir / c).is_dir()]
    if missing:
        print("WARNING: Missing class folders:", missing)
        print("Expected structure: dataset/lung_image_sets/{lung_aca, lung_n, lung_scc}/")
    else:
        for c in expected:
            n = len(list((dataset_dir / c).glob("*")))
            print(f"  {c}: {n} files")

    print("\nNext (Table 17):")
    journal_dir = repo_root / "journal_experiments"
    if os.name == "nt":
        print(f"  cd {journal_dir}")
        print(f"  {py} run_all.py --tables 17")
    else:
        print(f"  cd \"{journal_dir}\"")
        print(f"  \"{py}\" run_all.py --tables 17")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
