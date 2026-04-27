"""
setup_repo.py
=============
Run this script once to scaffold the full project directory for:
"Adult Census Income Analysis — Predicting Income Class with ML"

Usage:
    python setup_repo.py

Author: Rahmadhania
"""

import os

# ── Project root (change this if you want it somewhere else) ──────────────────
BASE_DIR = "adult-census-income-analysis"

# ── Folder structure ──────────────────────────────────────────────────────────
FOLDERS = [
    "data",
    "notebooks",
    "scripts",
    "visuals",
]

# ── Placeholder files to create inside each folder ───────────────────────────
PLACEHOLDER_FILES = {
    "data":      [".gitkeep"],          # drop adult.csv here after download
    "notebooks": [".gitkeep"],
    "scripts":   [".gitkeep"],
    "visuals":   [".gitkeep"],
}

# ── Root-level files ──────────────────────────────────────────────────────────
ROOT_FILES = {
    "README.md": "# Adult Census Income Analysis\n\n> Run `setup_repo.py` to scaffold this project.\n",
    ".gitignore": (
        "# Python\n__pycache__/\n*.pyc\n*.pyo\n.env\n\n"
        "# Jupyter\n.ipynb_checkpoints/\n\n"
        "# Data (do not commit raw data)\ndata/*.csv\ndata/*.data\ndata/*.test\n\n"
        "# OS\n.DS_Store\nThumbs.db\n"
    ),
    "requirements.txt": (
        "pandas>=2.0\n"
        "numpy>=1.24\n"
        "matplotlib>=3.7\n"
        "seaborn>=0.12\n"
        "scikit-learn>=1.3\n"
        "ucimlrepo>=0.0.3\n"
        "jupyter>=1.0\n"
        "ipykernel>=6.0\n"
    ),
}


def create_structure():
    print(f"\n📁  Creating project: '{BASE_DIR}'\n")

    # Create root
    os.makedirs(BASE_DIR, exist_ok=True)

    # Create subdirectories + placeholders
    for folder in FOLDERS:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  ✅  {path}/")

        for placeholder in PLACEHOLDER_FILES.get(folder, []):
            fp = os.path.join(path, placeholder)
            with open(fp, "w") as f:
                f.write("")  # empty placeholder so Git tracks the folder
            print(f"       └─ {placeholder}")

    # Create root-level files
    print()
    for filename, content in ROOT_FILES.items():
        fp = os.path.join(BASE_DIR, filename)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅  {fp}")

    print("\n🎉  Setup complete!")
    print(f"\nNext steps:")
    print(f"  1. cd {BASE_DIR}")
    print(f"  2. pip install -r requirements.txt")
    print(f"  3. Copy adult_income_analysis.py  → scripts/")
    print(f"  4. Copy adult_income_analysis.ipynb → notebooks/")
    print(f"  5. The dataset is fetched automatically via ucimlrepo inside the scripts.\n")


if __name__ == "__main__":
    create_structure()
