"""
Intern Performance Prediction — Project Setup Script
Run: python setup_project.py
Creates the full folder structure with empty files.
"""

import os

BASE = "intern_performance"

# ── Folder structure ──────────────────────────────────────────────────────────

folders = [
    f"{BASE}/data/raw",
    f"{BASE}/data/processed",
    f"{BASE}/notebooks",
    f"{BASE}/src",
    f"{BASE}/models",
    f"{BASE}/reports/figures",
    f"{BASE}/app",
]

# ── Files to create (all empty) ───────────────────────────────────────────────

files = [
    # Root
    f"{BASE}/requirements.txt",
    f"{BASE}/.gitignore",
    f"{BASE}/README.md",

    # Source modules
    f"{BASE}/src/__init__.py",              
    f"{BASE}/src/preprocess.py",           # Day 2 - EDA, feature engineering, train/test split
    f"{BASE}/src/train.py",                # Day 3 - train Random Forest + XGBoost, evaluate
    f"{BASE}/src/explain.py",              # Day 4 - SHAP plots

    # Streamlit app
    f"{BASE}/app/__init__.py",
    f"{BASE}/app/app.py",                  # Day 4 - Streamlit dashboard

    # Notebooks
    f"{BASE}/notebooks/01_eda.ipynb",              # Day 2 - exploratory analysis
    f"{BASE}/notebooks/02_modeling.ipynb",         # Day 3 - model training walkthrough
    f"{BASE}/notebooks/03_explainability.ipynb",   # Day 4 - SHAP deep dive

    # Reports
    f"{BASE}/reports/model_comparison.csv",        # filled after Day 3 training

    # Placeholders so Git tracks empty folders
    f"{BASE}/data/raw/.gitkeep",
    f"{BASE}/data/processed/.gitkeep",
    f"{BASE}/models/.gitkeep",
    f"{BASE}/reports/figures/.gitkeep",
]

# ── Run ───────────────────────────────────────────────────────────────────────

def setup():
    print("\n Creating folders...\n")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"   {folder}/")

    print("\n Creating files...\n")
    for filepath in files:
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                pass  # empty file
            print(f"  {filepath}")
        else:
            print(f" already exists, skipped: {filepath}")

    print("\nProject scaffold ready!\n")
    print("Next steps:")
    print("  1. cd intern_performance")
    print("  2. pip install -r requirements.txt")
    print("  3. Start coding in src/generate_data.py  (Day 1)\n")

if __name__ == "__main__":
    setup()