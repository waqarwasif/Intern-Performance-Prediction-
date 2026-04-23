"""
Intern Performance Prediction - Model Explainability
Loads the trained XGBoost model and generates a Feature Importance chart.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("🔍 Starting Model Explainability Analysis...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../models/xgb_model.pkl")
    figures_dir = os.path.join(current_dir, "../reports/figures")

    os.makedirs(figures_dir, exist_ok=True)

    print("🧠 Loading trained XGBoost model...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find model at {model_path}")
        print("Please run src/train.py first.")
        return

    print("📊 Calculating feature importances...")
    importances = model.feature_importances_
    feature_names = model.feature_names_in_

    importance_series = pd.Series(importances, index=feature_names).sort_values(
        ascending=True
    )

    print("🎨 Drawing the Explainability chart...")
    plt.figure(figsize=(10, 6))

    bars = plt.barh(importance_series.index, importance_series.values, color="#1f77b4")

    plt.title(
        "🧠 What drives Intern Performance?\n(XGBoost Feature Importance)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Importance Score (Higher = More Impact)", fontsize=12)
    plt.ylabel("Intern Metrics", fontsize=12)

    for bar in bars:
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    save_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Success! Explainability chart saved to: {save_path}")


if __name__ == "__main__":
    main()
