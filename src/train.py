"""
Intern Performance Prediction - Model Training Pipeline
This script loads preprocessed data, trains the champion XGBoost model
using optimal hyperparameters, and serializes the model for deployment.
"""

import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    print("Starting Production Training Pipeline...")

    # 1. Path Handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/processed/clean_intern_data.csv")
    model_dir = os.path.join(current_dir, "../models")

    # Ensure the models directory exists
    os.makedirs(model_dir, exist_ok=True)

    # 2. Load Data
    print("Loading clean dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find data at {data_path}")
        print("Please run src/preprocess.py first.")
        return

    # 3. Data Splitting
    print("Splitting data into training and testing sets...")
    X = df.drop(columns=["Performance_Score"])
    y = df["Performance_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Initialize Champion Model
    # Using the exact optimal parameters found via RandomizedSearchCV
    print("Initializing Tuned XGBoost Model...")
    champion_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        reg_alpha=0,
        random_state=42,
        verbosity=0,
    )

    # 5. Train the Model
    print("Training model on 80% data...")
    champion_model.fit(X_train, y_train)

    # 6. Evaluate on Test Set
    print("Evaluating model on 20% blind test data...")
    predictions = champion_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nFinal Production Metrics")
    print("-" * 30)
    print(f"Mean Absolute Error : {mae:.3f}")
    print(f"R² Score            : {r2:.3f}")
    print("-" * 30)

    # 7. Serialize and Save
    model_save_path = os.path.join(model_dir, "xgb_model.pkl")
    joblib.dump(champion_model, model_save_path)
    print(f"\n✅ Pipeline Complete! Model safely serialized to: {model_save_path}")



if __name__ == "__main__":
    main()
