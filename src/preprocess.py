import pandas as pd
import os 
from sklearn.impute import KNNImputer


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    if "Intern_ID" in df.columns:
        df = df.drop(columns=["Intern_ID"])

    # Filter out corrupted HR attendance records
    df = df[df["Attendance"] >= 5.0].copy()

    # Impute missing values (MAR) using KNN
    features_to_impute = ["Completion_Time", "Feedback_Rating", "Attendance"]
    imputer = KNNImputer(n_neighbors=5)
    df[features_to_impute] = imputer.fit_transform(df[features_to_impute])

    # Engineer predictive feature
    df["Efficiency_Score"] = df["Feedback_Rating"] / df["Completion_Time"]

    return df


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path  = os.path.join(BASE_DIR, "data", "raw", "intern_dataset_realistic (1).csv")
    output_path = os.path.join(BASE_DIR, "data", "processed", "clean_intern_data.csv")

    try:
        clean_df = load_and_preprocess(input_path)
        clean_df.to_csv(output_path, index=False)
        print(f"Preprocessing Complete! Clean dataset saved to: {output_path}")
        print(f"Final Dataset Shape: {clean_df.shape}")
    except FileNotFoundError:
        print(
            f"Error: Could not find the file at {input_path}. Please check your path."
        )
