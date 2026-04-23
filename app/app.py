"""
Intern Performance Prediction - Streamlit Web Dashboard
This script creates a web interface that allows users to input intern metrics
and instantly receive a performance prediction from our tuned XGBoost model.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Intern AI Predictor", page_icon="🚀", layout="centered")


# 2. Load the Model Safely
# use @st.cache_resource so the app doesn't reload the model on every single click!
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../models/xgb_model.pkl")
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error("❌ Model not found! Please run `python src/train.py` first.")
    st.stop()

# 3. Build the User Interface
st.title("🚀 AI Intern Performance Predictor")
st.markdown(
    """
Welcome to the HR Prediction Dashboard. 
Adjust the intern's metrics below, and our Tuned XGBoost AI will predict their final performance score.
"""
)

st.header("📋 Intern Metrics")

# Create interactive sliders in two columns for a clean UI
col1, col2 = st.columns(2)

with col1:
    attendance = st.slider(
        "Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0
    )
    completion_time = st.slider(
        "Task Completion Time (Days)", min_value=1.0, max_value=7.0, value=3.5, step=0.1
    )

with col2:
    feedback_rating = st.slider(
        "Manager Feedback Rating (1-5)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
    )

# 4. Feature Engineering (Hidden from user)
# Must calculate the Efficiency Score just like did in preprocess.py
if completion_time == 0:
    completion_time = 0.1  # Prevent division by zero
efficiency_score = feedback_rating / completion_time

st.info(
    f"⚙️ **Calculated Efficiency Score:** {efficiency_score:.2f} (Feedback points per day)"
)

# 5. The Prediction Engine
st.divider()

# When the user clicks the button...
if st.button("🎯 Predict Final Score", type="primary", use_container_width=True):

    # Package the inputs into a DataFrame with the EXACT column names the model expects
    input_data = pd.DataFrame(
        [[completion_time, feedback_rating, attendance, efficiency_score]],
        columns=[
            "Completion_Time",
            "Feedback_Rating",
            "Attendance",
            "Efficiency_Score",
        ],
    )

    # Run the prediction
    prediction = model.predict(input_data)[0]

    # Display the result beautifully
    st.subheader("Predicted Performance Score:")

    # Color-code the output based on how well the intern is expected to do
    if prediction >= 75:
        st.success(f"🌟 {prediction:.1f} / 100 (Excelling Intern)")
    elif prediction >= 50:
        st.warning(f"👍 {prediction:.1f} / 100 (Average Intern)")
    else:
        st.error(f"⚠️ {prediction:.1f} / 100 (Struggling Intern)")

    st.caption("Model Accuracy: MAE ± 7.4 points | Algorithm: Tuned XGBoost")
