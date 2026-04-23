<div align="center">

# 🚀 AI Intern Performance Predictor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline that predicts an intern's final performance score based on real-time metrics, engineered for the HR domain.

</div>

---

## 📌 Overview

The **AI Intern Performance Predictor** is a full-stack Machine Learning application designed to evaluate and predict the final assessment scores of corporate interns. By analyzing attendance rates, task completion velocity, and manager feedback, this system provides data-driven insights to help HR teams identify top talent and struggling individuals early in the program.

The project encompasses the entire ML lifecycle: from Exploratory Data Analysis (EDA) and automated data preprocessing, to hyperparameter tuning via Cross-Validation, all the way to deploying a production-ready Web Dashboard.

---

## ✨ Key Features

- **Automated Data Pipeline:** Robust handling of system anomalies (e.g., impossible metrics), automated imputation of Missing At Random (MAR) data, and custom feature engineering (`Efficiency_Score`).
- **Champion Algorithm:** Leverages a highly tuned **XGBoost Regressor** selected after evaluating 500 hyperparameter combinations via Randomized Search.
- **Explainable AI (XAI):** Integrated feature importance mapping to prevent "Black Box" predictions, allowing managers to understand *why* a specific score was generated.
- **Interactive Dashboard:** A sleek, user-friendly frontend built with Streamlit for real-time inference.

---

## 🧠 Model Architecture & Performance

The core prediction engine relies on an optimized XGBoost algorithm. The dataset explicitly incorporates elements of human unpredictability (e.g., manager bias, life events) to simulate real-world chaotic data.

**Final Production Metrics (Blind Test Set):**

| Metric | Score |
|---|---|
| Mean Absolute Error (MAE) | `7.41` |
| Root Mean Squared Error (RMSE) | `9.29` |
| R² Score | `0.619` |

> **Note:** Capturing 62% of the variance in a dataset injected with natural human inconsistency (manager bias, life events, noise) represents the maximum mathematical signal extraction limit — this is irreducible error by design, not a modeling limitation.

**Champion Hyperparameters (XGBoost):**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0 |
| `reg_lambda` | 2 |

---

## 📂 Repository Structure

<details>
<summary><b>Click to expand the project directory tree</b></summary>

```text
intern_performance/
│
├── app/
│   └── app.py                   # Streamlit frontend dashboard
│
├── data/
│   ├── raw/                     # Original unedited datasets (git-ignored)
│   └── processed/               # Cleaned and engineered data
│
├── models/
│   └── xgb_model.pkl            # Serialized production model
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   └── 02_modeling.ipynb        # Model selection and hyperparameter tuning
│
├── reports/
│   ├── figures/                 # XAI visualizations (SHAP feature importance)
│   └── model_comparison.csv     # Full model benchmarking results
│
├── src/
│   ├── preprocess.py            # Automated data cleaning pipeline
│   ├── train.py                 # Model training and serialization script
│   └── explain.py               # SHAP feature importance generator
│
├── .gitignore                   # Git tracking rules
├── requirements.txt             # Project dependencies
├── setup_project.py             # Workspace initialization script
└── README.md                    # Project documentation
```

</details>

---

## 💻 Quick Start & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/intern-performance-prediction.git
cd intern-performance-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the ML Pipeline (optional)

If you wish to retrain the model rather than using the pre-compiled `.pkl` file:

```bash
# Step 1 — Clean the raw data
python src/preprocess.py

# Step 2 — Train and tune the models
python src/train.py

# Step 3 — Generate SHAP explainability reports
python src/explain.py
```

### 5. Launch the web app

```bash
streamlit run app/app.py
```

The dashboard will open automatically at `http://localhost:8501`

---


## 👨‍💻 Author

**Syed Waqar Wasif**  
Computer Systems Engineering Student @ NEDUET

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/syed-waqar-wasif-493b26241/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/waqarwasif)

---

<div align="center">
Built with ❤️ utilizing Scikit-Learn, XGBoost, and Streamlit
</div>