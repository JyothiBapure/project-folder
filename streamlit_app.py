import streamlit as st
import pandas as pd

from model.logistic_regression import run_logic
from model.decision_tree import run_dt
from model.knn import run_knn
from model.naive_bayes import run_nb
from model.random_forest import run_rf
from model.xgboost_model import run_xgb

st.title("ML Models Evaluation")

st.markdown(
    """
    **Required columns in uploaded CSV file:**

    ```
    age, workclass, fnlwgt, education, education-num,
    marital-status, occupation, relationship, race, sex,
    capital-gain, capital-loss, hours-per-week,
    native-country, income
    ```

    • Column names must match exactly  
    • `income` column is required for evaluation  
    • If no file is uploaded, evaluation evaluation run on **internal test split**
    * If uploaded evaluation evaluation run on uploaded test data
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV with same columns as Adult dataset)",
    type=["csv"]
)

uploaded_test_df = None
if uploaded_file is not None:
    uploaded_test_df = pd.read_csv(uploaded_file)
    st.success("Test dataset uploaded successfully!")

selected_model = st.selectbox(
    "Choose Model",
    (
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier",
        "Ensemble Model - Random Forest",
        "Ensemble Model - XGBoost"
    )
)

if selected_model == "Logistic Regression":
    metrics, confusion_metrix, report = run_logic(uploaded_test_df)

elif selected_model == "Decision Tree Classifier":
    metrics, confusion_metrix, report = run_dt(uploaded_test_df)

elif selected_model == "K-Nearest Neighbor Classifier":
    metrics, confusion_metrix, report = run_knn(uploaded_test_df)

elif selected_model == "Naive Bayes Classifier":
    metrics, confusion_metrix, report = run_nb(uploaded_test_df)

elif selected_model == "Ensemble Model - Random Forest":
    metrics, confusion_metrix, report = run_rf(uploaded_test_df)

elif selected_model == "Ensemble Model - XGBoost":
    metrics, confusion_metrix, report = run_xgb(uploaded_test_df)

# Display results
st.subheader("Evaluation Metrics")
for k, v in metrics.items():
    st.metric(k, round(v, 4))

st.subheader("Confusion Matrix")
st.write(confusion_metrix)

report_df = pd.DataFrame.from_dict(report, orient="index")
report_df = report_df[
    ["precision", "recall", "f1-score", "support"]
].round(4)

st.subheader("Classification Report")
st.dataframe(report_df, use_container_width=True)
