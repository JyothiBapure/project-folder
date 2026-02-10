import streamlit as st
import pandas as pd
import numpy as np

from model.logistic_regression import run_logic
from model.decision_tree import run_dt
from model.knn import run_knn
from model.naive_bayes import run_nb
from model.random_forest import run_rf
from model.xgboost_model import run_xgb

st.set_page_config(page_title="ML Models Evaluation", layout="wide")
st.title("ML Models Evaluation to Predict Annual Income")

st.markdown("""
* If no test file is uploaded, evaluation will run on the **internal test split**.
* If a test file is uploaded, evaluation will run on the uploaded dataset.
""")

req_columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

#st.sidebar.subheader("📋 Required Dataset Columns")
#with st.sidebar.expander("Click to view required columns"):
#    for col in req_columns:
#        st.markdown(f"- `{col}`")
#

# --------------------------------------------------
# Download test dataset
# --------------------------------------------------
#st.sidebar("Download Test Dataset")

st.sidebar.subheader("Download Sample Test Dataset")
try:
    with open("data/adult.test", "rb") as f:
        adult_test_bytes = f.read()

    st.sidebar.download_button(
        label="Download Test Data adult.test",
        data=adult_test_bytes,
        file_name="adult.test",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("Test dataset not found in repository.")

st.markdown("---")

# Sidebar: Upload test CSV or .test
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (.csv or .test, with or without header)",
    type=["csv", "test"]
)

uploaded_test_df = None
if uploaded_file is not None:
    uploaded_test_df = pd.read_csv(
        uploaded_file,
        header=None,        # no header in adult.test
        names=req_columns,
        sep=r',\s*',        # handle spaces after commas
        engine='python'
    )

    # Strip spaces from column names
    uploaded_test_df.columns = uploaded_test_df.columns.str.strip()

    # Strip whitespace from string columns
    str_cols = uploaded_test_df.select_dtypes(include='object').columns
    uploaded_test_df[str_cols] = uploaded_test_df[str_cols].apply(lambda x: x.str.strip())

    # Remove trailing periods from income (e.g., <=50K. -> <=50K)
    if 'income' in uploaded_test_df.columns:
        uploaded_test_df['income'] = uploaded_test_df['income'].str.replace('.', '', regex=False)

    st.success("Test dataset uploaded and cleaned successfully!")

    st.subheader("📄 Preview of Uploaded Test Dataset (Top 10 Rows)")
    st.dataframe(uploaded_test_df.head(10), width="stretch")

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
    st.write("Running evaluation for Logistic Regression")
    metrics, confusion_metrix_, report = run_logic(uploaded_test_df)

elif selected_model == "Decision Tree Classifier":
    st.write("Running evaluation for Decision Tree Classifier")
    metrics, confusion_metrix_, report = run_dt(uploaded_test_df)

elif selected_model == "K-Nearest Neighbor Classifier":
    st.write("Running evaluation for K-Nearest Neighbor Classifier")
    metrics, confusion_metrix_, report = run_knn(uploaded_test_df)

elif selected_model == "Naive Bayes Classifier":
    st.write("Running evaluation for Naive Bayes Classifier")
    metrics, confusion_metrix_, report = run_nb(uploaded_test_df)

elif selected_model == "Ensemble Model - Random Forest":
    st.write("Running evaluation for Ensemble Model - Random Forest")
    metrics, confusion_metrix_, report = run_rf(uploaded_test_df)

elif selected_model == "Ensemble Model - XGBoost":
    st.write("Running evaluation for Ensemble Model - XGBoost")
    metrics, confusion_metrix_, report = run_xgb(uploaded_test_df)


# Display metrics
st.subheader("Evaluation Metrics")
#for k, v in metrics.items():
#    st.metric(k, v)
items = list(metrics.items())

for i in range(0, len(items), 2):
    col1, col2 = st.columns(2)

    with col1:
        st.metric(items[i][0], round(float(items[i][1]), 4))

    if i + 1 < len(items):
        with col2:
            st.metric(items[i+1][0], round(float(items[i+1][1]), 4))


# Display confusion matrix
st.subheader("Confusion Matrix")
st.write(confusion_metrix_)

# Display classification report
st.subheader("Classification Report")
filtered_report = {k: v for k, v in report.items() if isinstance(v, dict)}
report_df = pd.DataFrame.from_dict(filtered_report, orient="index")

# Add accuracy row safely
if "accuracy" in report:
    report_df.loc["accuracy", "precision"] = report["accuracy"]
    report_df.loc["accuracy", "recall"] = np.nan
    report_df.loc["accuracy", "f1-score"] = np.nan
    report_df.loc["accuracy", "support"] = np.nan

# Ensure numeric types and round
report_df = report_df[["precision", "recall", "f1-score", "support"]].astype(float).round(4)

st.dataframe(report_df, width="stretch")
