import streamlit as st
from model.logistic_regression import run_logic

st.title("ML Models Implementation")

st.subheader("Select Model to Display Evaluation Metrics")

selected_model = st.selectbox(
    "Choose a Model",
    ("Logistic Regression", "Decision Tree Classifier")
)

if selected_model == "Logistic Regression":
    st.info("Running Logistic Regression Model...")
    metrics, conf_matrix, class_report = run_logic()

    st.subheader("Evaluation Metrics")
    for key, value in metrics.items():
        st.metric(label=key, value=round(value, 4))

    st.subheader("Confusion Matrix")
    st.write(conf_matrix)

    import pandas as pd

    report_df = pd.DataFrame.from_dict(
        class_report,
        orient="index"
    )

    # Reorder columns to match sklearn display
    report_df = report_df[
        ["precision", "recall", "f1-score", "support"]
    ]

    # Round numeric values
    report_df = report_df.round(4)

    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)
