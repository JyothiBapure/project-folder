# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model.data_util import load_training_data, load_test_data, preprocess

from model import (
    logistic_model,
    #decision_tree_model,
    #knn_model,
    #naive_bayes_model,
    #random_forest_model,
    #xgboost_model
)

st.title("Adult Income Classification App")

# Upload test dataset
uploaded_file = st.file_uploader("Upload Test Dataset (adult.test)", type=["csv"])

if uploaded_file is not None:

    train_df = load_training_data("adult.data")
    test_df = load_test_data(uploaded_file)

    X_train, X_test, y_train, y_test = preprocess(train_df, test_df)

    model_choice = st.selectbox(
        "Select Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    if model_choice == "Logistic Regression":
        model, metrics = logistic_model.train_and_evaluate(X_train, X_test, y_train, y_test)

    st.subheader("Evaluation Metrics")

    for key, value in metrics.items():
        st.write(f"{key}: {value:.4f}")

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)

    st.pyplot(fig)

else:
    st.info("Please upload the adult.test file to proceed.")
