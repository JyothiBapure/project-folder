import numpy as np
import pandas as pd
import json
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

def run_logic():
    np.random.seed(42)

    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    file = "data/adult.data"
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]

    data = pd.read_csv(
        file, header=None, names=column_names,
        sep=r",\s*", engine="python"
    )

    data = data.replace('?', pd.NA).dropna()

    X = data.drop('income', axis=1)
    y = data['income'].apply(lambda x: 0 if x == '<=50K' else 1)

    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    }

    conf_matrix = confusion_matrix(y_test, preds)
    classification_report(y_test, preds, output_dict=True)
    return metrics, conf_matrix, class_report
