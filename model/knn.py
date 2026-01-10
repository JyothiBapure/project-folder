import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

def run_knn():
    np.random.seed(42)

    file = "data/adult.data"

    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]

    data = pd.read_csv(
        file,
        header=None,
        names=column_names,
        sep=r",\s*",
        engine="python"
    )

    # Handle missing values
    data = data.replace('?', pd.NA).dropna()

    X = data.drop('income', axis=1)
    y = data['income'].apply(lambda x: 0 if x == '<=50K' else 1)

    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # KNN REQUIRES FEATURE SCALING
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN model
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2   # Euclidean distance
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # For AUC, need probabilities
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

    class_report = classification_report(
        y_test,
        preds,
        output_dict=True
    )

    return metrics, conf_matrix, class_report
