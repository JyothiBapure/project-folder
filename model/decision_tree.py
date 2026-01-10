from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_utils import load_and_preprocess_data

def run_dt():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        scale_features=False
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=20,
        random_state=42
    )

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

    return (
        metrics,
        confusion_matrix(y_test, preds),
        classification_report(y_test, preds, output_dict=True)
    )
