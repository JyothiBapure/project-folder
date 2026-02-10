from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_utils import load_train_and_test_data

def run_logic(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=True,
        uploaded_test_file=uploaded_test_df
    )

    #logistic_model = LogisticRegression()
    logistic_model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        solver='lbfgs'
    )

    logistic_model.fit(X_train, y_train)

    #y_preds = logistic_model.predict(X_test)
    #y_probs = logistic_model.predict_proba(X_test)[:, 1]

    y_probs = logistic_model.predict_proba(X_test)[:, 1]

    threshold = 0.3   # try 0.3 or 0.4
    y_preds = (y_probs >= threshold).astype(int)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_preds), 4),
        "AUC": round(roc_auc_score(y_test, y_probs), 4),
        "Precision": round(precision_score(y_test, y_preds), 4),
        "Recall": round(recall_score(y_test, y_preds), 4),
        "F1": round(f1_score(y_test, y_preds), 4),
        "MCC": round(matthews_corrcoef(y_test, y_preds), 4)
    }

    return (
        metrics,
        confusion_matrix(y_test, y_preds),
        classification_report(y_test, y_preds, output_dict=True)
    )
