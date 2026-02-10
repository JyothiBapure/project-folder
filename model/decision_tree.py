from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_utils import load_train_and_test_data

def run_dt(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=True,
        uploaded_test_file=uploaded_test_df
    )

    #decision_model = DecisionTreeClassifier(
    #    criterion="gini",
    #    max_depth=10,
    #    min_samples_split=20,
    #    random_state=42
    #)

    #decision_model.fit(X_train, y_train)

    #y_preds = decision_model.predict(X_test)
    #y_probs = decision_model.predict_proba(X_test)[:, 1]

    decision_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_leaf=50,
        min_samples_split=50,
        class_weight='balanced',
        random_state=42
    )

    decision_model.fit(X_train, y_train)

    y_probs = decision_model.predict_proba(X_test)[:, 1]

    threshold = 0.4   # try 0.3, 0.4, 0.5
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
