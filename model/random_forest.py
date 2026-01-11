from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_utils import load_train_and_test_data

def run_rf(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=False,
        uploaded_test_file=uploaded_test_df
    )

    #model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
    random_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

    random_model.fit(X_train, y_train)

    y_preds = random_model.predict(X_test)
    y_probs = random_model.predict_proba(X_test)[:, 1]

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
