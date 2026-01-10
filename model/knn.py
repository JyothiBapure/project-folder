from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_utils import load_train_and_test_data

def run_knn(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=True,
        uploaded_test_df=uploaded_test_df
    )

    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    knn_model.fit(X_train, y_train)

    y_preds = knn_model.predict(X_test)
    y_probs = knn_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_preds),
        "AUC": roc_auc_score(y_test, y_probs),
        "Precision": precision_score(y_test, y_preds),
        "Recall": recall_score(y_test, y_preds),
        "F1": f1_score(y_test, y_preds),
        "MCC": matthews_corrcoef(y_test, y_preds)
    }

    return (
        metrics,
        confusion_matrix(y_test, preds),
        classification_report(y_test, preds, output_dict=True)
    )
