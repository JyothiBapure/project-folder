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
        uploaded_test_file=uploaded_test_df
    )

    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    knn_model.fit(X_train, y_train)

    #y_preds = knn_model.predict(X_test)
    #y_probs = knn_model.predict_proba(X_test)[:, 1]
    y_probs = knn_model.predict_proba(X_test)[:, 1]

    threshold = 0.4   # keep same threshold for fair comparison
    y_preds = (y_probs >= threshold).astype(int)


    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_preds, zero_division=0), 4),
        "AUC": round(roc_auc_score(y_test, y_probs, zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_preds, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_preds, zero_division=0), 4)
    }

    return (
        metrics,
        confusion_matrix(y_test, y_preds),
        classification_report(y_test, y_preds, output_dict=True)
    )
