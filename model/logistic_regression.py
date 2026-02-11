from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, precision_recall_curve
)

from model.data_utils import load_train_and_test_data

def get_metrics(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    y_preds = (y_probs >= best_threshold).astype(int)
    
    return {
        "Accuracy": round(accuracy_score(y_true, y_preds), 4),
        "AUC": round(roc_auc_score(y_true, y_probs), 4),
        "Precision": round(precision_score(y_true, y_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_preds, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_preds), 4),        
    }


def run_logic(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(scale_features=True, uploaded_test_file=uploaded_test_df)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    return get_metrics(y_test, y_probs)
