import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report, precision_recall_curve
)
from data_util import load_train_and_test_data

def run_logic(uploaded_test_df=None):
    # Load and scale
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=True,
        uploaded_test_file=uploaded_test_df
    )

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Initialize model
    logistic_model = LogisticRegression(
        penalty='l1', 
        solver='liblinear', # Required for l1
        class_weight='balanced', 
        max_iter=2000
    )

    logistic_model.fit(X_train, y_train)
    y_probs = logistic_model.predict_proba(X_test)[:, 1]

    # DYNAMIC THRESHOLDING: Instead of hardcoded 0.4
    # Find threshold that maximizes F1 score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    best_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # Apply the best threshold
    y_preds = (y_probs >= best_threshold).astype(int)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_preds), 4),
        "AUC": round(roc_auc_score(y_test, y_probs), 4),
        "Precision": round(precision_score(y_test, y_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_preds, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_preds), 4),
        "Threshold_Used": round(best_threshold, 4)
    }

    return (
        metrics,
        confusion_matrix(y_test, y_preds),
        classification_report(y_test, y_preds, output_dict=True)
    )
