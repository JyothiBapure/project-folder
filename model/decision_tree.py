import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report, precision_recall_curve
)
from model.data_utils import load_train_and_test_data


def run_dt(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=True,
        uploaded_test_file=uploaded_test_df
    )
    decision_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=300,
        min_samples_split=500,
        class_weight='balanced',
        random_state=42
    )
    #decision_model = DecisionTreeClassifier(criterion = 'entropy')
    decision_model.fit(X_train, y_train)
    y_probs = decision_model.predict_proba(X_test)[:, 1]

    # Calculate Optimal Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    best_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    y_preds = (y_probs >= best_threshold).astype(int)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_preds), 4),
        "AUC": round(roc_auc_score(y_test, y_probs), 4),
        "Precision": round(precision_score(y_test, y_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_preds, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_preds), 4),
        
    }
   
    return (
        metrics,
        confusion_matrix(y_test, y_preds),
        classification_report(y_test, y_preds, output_dict=True)
    )
