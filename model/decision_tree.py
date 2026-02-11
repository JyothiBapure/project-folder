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
    X_train, X_val, X_test, y_train, y_val, y_test = load_train_and_test_data(
        scale_features=True,
        uploaded_test_file=uploaded_test_df
    )
    decision_model = DecisionTreeClassifier(
        max_depth=8,
        criterion='entropy',
        min_samples_leaf=20,
        min_samples_split=50,
        class_weight=None,
        random_state=42
    )
    #decision_model = DecisionTreeClassifier(criterion = 'entropy')
    decision_model.fit(X_train, y_train)
    y_val_probs = decision_model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    best_threshold = thresholds[optimal_idx]

    # Evaluate on test
    y_probs = decision_model.predict_proba(X_test)[:, 1]
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
