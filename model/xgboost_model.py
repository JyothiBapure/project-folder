from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_utils import load_train_and_test_data

def run_xgb(uploaded_test_df=None):
    X_train, X_test, y_train, y_test = load_train_and_test_data(
        scale_features=False,
        uploaded_test_file=uploaded_test_df
    )

    xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    y_preds = xgb_model.predict(X_test)
    y_probs = xgb_model.predict_proba(X_test)[:, 1]

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
        confusion_matrix(y_test, y_preds),
        classification_report(y_test, y_preds, output_dict=True)
    )
