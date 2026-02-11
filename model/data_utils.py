import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week',
    'native-country', 'income'
]

def map_income(x):
    """Sanitizes and maps target labels for all census data variations."""
    x = str(x).strip().rstrip('.')
    return 1 if x == '>50K' else 0

def preprocess_data(data1, feature_columns=None, has_label=True):
    """
    Unified preprocessing logic:
    1. Cleans missing values and whitespace.
    2. Encodes categorical variables via get_dummies.
    3. Aligns columns to ensure Train/Test consistency for all models.
    """
    # Clean whitespace and handle missing values
    data1.columns = data1.columns.str.strip()
    data1 = data1.replace('?', pd.NA).dropna()
    
    # Extract Target
    if has_label and 'income' in data1.columns:
        y = data1['income'].apply(map_income)
        X = data1.drop(columns=['income'], errors='ignore')
    else:
        X = data1.copy()
        y = None

    # Identify Categorical Columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # One-Hot Encoding (Consistent for LR, KNN, Trees, and Ensemble)
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Align columns: Crucial for XGBoost and MultinomialNB 
    # Ensures test set features exactly match training set features
    if feature_columns is not None:
        X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    return X_encoded, y

def load_train_and_test_data(scale_features=False, uploaded_test_file=None):
    train_raw = pd.read_csv(
        "data/adult.data",
        header=None,
        names=column_names,
        sep=r",\s*",
        engine="python"
    )

    if uploaded_test_file is None:
        # Full preprocessing
        X_full, y_full = preprocess_data(train_raw)

        # First split: Train+Val and Test (80/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_full, y_full,
            test_size=0.2,
            random_state=42,
            stratify=y_full
        )

        # Second split: Train and Validation (75/25 of remaining 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25,   # 0.25 of 0.8 = 0.2 overall
            random_state=42,
            stratify=y_temp
        )

    else:
        # External evaluation case (no validation needed here)
        X_train, y_train = preprocess_data(train_raw)
        X_test, y_test = preprocess_data(
            uploaded_test_file,
            feature_columns=X_train.columns
        )
        X_val, y_val = None, None

    # Scaling (if required)
    if scale_features:
        scaler = StandardScaler()
        X_train_cols = X_train.columns

        X_train = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X_train_cols)
        if X_val is not None:
            X_val = pd.DataFrame(X_val, columns=X_train_cols)
        X_test = pd.DataFrame(X_test, columns=X_train_cols)

    return X_train, X_val, X_test, y_train, y_val, y_test
