import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Column names for the Adult Census dataset
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

def clean_income(x):
    """Removes trailing dots and whitespace from target labels."""
    if pd.isna(x): return x
    s = str(x).strip().rstrip('.')
    return 1 if s == '>50K' else 0

def load_train_and_test_data(scale_features=False, uploaded_test_file=None):
    """
    Unified loader for all 6 models.
    - If uploaded_test_file is None: Splits adult.data into 80/20.
    - If uploaded_test_file is provided: Uses adult.data for training and the file for testing.
    """
    # 1. Load Training Data
    train_df = pd.read_csv('adult.data', names=COLUMNS, sep=r',\s*', engine='python', na_values='?')
    train_df = train_df.dropna()
    
    # 2. Setup Training Features and Target
    X_train_raw = train_df.drop('income', axis=1)
    y_train = train_df['income'].apply(clean_income)
    
    # 3. Handle Test Data Source
    if uploaded_test_file is not None:
        # Load external test file
        test_df = pd.read_csv(uploaded_test_file, names=COLUMNS, sep=r',\s*', 
                             engine='python', na_values='?', skiprows=1)
        test_df = test_df.dropna()
        X_test_raw = test_df.drop('income', axis=1)
        y_test = test_df['income'].apply(clean_income)
    else:
        # Internal Split scenario
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

    # 4. Encoding (One-Hot)
    # Combine to ensure identical columns after get_dummies
    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    categorical_cols = combined.select_dtypes(include=['object']).columns
    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    # Split back
    X_train = combined_encoded.iloc[:len(X_train_raw)]
    X_test = combined_encoded.iloc[len(X_train_raw):]

    # 5. Scaling (Required for KNN, LR, and NB)
    if scale_features:
        scaler = StandardScaler()
        cols = X_train.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)

    return X_train, X_test, y_train, y_test
