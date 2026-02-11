import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week',
    'native-country', 'income'
]

def map_income(x):
    x = str(x).strip()
    if x in ['<=50K', '50K', '<=50K.']:
        return 0
    elif x in ['>50K', '>50K.']:
        return 1
    else:
        raise ValueError(f"Unknown income value: {x}")

def preprocess_data(data1, feature_columns=None, has_label=True):
    data1 = data1.replace('?', pd.NA).dropna()
    
    if has_label:
        y = data1['income'].apply(map_income)
        X = data1.drop(columns=['income'], errors='ignore')
    else:
        X = data1.copy()
        y = None

    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    if feature_columns is not None:
        X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    return X_encoded, y

def load_train_and_test_data(scale_features=False, uploaded_test_file=None):
    import numpy as np
    data = pd.read_csv("data/adult.data", header=None, names=column_names, sep=r",\s*", engine="python")
    data = data.replace('?', pd.NA).dropna()

    if uploaded_test_file is None:
        X = data.drop('income', axis=1)
        #y = data['income'].apply(lambda x: 0 if str(x).strip() in ['<=50K','50K'] else 1)
        y = data['income'].apply(map_income)
        X_encoded = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    else:
        uploaded_test_file.columns = uploaded_test_file.columns.str.strip()
        X_train, y_train = preprocess_data(data)
        X_test, y_test = preprocess_data(uploaded_test_file, feature_columns=X_train.columns, has_label=True)

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
