import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week',
    'native-country', 'income'
]

def preprocess_data(data1, feature_columns=None):
    """
    Preprocess dataframe and align columns.
    """
    data1 = data1.replace('?', pd.NA).dropna()

    X = data1.drop('income', axis=1)
    y = data1['income'].apply(lambda x: 0 if x == '<=50K' else 1)

    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    # Align test features with train features
    if feature_columns is not None:
        X_encoded = X_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return X_encoded, y


def load_train_and_test_data(
    scale_features=False,
    uploaded_test_file=None
):
    """
    Loads training data and optionally uses uploaded test data.
    """

    np.random.seed(42)
    file_path = "data/adult.data"

    data = pd.read_csv(file_path, header=None, names=column_names, sep=r",\s*", engine="python")

    data = data.replace('?', pd.NA).dropna()

    if uploaded_test_file is None:
        # Normal train-test split
        X = data.drop('income', axis=1)
        y = data['income'].apply(lambda x: 0 if x == '<=50K' else 1)

        # Identify categorical columns
        categorical_cols = X.select_dtypes(include='object').columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    else:
        # Use uploaded test data
        X_train, y_train = preprocess_data(data)
        X_test, y_test = preprocess_data(
            uploaded_test_file,
            feature_columns=X_train.columns
        )

    if scale_features:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
