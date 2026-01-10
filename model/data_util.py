import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(scale_features=False):
    """
    Loads Adult dataset, preprocesses it, and returns train-test split.
    
    Parameters:
    scale_features (bool): Whether to apply feature scaling

    Returns:
    X_train, X_test, y_train, y_test
    """

    np.random.seed(42)

    file = "data/adult.data"

    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]

    data = pd.read_csv(
        file,
        header=None,
        names=column_names,
        sep=r",\s*",
        engine="python"
    )

    # Handle missing values
    data = data.replace('?', pd.NA).dropna()

    X = data.drop('income', axis=1)
    y = data['income'].apply(lambda x: 0 if x == '<=50K' else 1)

    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if scale_features:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
