# data_util.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

def load_training_data(train_path="adult.data"):
    df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, na_values=" ?")
    df.dropna(inplace=True)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df


def load_test_data(uploaded_file):
    df = pd.read_csv(uploaded_file, header=0, names=COLUMN_NAMES, na_values=" ?")
    df.dropna(inplace=True)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df


def preprocess(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    categorical_cols = train_df.select_dtypes(include=["object"]).columns

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le

    X_train = train_df.drop("income", axis=1)
    y_train = train_df["income"]

    X_test = test_df.drop("income", axis=1)
    y_test = test_df["income"]

    return X_train, X_test, y_train, y_test
