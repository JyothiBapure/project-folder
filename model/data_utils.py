# data_util.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]


def load_training_data(train_path="data/adult.data"):
    df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, na_values=" ?")
    df.dropna(inplace=True)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df


def load_test_data(uploaded_file):
    df = pd.read_csv(uploaded_file, header=0, names=COLUMN_NAMES, na_values=" ?")
    df.dropna(inplace=True)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df


def encode_data(train_df, test_df=None):

    train_df = train_df.copy()

    categorical_cols = train_df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        label_encoders[col] = le

    if test_df is not None:
        test_df = test_df.copy()
        for col in categorical_cols:
            test_df[col] = label_encoders[col].transform(test_df[col])

    return train_df, test_df


def get_data(uploaded_file=None):

    train_df = load_training_data()

    if uploaded_file is not None:
        # External test file mode
        test_df = load_test_data(uploaded_file)

        train_df, test_df = encode_data(train_df, test_df)

        X_train = train_df.drop("income", axis=1)
        y_train = train_df["income"]

        X_test = test_df.drop("income", axis=1)
        y_test = test_df["income"]

    else:
        # Internal split mode
        train_df, _ = encode_data(train_df)

        X = train_df.drop("income", axis=1)
        y = train_df["income"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    return X_train, X_test, y_train, y_test
