import pandas as pd

def inspect_metadata(df):
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict()
    }


def get_column_stats(df, col):
    if df[col].dtype == "object":
        return df[col].value_counts().head(10).to_dict()

    return df[col].describe().to_dict()


def impute_missing(df, col, strategy):

    if strategy == "mean":
        df[col].fillna(df[col].mean(), inplace=True)

    elif strategy == "median":
        df[col].fillna(df[col].median(), inplace=True)

    elif strategy == "mode":
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def drop_column(df, col):
    return df.drop(columns=[col])


def encode_categorical(df, col):
    return pd.get_dummies(df, columns=[col], drop_first=True)


def create_interaction(df, expression):
    exec(expression)
    return df