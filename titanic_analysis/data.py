import pandas as pd

from .config import SEX_MAPPING, TEST_DATA_PATH, TRAIN_DATA_PATH


def load_training_data(path=TRAIN_DATA_PATH):
    return pd.read_csv(path)


def load_test_data(path=TEST_DATA_PATH):
    return pd.read_csv(path)


def compute_age_means_by_class(df):
    return {
        int(pclass): int(df.loc[df["Pclass"] == pclass, "Age"].mean())
        for pclass in sorted(df["Pclass"].dropna().unique())
    }


def fill_age_by_class(df, age_means):
    prepared_df = df.copy()
    for pclass, mean_age in age_means.items():
        mask = prepared_df["Pclass"] == pclass
        prepared_df.loc[mask, "Age"] = prepared_df.loc[mask, "Age"].fillna(mean_age)
    return prepared_df


def encode_sex(df):
    prepared_df = df.copy()
    prepared_df["Sex"] = prepared_df["Sex"].map(SEX_MAPPING).astype(int)
    return prepared_df


def prepare_training_data(df):
    age_means = compute_age_means_by_class(df)
    prepared_df = fill_age_by_class(df, age_means)
    prepared_df = encode_sex(prepared_df)
    return prepared_df, age_means


def prepare_test_data(df, train_df, age_means):
    prepared_df = fill_age_by_class(df, age_means)
    prepared_df = encode_sex(prepared_df)
    prepared_df["Fare"] = prepared_df["Fare"].fillna(train_df["Fare"].mean())
    return prepared_df
