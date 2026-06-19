import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import FEATURES
from .data import prepare_test_data, prepare_training_data


def build_random_forest_model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )


def build_logistic_regression_model():
    return LogisticRegression(solver="lbfgs", C=0.5)


def prepare_train_validation_split(train_df, validation_df):
    prepared_train_df, age_means = prepare_training_data(train_df)
    prepared_validation_df = prepare_test_data(validation_df, prepared_train_df, age_means)
    return prepared_train_df, prepared_validation_df


def cross_validate_without_leakage(model_factory, df, cv=5):
    X = df[FEATURES]
    y = df["Survived"]
    splitter = StratifiedKFold(n_splits=cv)
    scores = []

    for train_index, validation_index in splitter.split(X, y):
        train_df = df.iloc[train_index]
        validation_df = df.iloc[validation_index]
        prepared_train_df, prepared_validation_df = prepare_train_validation_split(
            train_df,
            validation_df,
        )

        model = model_factory()
        model.fit(prepared_train_df[FEATURES], prepared_train_df["Survived"])
        predictions = model.predict(prepared_validation_df[FEATURES])
        scores.append(accuracy_score(prepared_validation_df["Survived"], predictions))

    return np.array(scores)


def evaluate_models(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=64,
    )
    prepared_train_df, prepared_test_df = prepare_train_validation_split(train_df, test_df)

    train_X = prepared_train_df[FEATURES]
    train_y = prepared_train_df["Survived"]
    test_X = prepared_test_df[FEATURES]
    test_y = prepared_test_df["Survived"]

    model1 = build_random_forest_model()
    model1.fit(train_X, train_y)
    y_pred = model1.predict(test_X)
    model1_accuracy = accuracy_score(test_y, y_pred)
    model1_score = cross_validate_without_leakage(build_random_forest_model, df)

    model2 = build_logistic_regression_model()
    model2.fit(train_X, train_y)
    y2_pred = model2.predict(test_X)
    model2_accuracy = accuracy_score(test_y, y2_pred)
    model2_score = cross_validate_without_leakage(build_logistic_regression_model, df)

    return {
        "df": df,
        "model1": model1,
        "model1_accuracy": model1_accuracy,
        "model1_score": model1_score,
        "model2": model2,
        "model2_accuracy": model2_accuracy,
        "model2_score": model2_score,
    }


def choose_final_model(results):
    prepared_df, age_means = prepare_training_data(results["df"])
    X = prepared_df[FEATURES]
    y = prepared_df["Survived"]

    model_final = build_random_forest_model()
    if results["model2_score"].mean() > results["model1_score"].mean():
        model_final = build_logistic_regression_model()
    model_final.fit(X, y)
    return model_final, age_means
