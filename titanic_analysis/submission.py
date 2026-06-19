import pandas as pd

from .config import FEATURES, SUBMISSION_PATH


def build_submission(model, test_df):
    X_final = test_df[FEATURES]
    y_pred_final = model.predict(X_final)
    return pd.DataFrame(
        {
            "PassengerId": test_df.PassengerId,
            "Survived": y_pred_final,
        }
    )


def save_submission(submission_df, path=SUBMISSION_PATH):
    submission_df.to_csv(path, index=False)
    return path

