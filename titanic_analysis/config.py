from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "Titanic-Dataset.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission_final_newww2.csv"

FEATURES = [
    "Pclass",
    "Age",
    "Sex",
    "Fare",
    "Parch",
    "SibSp",
]

SEX_MAPPING = {"male": 0, "female": 1}

