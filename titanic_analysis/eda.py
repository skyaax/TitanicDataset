def class_survival_sizes(df, pclass):
    class_df = df[df["Pclass"] == pclass]
    survivors_df = class_df[class_df["Survived"] == 1]
    return [len(class_df.index) - len(survivors_df.index), len(survivors_df.index)]


def class_survivors(df, pclass):
    return df[(df["Pclass"] == pclass) & (df["Survived"] == 1)]


def women_and_children_survivor_ratio(survivors_df):
    men_survivors = survivors_df[survivors_df["Sex"] == "male"]
    women_and_children = survivors_df[
        (survivors_df["Age"] < 18) | (survivors_df["Sex"] == "female")
    ]
    return [len(men_survivors.index), len(women_and_children.index)]


def total_survival_sizes(df):
    survivors = df[df["Survived"] == 1]
    return [len(df.index) - len(survivors.index), len(survivors.index)]

