import matplotlib.pyplot as plt
import seaborn as sns

from .eda import (
    class_survival_sizes,
    class_survivors,
    total_survival_sizes,
    women_and_children_survivor_ratio,
)


def _class_label(pclass):
    suffixes = {1: "1st", 2: "2nd", 3: "3rd"}
    return suffixes.get(pclass, str(pclass))


def plot_class_survival(df, pclass):
    sizes = class_survival_sizes(df, pclass)
    plt.figure(facecolor="grey")
    plt.pie(
        sizes,
        labels=["Died", "Survived"],
        autopct="%1.1f%%",
        shadow=True,
        textprops={"color": "black"},
    )
    plt.title(f"Percentage of {_class_label(pclass)} class survivers against those who died")
    plt.show()


def plot_women_children_ratio(df, pclass):
    survivors = class_survivors(df, pclass)
    ratio = women_and_children_survivor_ratio(survivors)
    plt.figure(facecolor="grey")
    plt.pie(
        ratio,
        labels=["Men survivers", "Women and Children survivers"],
        autopct="%1.1f%%",
        textprops={"color": "black"},
        shadow=True,
    )
    plt.title(
        f"Percentage of {_class_label(pclass)} class man survivers against children and women survivers"
    )
    plt.show()


def plot_age_distribution(df):
    plt.hist(df["Age"], bins=20)
    plt.title("Age distribution on the Titanic")
    plt.grid(False)
    plt.show()


def plot_survivor_age_distribution(df):
    plt.hist(df.loc[df["Survived"] == 1, "Age"], bins=20)
    plt.title("Age distribution of survivers on the Titanic")
    plt.grid(False)
    plt.show()


def plot_total_survival(df):
    ratio_total = total_survival_sizes(df)
    plt.figure(facecolor="grey")
    plt.pie(
        ratio_total,
        labels=["Died", "Survived"],
        autopct="%1.1f%%",
        textprops={"color": "black"},
        shadow=True,
    )
    plt.title("Distribution of people who died against survivers")
    plt.show()


def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()
