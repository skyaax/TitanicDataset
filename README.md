# 🚢 Titanic Data Analysis & Survival Prediction

This project focuses on a comprehensive Exploratory Data Analysis (EDA) of the Titanic passenger dataset and the implementation of machine learning models to predict passenger survival. The project is structured within a Jupyter Notebook, placing a unique emphasis on evaluating social hypotheses and ensuring clean data preprocessing that prevents data leakage[cite: 1, 2].

## 🎯 Project Objective

Beyond standard survival prediction, the main goal of the EDA was to test a historical stereotype[cite: 1, 2]:
> *"For ages people were always saying that men from the 1st class were not saving children and women, unless men from other classes, which are considered more poor and more kind. And the aim of this EDA was to make the research and see which people are more 'kind'"*[cite: 1, 2]

To evaluate this, the analysis examines the survival rates of adult men in the 1st class relative to women and children of the same class, comparing these patterns against the 3rd class to uncover the actual behavioral trends during the evacuation[cite: 1, 2].

## 📊 Exploratory Data Analysis (EDA) Highlights

* **Class-Based Survival Rates (Pclass):** Investigated the overall percentage of survivors within each passenger class to establish an objective baseline[cite: 1, 2]. 
* **Age and Gender Analysis:** Visualized the demographic structure of the ship using age distribution histograms and evaluated survival distribution by age[cite: 1, 2].
* **The "Chivalry" Hypothesis:** Developed pie charts illustrating the ratio of male survivors to women and children survivors within the 1st and 3rd classes to check adherence to the "women and children first" protocol[cite: 1, 2].

## 🛠️ Machine Learning & Architecture

Two distinct machine learning models from the `scikit-learn` library were built and cross-validated[cite: 1, 2]:
1. **Random Forest Classifier**[cite: 1, 2]
2. **Logistic Regression**[cite: 1, 2]

### Features Used for Training:
The models utilize the following features extracted from the dataset[cite: 1, 2]:
* `Pclass` — Passenger class (1, 2, 3)[cite: 1, 2]
* `Age` — Passenger age[cite: 1, 2]
* `Sex` — Gender (mapped to binary values: `male: 0`, `female: 1`)[cite: 1, 2]
* `Fare` — Ticket fare[cite: 1, 2]
* `Parch` — Number of parents/children aboard[cite: 1, 2]
* `SibSp` — Number of siblings/spouses aboard[cite: 1, 2]

### 🛡️ Preventing Data Leakage
To maintain a strict division between training and validation data, missing values in the `Age` column were handled without data leakage[cite: 1, 2]:
* The data was split into training and testing sets *before* calculating the missing value replacements[cite: 1, 2].
* The mean age was computed **separately for each Pclass** using only the training split (`train_X`)[cite: 2].
* These training-derived means were then used to fill missing values in both the training and test sets (`train_X` and `test_X`), ensuring no future data leaked into the model training phase[cite: 2].

## 📈 Performance & Evaluation

* **Strongest Predictors:** Gender (`Sex`) and passenger class (`Pclass`) proved to be the most critical indicators for survival.
* **Model Comparison:** The `Random Forest` model demonstrated a strong performance, achieving an accuracy score of **~0.832** on the validation test split[cite: 2]. 
* **Final Submission:** The script automatically evaluates the cross-validation scores (`cross_val_score`) for both models and uses the superior model to fit the entire dataset and generate the final `submission_final_newww2.csv` file[cite: 2].
