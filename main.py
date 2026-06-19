from titanic_analysis.data import load_test_data, load_training_data, prepare_test_data
from titanic_analysis.modeling import choose_final_model, evaluate_models
from titanic_analysis.submission import build_submission, save_submission


def main():
    titanic_df = load_training_data()

    results = evaluate_models(titanic_df)
    print(results["model1_accuracy"])
    print(results["model1_score"])
    print(results["model2_accuracy"])
    print(results["model2_score"])

    model_final, age_means = choose_final_model(results)
    titanic_df_test = load_test_data()
    titanic_df_test = prepare_test_data(titanic_df_test, titanic_df, age_means)
    submission = build_submission(model_final, titanic_df_test)
    save_submission(submission)


if __name__ == "__main__":
    main()
