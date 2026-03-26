import os
import joblib
import shap
import matplotlib.pyplot as plt

from src.config import DATA_PATH, MODEL_PATH, TARGET
from src.data_prep import load_data, clean_data
from src.features import create_features


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET])

    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary.png", bbox_inches="tight")
    plt.close()

    print("Saved SHAP summary plot to outputs/figures/shap_summary.png")


if __name__ == "__main__":
    main()