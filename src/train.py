import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PATH, MODEL_PATH, TARGET, RANDOM_STATE
from src.data_prep import load_data, clean_data
from src.features import create_features


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_importance.png", bbox_inches="tight")
    plt.close()

    print(f"Training complete. ROC-AUC: {auc:.4f}")
    print("Saved model to outputs/models/credit_risk_model.pkl")
    print("Saved feature importance plot to outputs/figures/feature_importance.png")


if __name__ == "__main__":
    main()