import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODEL_PATH, TARGET, RANDOM_STATE
from src.data_prep import load_data, clean_data
from src.features import create_features


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()