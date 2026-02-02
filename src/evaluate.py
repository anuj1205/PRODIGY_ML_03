# src/evaluate.py

import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.config import MODEL_PATH, RANDOM_STATE


def evaluate_model():
    X, y = load_data()

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))


# 🔥 THIS WAS MISSING
if __name__ == "__main__":
    evaluate_model()
