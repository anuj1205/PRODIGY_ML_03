# src/train.py

import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.model import build_svm_model
from src.config import MODEL_PATH, RANDOM_STATE


def train_model():
    X, y = load_data()

    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = build_svm_model()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
