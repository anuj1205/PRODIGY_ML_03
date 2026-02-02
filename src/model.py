# src/model.py

from sklearn.svm import SVC


def build_svm_model():
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )
    return model
