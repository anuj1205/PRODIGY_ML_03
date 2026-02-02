# src/data_loader.py

import os
import numpy as np
from tqdm import tqdm

from src.config import DATA_DIR, IMAGE_SIZE, DATA_LIMIT
from src.feature_extraction import extract_hog_features


def load_data():
    X, y = [], []

    files = os.listdir(DATA_DIR)[:DATA_LIMIT]

    for file in tqdm(files, desc="Loading images"):
        path = os.path.join(DATA_DIR, file)

        if file.startswith("cat"):
            label = 0
        elif file.startswith("dog"):
            label = 1
        else:
            continue

        features = extract_hog_features(path, IMAGE_SIZE)
        if features is not None:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)
