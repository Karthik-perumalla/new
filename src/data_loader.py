# src/data_loader.py

import sys
import os

# Fix module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from config.settings import IMG_SIZE

def load_data(data_path):
    X, y = [], []

    classes = sorted(os.listdir(data_path))
    print("Classes found:", classes)

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            filename = file.lower()

            if filename.endswith(".png") or filename.endswith(".tif"):

                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)

                if img is None:
                    print("Skipped:", img_path)
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                X.append(img)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Loaded:", X.shape, y.shape)

    return X, y