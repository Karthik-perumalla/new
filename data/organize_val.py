import os
import random
from src.prediction import predict_image

VAL_PATH = r"E:\project_NareshIT\data\raw\03_Wheat_Disease\val\RGB"

img = random.choice(os.listdir(VAL_PATH))
img_path = os.path.join(VAL_PATH, img)

print("Random Image:", img_path)

pred = predict_image(img_path)

print("Prediction:", pred)