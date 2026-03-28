import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import requests
import os

IMG_SIZE = 224
CLASSES = ["Healthy", "Rust", "Other"]

# ✅ FIXED GOOGLE DRIVE LINK
MODEL_URL = "https://drive.google.com/uc?id=1v6VRvMKoIB1vRGz4Spt4yqZUr_KObuJ6"
MODEL_PATH = "crop_model.h5"

# ---------------- DOWNLOAD MODEL ---------------- #
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL, timeout=60)

            # ✅ check if download failed
            if r.status_code != 200:
                st.error("Failed to download model")
                return

            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- PREPROCESS ---------------- #
def preprocess(image):
    img = image.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ---------------- #
st.title("🌱 Crop Disease Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg","tif"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):
        img = preprocess(image)
        pred = model.predict(img)

        st.success(f"Prediction: {CLASSES[np.argmax(pred)]}")
        st.info(f"Confidence: {float(np.max(pred)):.2f}")
