import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import gdown

IMG_SIZE = 224
CLASSES = ["Healthy", "Rust", "Other"]

# Google Drive direct download (works with gdown)
MODEL_URL = "https://drive.google.com/uc?id=1v6VRvMKoIB1vRGz4Spt4yqZUr_KObuJ6"

# Save model inside a folder
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.h5")

# ---------------- DOWNLOAD MODEL ---------------- #
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model using gdown..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"gdown failed: {e}")
                return False

    # Validate file
    if os.path.getsize(MODEL_PATH) < 500000:  # too small → HTML file
        st.error("Downloaded file is too small. Google Drive returned HTML instead of model.")
        return False

    return True

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    ok = download_model()
    if not ok:
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

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
        if model is None:
            st.error("Model not loaded. Fix download issues above.")
        else:
            img = preprocess(image)
            pred = model.predict(img)

            st.success(f"Prediction: {CLASSES[np.argmax(pred)]}")
            st.info(f"Confidence: {float(np.max(pred)):.2f}")
