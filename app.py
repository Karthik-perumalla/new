import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


IMG_SIZE = 224  # change if different
CLASSES = ["Healthy", "Disease1", "Disease2"]  # replace with your classes
MODEL_PATH = "model.h5"  # or your saved model path


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


def preprocess(image):
    img = image.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ---------------- #

st.set_page_config(page_title="Crop Disease Detection", layout="centered")

st.title("Crop Disease Detection")
st.write("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            with st.spinner("Predicting..."):
                img = preprocess(image)
                pred = model.predict(img)

                class_idx = np.argmax(pred)
                confidence = float(np.max(pred))

            st.success(f"Prediction: {CLASSES[class_idx]}")
            st.info(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")