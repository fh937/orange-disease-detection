
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Orange Disease Detection",
    page_icon="🍊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #071b12;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton button {
    background-color: #16834a;
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton button:hover {
    background-color: #20a35e;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🍊 Orange Fruit Disease Detection")
st.write(
    "Upload an orange fruit image to detect whether it is "
    "healthy or affected by a disease."
)

# ---------------- MODEL PATHS ----------------

BASE_DIR = Path(__file__).parent

DISEASE_MODEL_PATH = BASE_DIR / "orange_disease_finetuned.keras"

CLASSES = [
    "Black spot",
    "Canker",
    "Greening",
    "Scab",
    "healthy"
]

# Temporary threshold: accepts your tested orange at 1.80%
ORANGE_THRESHOLD = 0.01

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(
        DISEASE_MODEL_PATH,
        compile=False
    )

@st.cache_resource
def load_imagenet_model():
    return MobileNetV2(weights="imagenet")

try:
    disease_model = load_disease_model()
    imagenet_model = load_imagenet_model()

except Exception as e:
    st.error("Could not load the models.")
    st.exception(e)
    st.stop()

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Choose an orange image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    except Exception:
        st.error("Unable to read this image.")
        st.stop()

    if st.button("Detect Disease"):

        with st.spinner("Checking image and detecting disease..."):

            # Resize image
            resized_image = image.resize((224, 224))

            # Convert to array
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # MobileNetV2 preprocessing
            processed_image = preprocess_input(image_array)

            # ---------------- STAGE 1: ORANGE CHECK ----------------

            imagenet_predictions = imagenet_model.predict(
                processed_image,
                verbose=0
            )

            imagenet_results = decode_predictions(
                imagenet_predictions,
                top=1000
            )[0]

            orange_probability = next(
                (
                    float(score)
                    for _, label, score in imagenet_results
                    if label == "orange"
                ),
                0.0
            )

            st.write(
                f"ImageNet orange score: "
                f"{orange_probability * 100:.2f}%"
            )

            if orange_probability < ORANGE_THRESHOLD:

                st.error(
                    "❌ This image could not be verified as an orange. "
                    "Please upload an orange fruit image."
                )
                st.stop()

            st.success("✅ Orange check passed!")

            # ---------------- STAGE 2: DISEASE DETECTION ----------------

            predictions = disease_model.predict(
                processed_image,
                verbose=0
            )

            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = CLASSES[predicted_index]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

            # ---------------- DISPLAY RESULT ----------------

            st.subheader("🍊 Detection Result")

            if predicted_class.lower() == "healthy":

                st.success("🍊 Prediction: Healthy Orange")

            else:

                st.error(
                    f"⚠️ Disease Detected: {predicted_class}"
                )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            # ---------------- ALL CLASS PROBABILITIES ----------------

            st.subheader("Class Probabilities")

            for i, class_name in enumerate(CLASSES):

                probability = float(
                    predictions[0][i]
                ) * 100

                st.write(
                    f"**{class_name}: {probability:.2f}%**"
                )

                st.progress(
                    min(max(probability / 100, 0.0), 1.0)
                )

st.markdown("---")
st.caption(
    "Orange Fruit Disease Detection | "
    "MobileNetV2 Deep Learning Model"
)
