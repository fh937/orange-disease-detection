
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
h1, h2, h3, h4, p, label, span {
    color: white;
}
.stButton button {
    background-color: #16834a;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #20a35e;
    color: white;
}
div[data-testid="stMetric"] {
    background-color: #103525;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🍊 Orange Fruit Disease Detection")

st.write(
    "Upload an orange image to identify its main disease "
    "or check whether it is healthy."
)

# ---------------- MODEL SETTINGS ----------------

BASE_DIR = Path(__file__).parent

DISEASE_MODEL_PATH = BASE_DIR / "orange_disease_improved.keras"

CLASSES = [
    "Black spot",
    "Canker",
    "Greening",
    "Scab",
    "healthy"
]

ORANGE_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 70.0

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

    if disease_model.output_shape[-1] != len(CLASSES):
        st.error("Disease model has an unexpected number of output classes.")
        st.stop()

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

    if st.button("Detect Disease", use_container_width=True):

        with st.spinner("Checking image and detecting disease..."):

            # Resize and convert to RGB
            resized_image = image.resize((224, 224))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_batch = np.expand_dims(image_array, axis=0)

            # ---------------- STAGE 1: ORANGE VERIFICATION ----------------

            # ImageNet model requires MobileNetV2 preprocessing
            imagenet_input = preprocess_input(image_batch.copy())

            imagenet_predictions = imagenet_model.predict(
                imagenet_input,
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
                f"Orange verification score: "
                f"{orange_probability * 100:.2f}%"
            )

            if orange_probability < ORANGE_THRESHOLD:
                st.error(
                    "❌ This image could not be verified as an orange. "
                    "Please upload an orange fruit image."
                )
                st.stop()

            st.success("✅ Orange verification passed.")

            # ---------------- STAGE 2: DISEASE DETECTION ----------------

            # IMPORTANT:
            # This model contains its own preprocessing layers.
            # Pass raw RGB images (0–255). Do not use preprocess_input here.

            predictions = disease_model(
                image_batch,
                training=False
            ).numpy()[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASSES[predicted_index]

            confidence = float(predictions[predicted_index]) * 100

            # Rank all five classes
            ranked_indices = np.argsort(predictions)[::-1]

            # ---------------- MAIN RESULT ----------------

            st.markdown("---")
            st.subheader("🍊 Main Detection Result")

            if confidence < CONFIDENCE_THRESHOLD:

                st.warning(
                    f"⚠️ Uncertain result: {predicted_class}"
                )

                st.write(
                    "The model's top score is below the confidence "
                    "threshold. Treat this prediction cautiously."
                )

            elif predicted_class == "healthy":

                st.success("🍊 Main prediction: Healthy Orange")

            else:

                st.error(
                    f"⚠️ Main prediction: {predicted_class}"
                )

            st.metric(
                "Top prediction score",
                f"{confidence:.2f}%"
            )

            # ---------------- ALL CLASS SCORES ----------------

            st.markdown("---")
            st.subheader("📊 Scores for All Classes")

            for i in ranked_indices:

                class_name = CLASSES[i]
                score = float(predictions[i]) * 100

                st.write(f"**{class_name}: {score:.2f}%**")

                st.progress(
                    float(np.clip(score / 100, 0, 1))
                )

            # ---------------- ALTERNATIVE POSSIBILITIES ----------------

            st.markdown("---")
            st.subheader("🔍 Other Possible Results")

            alternatives = ranked_indices[1:]

            for rank, index in enumerate(alternatives, start=2):

                alternative_name = CLASSES[index]
                alternative_score = float(predictions[index]) * 100

                st.write(
                    f"**{rank}. {alternative_name}** — "
                    f"{alternative_score:.2f}%"
                )

            st.caption(
                "The model selects one main class. Other scores are "
                "alternative model outputs, not confirmed additional diseases."
            )

            st.info(
                "These scores are model outputs, not clinically calibrated "
                "probabilities. Results may be less reliable on unfamiliar "
                "images or mixed symptoms."
            )

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Orange Fruit Disease Detection | "
    "MobileNetV2 Deep Learning Project"
)
