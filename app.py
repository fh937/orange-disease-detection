
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Orange Disease Detection",
    page_icon="🍊",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
}

.main-title {
    text-align: center;
    color: #1b5e20;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #33691e;
    font-size: 18px;
}

.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2e7d32;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# MODEL CONFIGURATION
# -----------------------------------

DISEASE_MODEL_PATH = "orange_disease_finetuned.keras"

VERIFICATION_MODEL_PATH = "orange_verification_fruits360.keras"

CLASSES = [
    "Black spot",
    "Canker",
    "Greening",
    "Scab",
    "healthy"
]

# -----------------------------------
# LOAD MODELS
# -----------------------------------

@st.cache_resource
def load_models():

    verification_model = tf.keras.models.load_model(
        VERIFICATION_MODEL_PATH
    )

    disease_model = tf.keras.models.load_model(
        DISEASE_MODEL_PATH
    )

    return verification_model, disease_model


# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = np.expand_dims(image, axis=0)

    return image


# -----------------------------------
# MAIN APPLICATION
# -----------------------------------

st.markdown(
    '<div class="main-title">🍊 Orange Fruit Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Orange Disease Classification</div>',
    unsafe_allow_html=True
)

st.write("Upload an orange image to identify its disease.")

try:
    verification_model, disease_model = load_models()

except Exception as e:
    st.error("Unable to load the models.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    processed_image = preprocess_image(image)

    # -----------------------------------
    # STEP 1: VERIFY ORANGE
    # -----------------------------------

    st.subheader("Image Verification")

    orange_probability = float(
        verification_model.predict(
            processed_image,
            verbose=0
        )[0][0]
    )

    st.metric(
        "Orange Probability",
        f"{orange_probability * 100:.2f}%"
    )

    # Conservative threshold to reject uncertain images
    if orange_probability < 0.70:

        st.error(
            "This image could not be confidently verified as an orange."
        )

        st.warning(
            "Please upload a clear image of an orange fruit."
        )

        st.stop()

    st.success("Orange image detected!")

    # -----------------------------------
    # STEP 2: DISEASE CLASSIFICATION
    # -----------------------------------

    st.subheader("Disease Classification")

    predictions = disease_model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASSES[predicted_index]

    confidence = float(predictions[predicted_index])

    st.markdown(
        f"""
        <div class="result-box">
            <h3>Prediction: {predicted_class}</h3>
            <h4>Confidence: {confidence * 100:.2f}%</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Class-wise Probabilities")

    for i, class_name in enumerate(CLASSES):

        st.write(
            f"{class_name}: {predictions[i] * 100:.2f}%"
        )

        st.progress(float(predictions[i]))

    st.info(
        "Note: Predictions are AI-generated and should be "
        "verified with expert assessment when needed."
    )
