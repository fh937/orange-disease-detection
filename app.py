
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Orange Disease Detection",
    page_icon="🍊",
    layout="wide"
)

# ==========================================
# 2. CUSTOM DARK GREEN THEME
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #10251b, #183c2b);
    color: #ffffff;
}

/* Main text */
.stApp p,
.stApp label,
.stApp span,
.stMarkdown,
.stMetric label,
.stMetric [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Title */
.main-title {
    text-align: center;
    color: #a5d6a7 !important;
    font-size: 38px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #c8e6c9 !important;
    font-size: 18px;
}

/* Result box */
.result-box {
    background-color: #244936;
    color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #81c784;
    margin-top: 15px;
}

.result-box h3,
.result-box h4 {
    color: #ffffff !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #244936;
    padding: 15px;
    border-radius: 10px;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #81c784;
}

/* Headings */
h1, h2, h3 {
    color: #a5d6a7 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================

DISEASE_MODEL_PATH = "orange_disease_finetuned.keras"

VERIFICATION_MODEL_PATH = "orange_verification_fruits360.keras"

CLASSES = [
    "Black spot",
    "Canker",
    "Greening",
    "Scab",
    "healthy"
]

ORANGE_THRESHOLD = 0.70

# ==========================================
# 4. LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    verification_model = tf.keras.models.load_model(
        VERIFICATION_MODEL_PATH
    )

    disease_model = tf.keras.models.load_model(
        DISEASE_MODEL_PATH
    )

    return verification_model, disease_model


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = np.expand_dims(image, axis=0)

    return image


# ==========================================
# 6. APPLICATION TITLE
# ==========================================

st.markdown(
    '<div class="main-title">🍊 Orange Fruit Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Orange Disease Classification</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload an orange fruit image to identify its disease."
)

st.divider()

# ==========================================
# 7. LOAD BOTH MODELS
# ==========================================

try:

    verification_model, disease_model = load_models()

except Exception as e:

    st.error("Unable to load the models.")

    st.exception(e)

    st.stop()


# ==========================================
# 8. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception:

        st.error("Unable to read this image. Please upload another image.")

        st.stop()

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    processed_image = preprocess_image(image)

    # ==========================================
    # 9. ORANGE IMAGE VERIFICATION
    # ==========================================

    st.subheader("🔍 Image Verification")

    orange_probability = float(
        verification_model.predict(
            processed_image,
            verbose=0
        )[0][0]
    )

    st.metric(
        label="Orange Probability",
        value=f"{orange_probability * 100:.2f}%"
    )

    st.progress(
        max(0.0, min(1.0, orange_probability))
    )

    if orange_probability < ORANGE_THRESHOLD:

        st.error(
            "This image could not be confidently verified as an orange."
        )

        st.warning(
            "Please upload a clear image of an orange fruit."
        )

        st.stop()

    st.success("Orange image detected successfully!")

    # ==========================================
    # 10. DISEASE CLASSIFICATION
    # ==========================================

    st.subheader("🧪 Disease Classification")

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

    # ==========================================
    # 11. CLASS-WISE PROBABILITIES
    # ==========================================

    st.subheader("📊 Class-wise Probabilities")

    for i, class_name in enumerate(CLASSES):

        probability = float(predictions[i])

        st.write(
            f"**{class_name}: {probability * 100:.2f}%**"
        )

        st.progress(
            max(0.0, min(1.0, probability))
        )

    # ==========================================
    # 12. DISCLAIMER
    # ==========================================

    st.info(
        "Note: Predictions are AI-generated and should "
        "be verified with expert assessment when needed."
    )

st.divider()

st.caption(
    "Orange Fruit Disease Detection | Deep Learning Project"
)
