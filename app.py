
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

st.set_page_config(page_title="Orange Disease Detection", page_icon="🍊")

@st.cache_resource
def load_models():
    verification = tf.keras.models.load_model(
        PROJECT_DIR / "orange_verification_model_final.keras"
    )
    disease = tf.keras.models.load_model(
        PROJECT_DIR / "orange_disease_finetuned.keras"
    )
    return verification, disease

verification_model, disease_model = load_models()

classes = ["Black spot", "Canker", "Greening", "Scab", "healthy"]

st.title("🍊 Orange Fruit Disease Detection")
st.write("Upload an orange image to check its health.")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Disease"):
        with st.spinner("Analyzing image..."):
            image_array = np.array(image.resize((224, 224)), dtype=np.float32)
            image_batch = np.expand_dims(image_array, axis=0)

            orange_probability = float(
                verification_model.predict(image_batch, verbose=0)[0][0]
            )

            st.write(f"Orange verification: {orange_probability * 100:.2f}%")

            if orange_probability < 0.5:
                st.error("Not an orange! Please upload an orange fruit image.")
            else:
                disease_input = preprocess_input(image_batch.copy())
                predictions = disease_model.predict(disease_input, verbose=0)[0]
                index = int(np.argmax(predictions))
                predicted_class = classes[index]
                confidence = float(predictions[index])

                if predicted_class == "healthy":
                    st.success("Your orange appears healthy!")
                else:
                    st.warning(f"Predicted disease: {predicted_class}")

                st.metric("Prediction confidence", f"{confidence * 100:.2f}%")
                st.subheader("Class probabilities")
                st.bar_chart({
                    classes[i]: float(predictions[i])
                    for i in range(len(classes))
                })

st.caption("Research prototype. Verify predictions with expert assessment.")
