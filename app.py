# Stage 1: Verify orange using ImageNet
imagenet_predictions = imagenet_model.predict(
    processed_image, verbose=0
)

imagenet_results = decode_predictions(
    imagenet_predictions, top=1000
)[0]

orange_probability = next(
    (
        score for _, label, score in imagenet_results
        if label == "orange"
    ),
    0.0
)

if orange_probability < 0.50:
    st.error("This image does not appear to be an orange.")
    st.stop()

st.success(
    f"Orange verified: {orange_probability * 100:.2f}%"
)

# Stage 2: Predict orange disease
predictions = disease_model.predict(
    processed_image, verbose=0
)

predicted_index = np.argmax(predictions[0])
predicted_class = CLASSES[predicted_index]
confidence = float(predictions[0][predicted_index]) * 100
