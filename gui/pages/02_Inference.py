"""
Inference page: Upload images and get predictions.
"""

import streamlit as st
from PIL import Image
import numpy as np

st.title("🔍 Inference")

st.markdown("## Image Classification")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Prediction")

        # TODO: Load model and run inference
        # prediction = model.predict(image)

        st.info("TODO: Integrate with inference pipeline")

        # Placeholder
        st.metric("Predicted Class", "Y - Level 2")
        st.metric("Confidence", "87.3%")

st.markdown("---")
st.markdown("### Settings")

channel = st.selectbox("Select Channel", ["C", "K", "M", "Y"])
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

st.write(f"Predictions below {confidence_threshold:.1%} will be marked uncertain.")
