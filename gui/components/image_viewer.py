"""
Image viewing and display utilities.
"""

import streamlit as st
from PIL import Image
import numpy as np


def display_image_with_prediction(image_path: str, prediction: Dict, confidence: float):
    """
    Display image alongside prediction results.

    Args:
        image_path: Path to image file
        prediction: Dict with prediction details
        confidence: Confidence score (0-1)
    """
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(image_path)
        st.image(image, caption="Input Image", use_column_width=True)

    with col2:
        st.subheader("Prediction Results")
        st.metric("Predicted Class", prediction.get("class", "N/A"))
        st.metric("Confidence", f"{confidence:.2%}")

        if "details" in prediction:
            st.write("**Details:**")
            for key, val in prediction["details"].items():
                st.write(f"  - {key}: {val}")
