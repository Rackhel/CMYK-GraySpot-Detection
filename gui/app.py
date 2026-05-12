"""
gui/app.py

Main Streamlit application entry point for Grayspot Classification System.

Provides multi-page interface:
  - Dashboard: Model performance summary & metrics
  - Inference: Upload image and get predictions
  - Model Info: Architecture, training history
  - Configuration: Model selection, settings
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="Grayspot Classification System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🎨 Grayspot Defect System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Inference", "Model Info", "Configuration"],
    icons=["📊", "🔍", "🧠", "⚙️"],
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **Grayspot Defect Classification**
    
    CMYK Printer Defect Detection using Deep Learning
    
    [GitHub](https://github.com/) | [Docs](/)
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Page Routing
# ─────────────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("📊 Dashboard")
    st.write("Model performance summary, metrics, and evaluation results.")
    # TODO: Import and run pages.dashboard module

elif page == "Inference":
    st.title("🔍 Inference")
    st.write("Upload images and get defect predictions.")
    # TODO: Import and run pages.inference module

elif page == "Model Info":
    st.title("🧠 Model Architecture & Training History")
    st.write("Model architecture details, training curves, and checkpoint info.")
    # TODO: Import and run pages.model_info module

elif page == "Configuration":
    st.title("⚙️ Configuration")
    st.write("Select model checkpoint, adjust inference settings.")
    # TODO: Import and run pages.configuration module

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.8em; color: gray;'>"
    "Grayspot v1.0.0 | 2026 CMYK Team</div>",
    unsafe_allow_html=True,
)
