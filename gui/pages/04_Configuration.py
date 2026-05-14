"""
Configuration page: Model selection and inference settings.
"""

import streamlit as st
from pathlib import Path
import json

st.title("⚙️ Configuration")

st.markdown("## Model Selection")

baseline_dir = Path("data_set/baseline")

# Select channel
channel = st.selectbox("Select Channel", ["C", "K", "M", "Y"])

# Select checkpoint
checkpoint_file = baseline_dir / f"best_{channel}.pt"

if checkpoint_file.exists():
    st.success(f"✅ Checkpoint found: `{checkpoint_file}`")
else:
    st.error(f"❌ Checkpoint not found for channel {channel}")

st.markdown("---")
st.markdown("## Inference Settings")

batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
device = st.selectbox("Device", ["CPU", "GPU (CUDA)"])

st.markdown("---")
st.markdown("## Load Custom Config")

config_file = st.file_uploader("Upload config.json", type=["json"])

if config_file is not None:
    config = json.loads(config_file.read())
    st.write("**Loaded Configuration:**")
    st.json(config)

st.info("💡 Modify settings above and save to apply custom configuration.")
