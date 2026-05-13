"""
Model Info page: Architecture, training history, and checkpoint details.
"""

import streamlit as st
from pathlib import Path
import json

st.title("🧠 Model Architecture & Training History")

st.markdown("## Model Architecture")
st.info("TODO: Load and display model architecture from config")

st.markdown("---")
st.markdown("## Training History")

# Load training history
baseline_dir = Path("data_set/baseline")

channel = st.selectbox("Select Channel", ["C", "K", "M", "Y"])

history_file = baseline_dir / f"phase2_history_{channel}.csv"

if history_file.exists():
    import pandas as pd

    df = pd.read_csv(history_file)

    st.write(f"**Training History - Channel {channel}**")
    st.line_chart(df.set_index(df.columns[0]))

    st.write("**Raw Data:**")
    st.dataframe(df)
else:
    st.warning(f"No training history found for channel {channel}")

st.markdown("---")
st.markdown("## Checkpoints")

st.info("TODO: List available model checkpoints and metadata")
