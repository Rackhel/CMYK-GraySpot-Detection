"""
Dashboard page: Model performance summary and evaluation metrics.
"""

import streamlit as st
from pathlib import Path
import json

st.title("📊 Dashboard")

st.markdown("## Model Performance Summary")

# Placeholder for loading baseline_summary.json
baseline_dir = Path("data_set/baseline")
summary_file = baseline_dir / "baseline_summary.json"

if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)

    st.write("**Baseline Model Summary:**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mode", summary.get("mode", "N/A"))
    with col2:
        st.metric("Backbone", summary.get("backbone", "N/A"))
    with col3:
        st.metric("Epochs", summary.get("epochs", "N/A"))
    with col4:
        st.metric("Channels", len(summary.get("results", [])))

    st.markdown("---")
    st.write("**Per-Channel Results:**")
    st.json(summary.get("results", []))

else:
    st.warning("⚠️ No baseline summary found. Run training first.")
    st.info("Expected path: `data_set/baseline/baseline_summary.json`")

st.markdown("---")
st.markdown("## Evaluation Metrics")
st.info("TODO: Load and display evaluation results from `outputs/reports/`")
