"""
Reusable metric display components.
"""

import streamlit as st
from typing import Dict, List


def display_metric_card(
    label: str, value: float, suffix: str = "", color: str = "blue"
):
    """
    Display a single metric card.

    Args:
        label: Metric name
        value: Metric value
        suffix: Unit suffix (%, etc.)
        color: Color theme (blue, green, red, orange)
    """
    st.metric(label=label, value=f"{value:.4f} {suffix}".strip())


def display_metrics_grid(metrics: Dict[str, float], cols: int = 3):
    """
    Display multiple metrics in a grid layout.

    Args:
        metrics: Dict of metric_name -> value
        cols: Number of columns in grid
    """
    col_list = st.columns(cols)

    for idx, (name, value) in enumerate(metrics.items()):
        with col_list[idx % cols]:
            st.metric(label=name, value=f"{value:.4f}")
