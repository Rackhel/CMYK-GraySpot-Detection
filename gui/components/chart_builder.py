"""
Chart building utilities for Streamlit.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List


def plot_confusion_matrix_streamlit(y_true: List, y_pred: List, title: str = "Confusion Matrix"):
    """
    Display confusion matrix as Plotly heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Chart title
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale="Blues",
    ))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True")
    
    st.plotly_chart(fig, use_container_width=True)


def plot_accuracy_history(history: Dict[str, List[float]], title: str = "Training History"):
    """
    Display training accuracy/loss curves.
    
    Args:
        history: Dict with keys like 'train_acc', 'val_acc', 'loss'
        title: Chart title
    """
    fig = go.Figure()
    
    for key, values in history.items():
        fig.add_trace(go.Scatter(
            y=values,
            mode='lines',
            name=key,
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Value",
        hovermode="x unified",
    )
    
    st.plotly_chart(fig, use_container_width=True)
