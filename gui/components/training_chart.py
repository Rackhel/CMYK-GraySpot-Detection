"""TrainingChart — 실시간 학습 곡선 위젯.
Accumulates epoch metrics from log strings and renders a Plotly line chart.
"""

from __future__ import annotations

import re
from typing import Any

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .plotly_widget import PlotlyWidget

_EPOCH_RE = re.compile(
    r"\[epoch\s*(\d+)\].*?loss[=: ]+([0-9.]+).*?val_?(?:loss|acc)[=: ]+([0-9.]+)",
    re.IGNORECASE,
)


class TrainingChart(QWidget):
    """Plotly line chart that builds up training history epoch-by-epoch.

    Feed data via `append_epoch()` or parse log strings via `parse_log_line()`.
    Call `render()` to update the chart (batched for performance).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history: list[dict[str, Any]] = []
        self._chart = PlotlyWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._chart)
        self._render_placeholder()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._history.clear()
        self._render_placeholder()

    def append_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ) -> None:
        self._history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    def parse_log_line(self, line: str) -> bool:
        """Try to parse an epoch summary line; returns True if parsed."""
        m = _EPOCH_RE.search(line)
        if not m:
            return False
        epoch = int(m.group(1))
        train_val = float(m.group(2))
        other_val = float(m.group(3))
        is_acc = "acc" in line.lower()[m.start() :]
        self.append_epoch(
            epoch,
            train_loss=train_val,
            val_loss=None if is_acc else other_val,
            val_acc=other_val if is_acc else None,
        )
        return True

    def render(self) -> None:
        """Redraw chart from accumulated history."""
        if not self._history:
            self._render_placeholder()
            return

        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        epochs = [r["epoch"] for r in self._history]
        train_loss = [r["train_loss"] for r in self._history]
        val_loss = [
            r["val_loss"] for r in self._history if r.get("val_loss") is not None
        ]
        val_acc = [r["val_acc"] for r in self._history if r.get("val_acc") is not None]
        val_epochs = [
            r["epoch"] for r in self._history if r.get("val_loss") is not None
        ]
        acc_epochs = [r["epoch"] for r in self._history if r.get("val_acc") is not None]

        traces = [
            go.Scatter(x=epochs, y=train_loss, name="Train Loss", mode="lines+markers")
        ]
        if val_loss:
            traces.append(
                go.Scatter(
                    x=val_epochs, y=val_loss, name="Val Loss", mode="lines+markers"
                )
            )
        if val_acc:
            traces.append(
                go.Scatter(
                    x=acc_epochs,
                    y=val_acc,
                    name="Val Acc",
                    mode="lines+markers",
                    yaxis="y2",
                )
            )

        layout = go.Layout(
            title="Training Curves",
            xaxis={"title": "Epoch"},
            yaxis={"title": "Loss"},
            yaxis2={
                "title": "Accuracy",
                "overlaying": "y",
                "side": "right",
                "rangemode": "tozero",
            },
            template="plotly_dark",
            height=280,
            legend={"orientation": "h", "y": -0.2},
            margin={"t": 40, "b": 60},
        )
        self._chart.set_figure(go.Figure(data=traces, layout=layout))

    def load_history_from_result(self, result: dict[str, Any]) -> None:
        """Load final metrics as a bar chart (no per-epoch data available)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        metrics = {
            k: result[k] for k in ("best_val_acc", "test_acc", "mae") if k in result
        }
        if not metrics:
            return

        fig = go.Figure(
            data=go.Bar(
                x=list(metrics.keys()), y=list(metrics.values()), marker_color="#60a5fa"
            )
        )
        fig.update_layout(
            title="Training Result",
            template="plotly_dark",
            height=240,
            margin={"t": 40},
        )
        self._chart.set_figure(fig)

    # ── Private ───────────────────────────────────────────────────────────────

    def _render_placeholder(self) -> None:
        self._chart.set_html(
            "<div style='color:#888;font-family:sans-serif;padding:20px;text-align:center'>"
            "학습을 시작하면 곡선이 여기에 표시됩니다.<br>"
            "Training curves will appear here after starting.</div>"
        )
