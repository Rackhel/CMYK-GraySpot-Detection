"""Reusable Plotly chart widget for PyQt6."""

from __future__ import annotations

from PyQt6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget


class PlotlyWidget(QWidget):
    """Embed Plotly HTML inside a Qt widget.

    QWebEngineView is used when PyQt6-WebEngine is installed. A QTextBrowser
    fallback keeps tests and lightweight environments importable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            self.view = QWebEngineView()
            self._web_engine = True
        except Exception:
            self.view = QTextBrowser()
            self._web_engine = False
        layout.addWidget(self.view)

    def set_figure(self, figure) -> None:
        """Render a Plotly figure object."""

        html = figure.to_html(include_plotlyjs="cdn", full_html=False)
        self.set_html(html)

    def set_html(self, html: str) -> None:
        """Render raw HTML."""

        if self._web_engine:
            self.view.setHtml(html)
        else:
            self.view.setHtml("<p>Plotly WebEngine backend unavailable.</p>")

    def show_matrix(self, matrix: list[list[int]], title: str = "Confusion Matrix") -> None:
        """Render a confusion matrix heatmap."""

        import plotly.graph_objects as go

        figure = go.Figure(data=go.Heatmap(z=matrix, colorscale="Blues"))
        figure.update_layout(title=title, template="plotly_dark", height=360)
        self.set_figure(figure)

    def load_file(self, path: str) -> None:
        """Load a locally-saved HTML file (e.g., confusion matrix from evaluation)."""
        if self._web_engine:
            from PyQt6.QtCore import QUrl
            self.view.load(QUrl.fromLocalFile(path))
        else:
            self.view.setHtml("<p>WebEngine unavailable — cannot display local HTML.</p>")

    def show_scatter(
        self,
        points: list[list[float]],
        labels: list[int],
        title: str = "Embedding Projection",
    ) -> None:
        """Render a 2D embedding scatter plot."""

        import plotly.graph_objects as go

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        figure = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode="markers", marker={"color": labels}))
        figure.update_layout(title=title, template="plotly_dark", height=360)
        self.set_figure(figure)
