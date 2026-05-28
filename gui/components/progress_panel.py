from PyQt6.QtWidgets import QProgressBar, QVBoxLayout, QWidget

from gui.components.log_panel import LogPanel


class ProgressPanel(QWidget):
    """Reusable progress bar plus log panel."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.logs = LogPanel()
        layout.addWidget(self.logs)

    def set_progress(self, value: int) -> None:
        """Update the progress bar."""

        self.progress.setValue(int(value))

    def append_log(self, message: str) -> None:
        """Append one progress log line."""

        self.logs.append(message)
