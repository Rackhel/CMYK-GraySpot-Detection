from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class LogPanel(QWidget):
    """Read-only runtime log stream widget."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def append(self, message: str) -> None:
        """Append one log line."""

        self.log.appendPlainText(message)

    def clear(self) -> None:
        """Clear all visible logs."""

        self.log.clear()
