from typing import Any

from PyQt6.QtWidgets import QWidget


class BaseTab(QWidget):
    """Base interface for all tabs."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}

    def refresh(self) -> None:
        """Refresh tab state when selected."""

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        """Handle a worker completion payload."""
