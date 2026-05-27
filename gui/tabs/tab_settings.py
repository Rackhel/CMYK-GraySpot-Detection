"""Settings tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from gui.components.log_panel import LogPanel
from gui.tabs.base_tab import BaseTab


class SettingsTab(BaseTab):
    """JSON-backed settings editor separated from UI logic."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_path: Path | None = None) -> None:
        super().__init__(cfg)
        self.config_path = config_path or Path("gui/assets/config.json")
        self.data_root = QLineEdit(self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
        self.checkpoint_path = QLineEdit("")
        self.log_panel = LogPanel()
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Settings"))
        form = QFormLayout()
        form.addRow("Labeled Dataset Directory", self.data_root)
        form.addRow("Checkpoint Path", self.checkpoint_path)
        layout.addLayout(form)
        layout.addWidget(save_button)
        layout.addWidget(self.log_panel)

    def save_settings(self) -> None:
        """Persist GUI settings to JSON."""

        payload = {"labeled_dir": self.data_root.text(), "checkpoint_path": self.checkpoint_path.text()}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log_panel.append("Settings saved.")
