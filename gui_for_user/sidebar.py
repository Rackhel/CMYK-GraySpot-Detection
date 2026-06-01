"""SidebarWidget — 모델 가중치 경로 + 추론 설정 (숨기기 가능).
Collapsible sidebar for checkpoint paths and inference settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui_for_user.i18n import t

_ROOT = Path(__file__).resolve().parents[1]
_CFG_PATH = Path(__file__).resolve().parent / "assets" / "config.json"
_CHANNELS = ["Y", "M", "C", "K"]

_CH_COLOR = {"Y": "#f9e2af", "M": "#f38ba8", "C": "#89dceb", "K": "#a6adc8"}


class SidebarWidget(QWidget):
    """가중치 파일 경로 + 추론 설정을 편집하는 사이드바.

    Signals:
        settings_applied(dict): Apply 버튼 클릭 시 발생.
            payload = {
                "checkpoints": {"Y": str, "M": str, "C": str, "K": str},
                "device":       str,   # "auto"|"cpu"|"cuda"|"mps"
                "channel_mode": str,   # "all"|"Y"|"M"|"C"|"K"
            }
    """

    settings_applied = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(250)
        self.setMaximumWidth(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setSpacing(12)
        v.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(inner)

        self._hdr_lbl = QLabel(t("sidebar_title"))
        self._hdr_lbl.setStyleSheet("font-size:13px; font-weight:bold; color:#cba6f7;")
        v.addWidget(self._hdr_lbl)

        self._ckpt_group = self._build_ckpt_group()
        v.addWidget(self._ckpt_group)

        self._settings_group = self._build_settings_group()
        v.addWidget(self._settings_group)

        # 모델 경로 안내 레이블 / Model path info label
        models_dir = _ROOT / "data_set" / "models"
        self._path_info_lbl = QLabel(
            f"<span style='font-size:10px; color:#585b70;'>"
            f"모델 위치 / Models dir:<br>"
            f"<b style='color:#a6adc8;'>{models_dir}</b><br>"
            f"예상 파일 / Expected: best_Y.pt, best_M.pt, best_C.pt, best_K.pt</span>"
        )
        self._path_info_lbl.setWordWrap(True)
        v.addWidget(self._path_info_lbl)

        self._apply_btn = QPushButton(t("btn_apply"))
        self._apply_btn.setFixedHeight(34)
        self._apply_btn.setStyleSheet(
            "QPushButton { background:#89b4fa; color:#1e1e2e; font-weight:bold;"
            " border-radius:5px; border:none; }"
            "QPushButton:hover { background:#74c7ec; }"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        v.addWidget(self._apply_btn)
        v.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._load_saved()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ckpt_group(self) -> QGroupBox:
        g = QGroupBox(t("grp_checkpoints"))
        v = QVBoxLayout(g)
        v.setSpacing(8)
        v.setContentsMargins(8, 10, 8, 8)

        self._ckpt_edits: dict[str, QLineEdit] = {}
        self._browse_btns: dict[str, QPushButton] = {}
        self._auto_btns: dict[str, QPushButton] = {}

        for ch in _CHANNELS:
            col = _CH_COLOR[ch]
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            lbl = QLabel(f"<b>{ch}</b>")
            lbl.setFixedWidth(18)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color:{col}; font-size:13px;")

            edit = QLineEdit()
            edit.setPlaceholderText(t("ckpt_placeholder"))
            self._ckpt_edits[ch] = edit

            browse_btn = QPushButton("…")
            browse_btn.setFixedSize(28, 26)
            browse_btn.setToolTip(t("tooltip_browse"))
            browse_btn.clicked.connect(lambda _, c=ch: self._browse(c))
            self._browse_btns[ch] = browse_btn

            auto_btn = QPushButton("🔍")
            auto_btn.setFixedSize(28, 26)
            auto_btn.setToolTip(t("tooltip_auto"))
            auto_btn.clicked.connect(lambda _, c=ch: self._auto_detect(c))
            self._auto_btns[ch] = auto_btn

            row.addWidget(lbl)
            row.addWidget(edit, stretch=1)
            row.addWidget(browse_btn)
            row.addWidget(auto_btn)
            v.addWidget(row_w)

        self._detect_all_btn = QPushButton(t("btn_detect_all"))
        self._detect_all_btn.setToolTip(t("tooltip_detect_all"))
        self._detect_all_btn.clicked.connect(self._auto_detect_all)
        v.addWidget(self._detect_all_btn)
        return g

    def _build_settings_group(self) -> QGroupBox:
        g = QGroupBox(t("grp_settings"))
        f = QFormLayout(g)
        f.setVerticalSpacing(8)
        f.setContentsMargins(8, 10, 8, 8)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "mps"])

        self._mode_combo = QComboBox()
        self._mode_combo.addItem(t("mode_ensemble"), userData="all")
        for ch in _CHANNELS:
            self._mode_combo.addItem(t("mode_single_ch", ch=ch), userData=ch)

        self._device_row_lbl = QLabel(t("lbl_device"))
        self._mode_row_lbl = QLabel(t("lbl_ch_mode"))

        f.addRow(self._device_row_lbl, self._device_combo)
        f.addRow(self._mode_row_lbl, self._mode_combo)
        return g

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse(self, ch: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("dlg_select_ckpt"),
            str(_ROOT),
            "PyTorch (*.pt *.pth)",
        )
        if path:
            self._ckpt_edits[ch].setText(path)

    def _auto_detect(self, ch: str) -> None:
        from gui.workers._ckpt_utils import auto_find_checkpoint

        cfg = self._load_src_cfg()
        p = auto_find_checkpoint(cfg, ch)
        if p:
            self._ckpt_edits[ch].setText(p)
            self._ckpt_edits[ch].setStyleSheet("")
        else:
            self._ckpt_edits[ch].setText("")
            self._ckpt_edits[ch].setPlaceholderText(f"⚠ {ch} 미발견")

    def _auto_detect_all(self) -> None:
        for ch in _CHANNELS:
            self._auto_detect(ch)

    def _on_apply(self) -> None:
        s = self.collect_settings()
        self._save(s)
        self.settings_applied.emit(s)

    # ── Public ────────────────────────────────────────────────────────────────

    def collect_settings(self) -> dict[str, Any]:
        return {
            "checkpoints": {
                ch: self._ckpt_edits[ch].text().strip() for ch in _CHANNELS
            },
            "device": self._device_combo.currentText(),
            "channel_mode": self._mode_combo.currentData(),
        }

    # ── i18n ─────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._hdr_lbl.setText(t("sidebar_title"))
        self._ckpt_group.setTitle(t("grp_checkpoints"))
        self._settings_group.setTitle(t("grp_settings"))
        self._apply_btn.setText(t("btn_apply"))
        self._detect_all_btn.setText(t("btn_detect_all"))
        self._detect_all_btn.setToolTip(t("tooltip_detect_all"))
        for ch in _CHANNELS:
            self._ckpt_edits[ch].setPlaceholderText(t("ckpt_placeholder"))
            self._browse_btns[ch].setToolTip(t("tooltip_browse"))
            self._auto_btns[ch].setToolTip(t("tooltip_auto"))
        self._device_row_lbl.setText(t("lbl_device"))
        self._mode_row_lbl.setText(t("lbl_ch_mode"))
        # 모드 콤보 텍스트 업데이트
        self._mode_combo.setItemText(0, t("mode_ensemble"))
        for i, ch in enumerate(_CHANNELS, start=1):
            self._mode_combo.setItemText(i, t("mode_single_ch", ch=ch))

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_src_cfg() -> dict:
        try:
            from src.utils.utils_config import load_config

            return load_config()
        except Exception:
            return {}

    def _save(self, s: dict) -> None:
        try:
            _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CFG_PATH.write_text(
                json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_saved(self) -> None:
        try:
            if _CFG_PATH.exists():
                d = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
                for ch, p in d.get("checkpoints", {}).items():
                    if ch in self._ckpt_edits and p:
                        self._ckpt_edits[ch].setText(p)
                self._device_combo.setCurrentText(d.get("device", "auto"))
                idx = self._mode_combo.findData(d.get("channel_mode", "all"))
                if idx >= 0:
                    self._mode_combo.setCurrentIndex(idx)
        except Exception:
            pass
        self._auto_detect_all()
