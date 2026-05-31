"""InferenceView — raw 이미지를 CMYK 채널별로 레벨 분류.
Classifies raw images per CMYK channel and shows per-channel level results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui_for_user.i18n import t

_ROOT = Path(__file__).resolve().parents[1]
_CHANNELS = ["Y", "M", "C", "K"]
_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 레벨별 색상 (0=최상, 높을수록 불량)
_LEVEL_COLORS = {
    0: "#22c55e",
    1: "#84cc16",
    2: "#eab308",
    3: "#f97316",
    4: "#ef4444",
    5: "#dc2626",
}
_CH_COLOR = {"Y": "#f9e2af", "M": "#f38ba8", "C": "#89dceb", "K": "#a6adc8"}


def _lv_color(level: int) -> str:
    return _LEVEL_COLORS.get(level, "#a6adc8")


# ── 채널 결과 카드 ─────────────────────────────────────────────────────────────


class ChannelCard(QWidget):
    """Y / M / C / K 채널 하나의 레벨+신뢰도 카드."""

    def __init__(self, channel: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel = channel
        self.setFixedWidth(120)
        self.setMinimumHeight(120)
        self.setStyleSheet("background:#313244; border-radius:8px;")

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 8, 6, 8)
        v.setSpacing(4)

        ch_col = _CH_COLOR.get(channel, "#cdd6f4")
        self._ch_lbl = QLabel(channel)
        self._ch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ch_lbl.setStyleSheet(f"font-size:14px; font-weight:bold; color:{ch_col};")

        self._level_lbl = QLabel("—")
        self._level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._level_lbl.setStyleSheet(
            "font-size:30px; font-weight:bold; color:#585b70;"
        )

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(7)
        self._bar.setTextVisible(False)

        self._conf_lbl = QLabel("—")
        self._conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conf_lbl.setStyleSheet("font-size:11px; color:#a6adc8;")

        v.addWidget(self._ch_lbl)
        v.addWidget(self._level_lbl)
        v.addWidget(self._bar)
        v.addWidget(self._conf_lbl)

    def update_result(self, level: int, confidence: float) -> None:
        col = _lv_color(level)
        self._level_lbl.setText(f"Lv {level}")
        self._level_lbl.setStyleSheet(f"font-size:30px; font-weight:bold; color:{col};")
        pct = int(confidence * 100)
        self._bar.setValue(pct)
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background:{col}; border-radius:3px; }}"
        )
        self._conf_lbl.setText(f"{confidence:.1%}")

    def reset(self) -> None:
        self._level_lbl.setText("—")
        self._level_lbl.setStyleSheet(
            "font-size:30px; font-weight:bold; color:#585b70;"
        )
        self._bar.setValue(0)
        self._bar.setStyleSheet("")
        self._conf_lbl.setText("—")


class EnsembleCard(QWidget):
    """앙상블 최종 결과 카드 (더 크게 표시)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(150)
        self.setMinimumHeight(130)
        self.setStyleSheet(
            "background:#1e1e2e; border:2px solid #45475a; border-radius:10px;"
        )

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 10, 8, 10)
        v.setSpacing(4)

        self._hdr = QLabel(t("lbl_ensemble"))
        self._hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hdr.setStyleSheet("font-size:11px; color:#6c7086; font-weight:bold;")

        self._level_lbl = QLabel("—")
        self._level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._level_lbl.setStyleSheet(
            "font-size:36px; font-weight:bold; color:#585b70;"
        )

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)

        self._conf_lbl = QLabel("—")
        self._conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conf_lbl.setStyleSheet("font-size:13px; color:#a6adc8; font-weight:bold;")

        v.addWidget(self._hdr)
        v.addWidget(self._level_lbl)
        v.addWidget(self._bar)
        v.addWidget(self._conf_lbl)

    def update_result(self, level: int, confidence: float) -> None:
        col = _lv_color(level)
        self._level_lbl.setText(f"Lv {level}")
        self._level_lbl.setStyleSheet(f"font-size:36px; font-weight:bold; color:{col};")
        self._bar.setValue(int(confidence * 100))
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background:{col}; border-radius:4px; }}"
        )
        self._conf_lbl.setText(f"{confidence:.1%}")
        self.setStyleSheet(
            f"background:#1e1e2e; border:2px solid {col}; border-radius:10px;"
        )

    def reset(self) -> None:
        self._level_lbl.setText("—")
        self._level_lbl.setStyleSheet(
            "font-size:36px; font-weight:bold; color:#585b70;"
        )
        self._bar.setValue(0)
        self._bar.setStyleSheet("")
        self._conf_lbl.setText("—")
        self.setStyleSheet(
            "background:#1e1e2e; border:2px solid #45475a; border-radius:10px;"
        )

    def retranslate(self) -> None:
        self._hdr.setText(t("lbl_ensemble"))


# ── 메인 뷰 ────────────────────────────────────────────────────────────────────


class InferenceView(QWidget):
    """메인 추론 패널 — 단일 이미지 추론."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._settings: dict[str, Any] = {
            "checkpoints": {ch: "" for ch in _CHANNELS},
            "device": "auto",
            "channel_mode": "all",
        }
        self._selected_image = ""
        self._infer_worker = None

        main_v = QVBoxLayout(self)
        main_v.setContentsMargins(8, 8, 8, 8)
        main_v.setSpacing(8)

        main_v.addWidget(self._build_toolbar())
        main_v.addWidget(self._build_upper_area(), stretch=1)
        main_v.addWidget(self._build_status_bar())

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(46)
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self._browse_img_btn = QPushButton(t("btn_select_img"))
        self._browse_img_btn.clicked.connect(self._browse_image)

        self._run_btn = QPushButton(t("btn_run"))
        self._run_btn.setStyleSheet(
            "QPushButton { background:#a6e3a1; color:#1e1e2e; font-weight:bold; border-radius:5px; }"
            "QPushButton:hover { background:#94e2d5; }"
            "QPushButton:disabled { background:#313244; color:#585b70; }"
        )
        self._run_btn.clicked.connect(self._run_single)
        self._run_btn.setEnabled(False)

        self._stop_btn = QPushButton(t("btn_stop"))
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_inference)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._file_lbl = QLabel(t("toolbar_placeholder"))
        self._file_lbl.setStyleSheet("color:#6c7086; font-size:11px;")
        self._file_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        h.addWidget(self._browse_img_btn)
        h.addWidget(self._run_btn)
        h.addWidget(self._stop_btn)
        h.addWidget(self._progress, stretch=1)
        return bar

    def _build_upper_area(self) -> QWidget:
        """이미지 미리보기 + 채널 결과 카드 (수평 스플리터)."""
        # 왼쪽: 이미지 미리보기
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        self._preview_lbl_hint = QLabel(t("lbl_preview"))
        self._preview_lbl_hint.setStyleSheet("font-size:11px; color:#6c7086;")
        lv.addWidget(self._preview_lbl_hint)

        self._img_preview = QLabel()
        self._img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_preview.setMinimumSize(240, 240)
        self._img_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._img_preview.setStyleSheet(
            "background:#181825; border:1px dashed #45475a; border-radius:6px;"
        )
        self._img_preview.setText(t("preview_hint"))
        lv.addWidget(self._img_preview, stretch=1)

        self._file_name_lbl = QLabel("")
        self._file_name_lbl.setStyleSheet("font-size:10px; color:#6c7086;")
        self._file_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self._file_name_lbl)

        # 오른쪽: 결과 카드들
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        self._ch_result_lbl = QLabel(
            f"<span style='color:#6c7086; font-size:11px;'>{t('lbl_ch_result')}</span>"
        )
        rv.addWidget(self._ch_result_lbl)

        cards_w = QWidget()
        cards_h = QHBoxLayout(cards_w)
        cards_h.setContentsMargins(0, 0, 0, 0)
        cards_h.setSpacing(8)

        self._ch_cards: dict[str, ChannelCard] = {}
        for ch in _CHANNELS:
            card = ChannelCard(ch)
            self._ch_cards[ch] = card
            cards_h.addWidget(card)

        self._ensemble_card = EnsembleCard()
        cards_h.addSpacing(12)
        cards_h.addWidget(self._ensemble_card)
        cards_h.addStretch()
        rv.addWidget(cards_w)

        self._top3_hdr_lbl = QLabel(
            f"<span style='color:#6c7086; font-size:11px;'>{t('lbl_top3')}</span>"
        )
        rv.addWidget(self._top3_hdr_lbl)

        self._top3_lbl = QLabel("—")
        self._top3_lbl.setStyleSheet(
            "background:#181825; border-radius:6px; padding:8px; font-size:12px;"
        )
        self._top3_lbl.setWordWrap(True)
        rv.addWidget(self._top3_lbl)

        self._status_hdr_lbl = QLabel(
            f"<span style='color:#6c7086; font-size:11px;'>{t('lbl_status')}</span>"
        )
        rv.addWidget(self._status_hdr_lbl)

        self._log_lbl = QLabel(t("status_ready"))
        self._log_lbl.setStyleSheet(
            "background:#181825; border-radius:6px; padding:8px;"
            " font-size:11px; color:#a6adc8;"
        )
        self._log_lbl.setWordWrap(True)
        self._log_lbl.setFixedHeight(70)
        rv.addWidget(self._log_lbl)
        rv.addStretch()

        # 수평 스플리터로 좌우 크기 조절
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.addWidget(left)
        hsplit.addWidget(right)
        hsplit.setSizes([300, 600])
        hsplit.setHandleWidth(4)

        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(hsplit)
        return w

    def _build_status_bar(self) -> QLabel:
        self._status_lbl = QLabel(t("status_main"))
        self._status_lbl.setFixedHeight(20)
        self._status_lbl.setStyleSheet(
            "font-size:10px; color:#6c7086; padding-left:4px;"
        )
        return self._status_lbl

    # ── Settings from sidebar ─────────────────────────────────────────────────

    def apply_settings(self, settings: dict[str, Any]) -> None:
        self._settings = settings
        mode = settings.get("channel_mode", "all")
        mode_str = t("mode_label_ens") if mode == "all" else t("mode_label_ch", ch=mode)
        self._status_lbl.setText(
            f"{t('settings_applied')} — {mode_str}  Device: {settings.get('device','auto')}"
        )

    # ── Drag & Drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() in _IMG_EXTS:
            self._set_image(str(path))

    # ── Browse / set ──────────────────────────────────────────────────────────

    def _browse_image(self) -> None:
        src_dir = str(_ROOT / "data_set" / "raw")
        path, _ = QFileDialog.getOpenFileName(
            self, t("dlg_select_img"), src_dir, _IMG_FILTER
        )
        if path:
            self._set_image(path)

    def _set_image(self, path: str) -> None:
        self._selected_image = path
        self._file_lbl.setText(Path(path).name)
        self._file_name_lbl.setText(Path(path).name)
        px = QPixmap(path)
        if not px.isNull():
            self._img_preview.setPixmap(
                px.scaled(
                    self._img_preview.width() or 300,
                    self._img_preview.height() or 300,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._run_btn.setEnabled(True)
        self._reset_results()

    # ── Run / Stop ────────────────────────────────────────────────────────────

    def _run_single(self) -> None:
        if not self._selected_image:
            return
        from gui.workers.inference_worker import InferenceWorker

        cfg = self._build_cfg()
        ch = self._settings.get("channel_mode", "all")
        ckpt = self._settings.get("checkpoints", {}).get(ch, "") if ch != "all" else ""

        self._infer_worker = InferenceWorker(
            cfg, self._selected_image, ckpt, channel=ch
        )
        self._infer_worker.progress_updated.connect(self._progress.setValue)
        self._infer_worker.log_emitted.connect(self._on_log)
        self._infer_worker.finished.connect(self._on_single_done)
        self._infer_worker.error_occurred.connect(self._on_error)
        self._set_running(True)
        self._reset_results()
        self._infer_worker.start()

    def _stop_inference(self) -> None:
        if self._infer_worker is not None and self._infer_worker.isRunning():
            self._infer_worker.cancel()
        self._set_running(False)

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_single_done(self, result: dict) -> None:
        ch = result.get("channel", "all")
        pred = result.get("pred_level", 0)
        conf = result.get("confidence", 0.0)

        if ch == "all":
            per = result.get("per_channel", {})
            for c in _CHANNELS:
                if c in per:
                    self._ch_cards[c].update_result(per[c]["pred"], per[c]["conf"])
            self._ensemble_card.update_result(pred, conf)
        else:
            self._ch_cards[ch].update_result(pred, conf)
            self._ensemble_card.update_result(pred, conf)

        top3 = result.get("top3", [])
        if top3:
            lines = [f"  {i+1}위  Lv {lv}  ({p:.1%})" for i, (lv, p) in enumerate(top3)]
            self._top3_lbl.setText("\n".join(lines))

        self._on_log(f"{t('log_done')} — Lv {pred}  ({conf:.1%})")
        self._set_running(False)

    def _on_log(self, msg: str) -> None:
        line = msg.strip().split("\n")[0][:120]
        self._log_lbl.setText(line)
        self._status_lbl.setText(line)

    def _on_error(self, msg: str) -> None:
        first_line = msg.split("\n")[0]
        self._log_lbl.setText(f"{t('log_error')}: {first_line}")
        self._status_lbl.setText(f"{t('log_error')}: {first_line}")
        self._set_running(False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_cfg(self) -> dict:
        try:
            from src.utils.utils_config import load_config

            cfg = load_config()
        except Exception:
            cfg = {
                "data": {"image_size": 128, "num_levels": 6, "channels": _CHANNELS},
                "storage": {"models_dir": "data_set/models"},
                "system": {},
            }
        cfg.setdefault("system", {})["device"] = self._settings.get("device", "auto")
        ckpts = self._settings.get("checkpoints", {})
        for ch, p in ckpts.items():
            if p:
                cfg.setdefault("storage", {})[f"checkpoint_{ch}"] = p
        return cfg

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._browse_img_btn.setEnabled(not running)
        if not running:
            self._progress.setValue(0)

    def _reset_results(self) -> None:
        for card in self._ch_cards.values():
            card.reset()
        self._ensemble_card.reset()
        self._top3_lbl.setText("—")
        self._log_lbl.setText(t("status_ready"))

    # ── i18n ─────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._browse_img_btn.setText(t("btn_select_img"))
        self._run_btn.setText(t("btn_run"))
        self._stop_btn.setText(t("btn_stop"))
        self._file_lbl.setText(t("toolbar_placeholder"))
        self._preview_lbl_hint.setText(t("lbl_preview"))
        self._img_preview.setText(t("preview_hint"))
        self._ch_result_lbl.setText(
            f"<span style='color:#6c7086; font-size:11px;'>{t('lbl_ch_result')}</span>"
        )
        self._top3_hdr_lbl.setText(
            f"<span style='color:#6c7086; font-size:11px;'>{t('lbl_top3')}</span>"
        )
        self._status_hdr_lbl.setText(
            f"<span style='color:#6c7086; font-size:11px;'>{t('lbl_status')}</span>"
        )
        self._log_lbl.setText(t("status_ready"))
        self._status_lbl.setText(t("status_main"))
        self._ensemble_card.retranslate()
