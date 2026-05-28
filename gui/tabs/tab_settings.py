"""Settings tab — 모델·학습 파라미터 전체 편집 및 저장.
Contract: Contract_gui.md §3.4  /  SSOT_GUI.md §6.4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.components.log_panel import LogPanel
from gui.i18n import t, set_lang, get_lang
from gui.tabs.base_tab import BaseTab

_ROOT       = Path(__file__).resolve().parents[2]
_SRC_CONFIG = _ROOT / "src" / "config" / "config.json"

# ── 레이아웃 상수 / Layout constants ─────────────────────────────────────────
_LABEL_W  = 160   # 라벨 고정 폭 (px) / fixed label column width
_FIELD_W  = 180   # 필드 최대 폭 (px) / max field width


class SettingsTab(BaseTab):
    """모든 config 파라미터를 폼 형태로 편집하고 src/config/config.json에 저장한다."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__(cfg)
        self.gui_config_path = config_path or (_ROOT / "gui" / "assets" / "config.json")

        # ── 스크롤 영역 / Scroll area ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        main_form = QVBoxLayout(content)
        main_form.setSpacing(12)
        main_form.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(content)

        # ── 섹션별 위젯 생성 / Build section widgets ──────────────────────────
        main_form.addWidget(self._build_appearance_group())
        main_form.addWidget(self._build_storage_group())
        main_form.addWidget(self._build_worker_settings_group())
        main_form.addWidget(self._build_phase2_group())
        main_form.addWidget(self._build_phase0_group())
        main_form.addWidget(self._build_train_group())
        main_form.addStretch()

        # ── 버튼 / Buttons ────────────────────────────────────────────────────
        self.log_panel = LogPanel()
        self._save_btn  = QPushButton(t("btn_save_settings"))
        self._reset_btn = QPushButton(t("btn_reset"))
        self._save_btn.clicked.connect(self.save_settings)
        self._reset_btn.clicked.connect(self.refresh)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._reset_btn)

        # ── 최상위 레이아웃 / Top-level layout ───────────────────────────────
        layout = QVBoxLayout(self)
        layout.addWidget(scroll, stretch=1)
        layout.addLayout(btn_row)
        layout.addWidget(self.log_panel)

    # ── BaseTab interface ─────────────────────────────────────────────────────

    def refresh(self) -> None:
        s = self.cfg.get("storage", {})
        self._labeled_dir.setText(s.get("labeled_dir", "data_set/labeled"))
        self._models_dir.setText(s.get("models_dir", "data_set/models"))
        self._reports_dir.setText(s.get("reports_dir", "outputs/reports"))
        self._checkpoint_path.setText(self._load_gui_config().get("checkpoint_path", ""))

        p2 = self.cfg.get("phase2", {})
        self._p2_epochs.setValue(int(p2.get("epochs", 50)))
        self._p2_batch.setValue(int(p2.get("batch_size", 16)))
        self._p2_lr.setValue(float(p2.get("learning_rate", 1e-4)))
        self._p2_wd.setValue(float(p2.get("weight_decay", 1e-4)))
        self._p2_dropout.setValue(float(p2.get("dropout", 0.3)))
        self._p2_hidden.setValue(int(p2.get("hidden_dim", 256)))
        self._p2_oversample.setChecked(bool(p2.get("oversample", True)))
        es = p2.get("early_stopping", {})
        self._p2_es_enabled.setChecked(bool(es.get("enabled", True)))
        self._p2_es_patience.setValue(int(es.get("patience", 10)))

        p0 = self.cfg.get("phase0", {})
        self._p0_epochs.setValue(int(p0.get("epochs", 10)))
        self._p0_batch.setValue(int(p0.get("batch_size", 16)))
        self._p0_lr.setValue(float(p0.get("learning_rate", 1e-3)))
        self._p0_temp.setValue(float(p0.get("temperature", 0.1)))

        train_cfg = self.cfg.get("train", {})
        self._seed.setValue(int(train_cfg.get("seed", 42)))
        self._num_workers.setValue(int(train_cfg.get("num_workers", 0)))
        self._scheduler.setCurrentText(train_cfg.get("scheduler", "cosine"))
        self._grad_clip.setValue(float(train_cfg.get("gradient_clip", 1.0)))

        sys_cfg = self.cfg.get("system", {})
        self._device_combo.setCurrentText(sys_cfg.get("device", "auto"))
        self._dataloader_workers.setValue(int(train_cfg.get("num_workers", 0)))
        self._infer_batch_size.setValue(int(sys_cfg.get("inference_batch_size", 1)))
        self._train_timeout.setValue(int(sys_cfg.get("training_timeout_min", 60)))

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        """Settings 탭은 Worker를 사용하지 않음."""

    # ── Public API ────────────────────────────────────────────────────────────

    def get_checkpoint_path(self) -> str:
        return self._checkpoint_path.text().strip()

    def get_labeled_dir(self) -> str:
        return self._labeled_dir.text().strip()

    def save_settings(self) -> None:
        try:
            src_cfg: dict = json.loads(_SRC_CONFIG.read_text(encoding="utf-8"))

            src_cfg.setdefault("storage", {}).update({
                "labeled_dir": self._labeled_dir.text().strip(),
                "models_dir":  self._models_dir.text().strip(),
                "reports_dir": self._reports_dir.text().strip(),
            })
            src_cfg.setdefault("phase2", {}).update({
                "epochs":        self._p2_epochs.value(),
                "batch_size":    self._p2_batch.value(),
                "learning_rate": self._p2_lr.value(),
                "weight_decay":  self._p2_wd.value(),
                "dropout":       self._p2_dropout.value(),
                "hidden_dim":    self._p2_hidden.value(),
                "oversample":    self._p2_oversample.isChecked(),
            })
            src_cfg["phase2"].setdefault("early_stopping", {}).update({
                "enabled":  self._p2_es_enabled.isChecked(),
                "patience": self._p2_es_patience.value(),
            })
            src_cfg.setdefault("phase0", {}).update({
                "epochs":        self._p0_epochs.value(),
                "batch_size":    self._p0_batch.value(),
                "learning_rate": self._p0_lr.value(),
                "temperature":   self._p0_temp.value(),
            })
            src_cfg.setdefault("train", {}).update({
                "seed":          self._seed.value(),
                "num_workers":   self._dataloader_workers.value(),
                "scheduler":     self._scheduler.currentText(),
                "gradient_clip": self._grad_clip.value(),
            })
            src_cfg.setdefault("system", {}).update({
                "device":                self._device_combo.currentText(),
                "inference_batch_size":  self._infer_batch_size.value(),
                "training_timeout_min":  self._train_timeout.value(),
            })

            _SRC_CONFIG.write_text(
                json.dumps(src_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.cfg.update(src_cfg)

            self.gui_config_path.parent.mkdir(parents=True, exist_ok=True)
            self.gui_config_path.write_text(
                json.dumps({
                    "labeled_dir":     self._labeled_dir.text().strip(),
                    "checkpoint_path": self._checkpoint_path.text().strip(),
                    "theme":           self._theme_combo.currentData(),
                    "lang":            self._lang_combo.currentData(),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.log_panel.append("✅ Settings saved → src/config/config.json")
        except Exception as exc:
            self.log_panel.append(f"❌ Save failed: {exc}")

    # ── retranslate ───────────────────────────────────────────────────────────

    def retranslate_ui(self, lang: str) -> None:
        self._grp_appearance.setTitle(t("grp_appearance"))
        self._grp_storage.setTitle(t("grp_storage"))
        self._grp_phase2.setTitle(t("grp_phase2"))
        self._grp_phase0.setTitle(t("grp_phase0"))
        self._grp_train.setTitle(t("grp_train_cmn"))
        self._save_btn.setText(t("btn_save_settings"))
        self._reset_btn.setText(t("btn_reset"))

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_appearance_group(self) -> QGroupBox:
        gui_cfg = self._load_gui_config()
        self._grp_appearance = QGroupBox(t("grp_appearance"))
        f = self._make_form(self._grp_appearance)

        # 테마 콤보박스 / Theme combo
        self._theme_combo = QComboBox()
        self._theme_combo.addItem(t("theme_dark"),  userData="dark")
        self._theme_combo.addItem(t("theme_light"), userData="light")
        self._theme_combo.setMaximumWidth(_FIELD_W)
        saved_theme = gui_cfg.get("theme", "dark")
        idx = self._theme_combo.findData(saved_theme)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)

        # 언어 콤보박스 / Language combo
        self._lang_combo = QComboBox()
        self._lang_combo.addItem("한국어", userData="ko")
        self._lang_combo.addItem("English", userData="en")
        self._lang_combo.setMaximumWidth(_FIELD_W)
        saved_lang = gui_cfg.get("lang", "ko")
        idx = self._lang_combo.findData(saved_lang)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)

        f.addRow(t("lbl_theme"), self._theme_combo)
        f.addRow(t("lbl_lang"),  self._lang_combo)
        return self._grp_appearance

    def _build_storage_group(self) -> QGroupBox:
        self._grp_storage = QGroupBox(t("grp_storage"))
        g = self._grp_storage
        f = self._make_form(g)

        self._labeled_dir  = self._lineedit(
            self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
        self._models_dir   = self._lineedit(
            self.cfg.get("storage", {}).get("models_dir", "data_set/models"))
        self._reports_dir  = self._lineedit(
            self.cfg.get("storage", {}).get("reports_dir", "outputs/reports"))

        # 체크포인트 경로 + Browse 버튼
        self._checkpoint_path = QLineEdit(
            self._load_gui_config().get("checkpoint_path", ""))
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.setContentsMargins(0, 0, 0, 0)
        ckpt_row.addWidget(self._checkpoint_path)
        ckpt_row.addWidget(browse_btn)
        ckpt_widget = QWidget()
        ckpt_widget.setLayout(ckpt_row)

        f.addRow("Labeled Dataset Dir", self._labeled_dir)
        f.addRow("Models Dir",          self._models_dir)
        f.addRow("Reports Dir",         self._reports_dir)
        f.addRow("Checkpoint (.pt)",    ckpt_widget)
        return g

    def _build_worker_settings_group(self) -> QGroupBox:
        """Worker 설정 그룹: device, num_workers, inference batch size, timeout."""
        self._grp_worker = QGroupBox("Worker Settings")
        f = self._make_form(self._grp_worker)
        sys_cfg   = self.cfg.get("system", {})
        train_cfg = self.cfg.get("train", {})

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self._device_combo.setCurrentText(sys_cfg.get("device", "auto"))
        self._device_combo.setMaximumWidth(_FIELD_W)

        self._infer_batch_size = self._spin(sys_cfg.get("inference_batch_size", 1), 1, 256)
        self._dataloader_workers = self._spin(train_cfg.get("num_workers", 0), 0, 32)
        self._train_timeout = self._spin(sys_cfg.get("training_timeout_min", 60), 1, 600)

        f.addRow("Device",                  self._device_combo)
        f.addRow("DataLoader Workers",      self._dataloader_workers)
        f.addRow("Inference Batch Size",    self._infer_batch_size)
        f.addRow("Training Timeout (min)",  self._train_timeout)
        return self._grp_worker

    def _build_phase2_group(self) -> QGroupBox:
        self._grp_phase2 = QGroupBox(t("grp_phase2"))
        g = self._grp_phase2
        f = self._make_form(g)
        p2 = self.cfg.get("phase2", {})
        es = p2.get("early_stopping", {})

        self._p2_epochs  = self._spin(p2.get("epochs", 50),  1, 500)
        self._p2_batch   = self._spin(p2.get("batch_size", 16), 1, 256)
        self._p2_lr      = self._dspin(p2.get("learning_rate", 1e-4), 1e-6, 1.0, 6)
        self._p2_wd      = self._dspin(p2.get("weight_decay",  1e-4), 1e-7, 1.0, 7)
        self._p2_dropout = self._dspin(p2.get("dropout", 0.3), 0.0, 0.9, 2)
        self._p2_hidden  = self._spin(p2.get("hidden_dim", 256), 64, 2048, step=64)

        self._p2_oversample = QCheckBox("Oversample minority classes")
        self._p2_oversample.setChecked(bool(p2.get("oversample", True)))
        self._p2_es_enabled = QCheckBox("Early Stopping enabled")
        self._p2_es_enabled.setChecked(bool(es.get("enabled", True)))
        self._p2_es_patience = self._spin(es.get("patience", 10), 1, 200)

        f.addRow("Epochs",        self._p2_epochs)
        f.addRow("Batch Size",    self._p2_batch)
        f.addRow("Learning Rate", self._p2_lr)
        f.addRow("Weight Decay",  self._p2_wd)
        f.addRow("Dropout",       self._p2_dropout)
        f.addRow("Hidden Dim",    self._p2_hidden)
        f.addRow(self._p2_oversample)   # full-span
        f.addRow(self._p2_es_enabled)   # full-span
        f.addRow("ES Patience",   self._p2_es_patience)
        return g

    def _build_phase0_group(self) -> QGroupBox:
        self._grp_phase0 = QGroupBox(t("grp_phase0"))
        g = self._grp_phase0
        f = self._make_form(g)
        p0 = self.cfg.get("phase0", {})

        self._p0_epochs = self._spin(p0.get("epochs", 10), 1, 200)
        self._p0_batch  = self._spin(p0.get("batch_size", 16), 1, 256)
        self._p0_lr     = self._dspin(p0.get("learning_rate", 1e-3), 1e-6, 1.0, 6)
        self._p0_temp   = self._dspin(p0.get("temperature", 0.1), 0.01, 1.0, 3)

        f.addRow("Epochs",        self._p0_epochs)
        f.addRow("Batch Size",    self._p0_batch)
        f.addRow("Learning Rate", self._p0_lr)
        f.addRow("Temperature",   self._p0_temp)
        return g

    def _build_train_group(self) -> QGroupBox:
        self._grp_train = QGroupBox(t("grp_train_cmn"))
        g = self._grp_train
        f = self._make_form(g)
        train_cfg = self.cfg.get("train", {})

        self._seed        = self._spin(train_cfg.get("seed", 42), 0, 99999)
        self._num_workers = self._spin(train_cfg.get("num_workers", 0), 0, 16)
        self._scheduler   = QComboBox()
        self._scheduler.addItems(["cosine", "step", "none"])
        self._scheduler.setCurrentText(train_cfg.get("scheduler", "cosine"))
        self._scheduler.setMaximumWidth(_FIELD_W)
        self._grad_clip   = self._dspin(train_cfg.get("gradient_clip", 1.0), 0.0, 10.0, 2)

        f.addRow("Seed",          self._seed)
        f.addRow("Num Workers",   self._num_workers)
        f.addRow("Scheduler",     self._scheduler)
        f.addRow("Gradient Clip", self._grad_clip)
        return g

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_form(parent: QGroupBox) -> QFormLayout:
        """일관된 폼 레이아웃을 생성한다 / Create consistently-styled QFormLayout."""
        f = QFormLayout(parent)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        f.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        f.setContentsMargins(8, 6, 8, 6)
        return f

    @staticmethod
    def _spin(val, mn, mx, step=1) -> QSpinBox:
        w = QSpinBox()
        w.setRange(mn, mx)
        w.setSingleStep(step)
        w.setValue(int(val))
        w.setMaximumWidth(_FIELD_W)
        return w

    @staticmethod
    def _dspin(val, mn, mx, decimals=4) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(mn, mx)
        w.setDecimals(decimals)
        w.setSingleStep(10 ** (-decimals))
        w.setValue(float(val))
        w.setMaximumWidth(_FIELD_W)
        return w

    @staticmethod
    def _lineedit(text: str = "") -> QLineEdit:
        w = QLineEdit(text)
        return w

    def _load_gui_config(self) -> dict:
        try:
            if self.gui_config_path.exists():
                return json.loads(self.gui_config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", str(_ROOT), "PyTorch Checkpoint (*.pt *.pth)"
        )
        if path:
            self._checkpoint_path.setText(path)
