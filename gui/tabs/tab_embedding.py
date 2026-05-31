"""Embedding tab — t-SNE 시각화, 다채널 비교, 레벨 순도 분석, 유사 이미지 탐색, 라벨 교정.
Multi-channel t-SNE, level purity analysis, similar-image finder, versioned label correction.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.components.image_viewer import ImageViewer
from gui.components.log_panel import LogPanel
from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.services.embedding_service import EmbeddingService
from gui.workers.embedding_worker import EmbeddingWorker

from .base_tab import BaseTab

_CHANNELS = ["Y", "M", "C", "K"]
_CH_COLORS = {"Y": "#facc15", "M": "#f472b6", "C": "#38bdf8", "K": "#a3a3a3"}


class EmbeddingTab(BaseTab):
    """t-SNE embedding visualization with multi-channel comparison, level purity, and label correction."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        labels_dir: Path | None = None,
        settings_tab=None,
    ) -> None:
        super().__init__(cfg)
        self.service = EmbeddingService()
        self.worker: EmbeddingWorker | None = None
        self._settings_tab = settings_tab
        self.labels_dir = (
            Path(labels_dir) if labels_dir is not None else Path("data_set")
        )

        self._selected_path: str = ""
        self._embedding_paths: list[str] = []
        self._all_channel_data: dict[str, dict] = {}  # ch → {points, labels, paths}

        # ── 컨트롤 바 / Control bar ───────────────────────────────────────
        ctrl_group = QGroupBox("임베딩 추출 / Extract Embeddings")
        ctrl_v = QVBoxLayout(ctrl_group)

        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K", "전체 비교 (All)"])
        self.channel_box.setMaximumWidth(200)

        self._run_btn = QPushButton("▶  임베딩 추출 / Extract")
        self._stop_btn = QPushButton("■  중지 / Stop")
        self._run_btn.clicked.connect(self.start_embedding)
        self._stop_btn.clicked.connect(self.stop_embedding)
        self._stop_btn.setEnabled(False)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("채널:"))
        ctrl_row.addWidget(self.channel_box)
        ctrl_row.addWidget(self._run_btn)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch()
        ctrl_v.addLayout(ctrl_row)

        # ── 내부 탭: Scatter / Purity / Similar ──────────────────────────
        self._inner_tabs = QTabWidget()
        self._inner_tabs.addTab(self._build_scatter_tab(), "🗺️  t-SNE 산점도")
        self._inner_tabs.addTab(self._build_purity_tab(), "📊  레벨 순도")
        self._inner_tabs.addTab(self._build_similar_tab(), "🔍  유사 이미지")
        self._inner_tabs.addTab(self._build_correction_tab(), "✏️  라벨 교정")

        # ── 진행 / Progress ───────────────────────────────────────────────
        self.progress = ProgressPanel()
        self.progress.setMaximumHeight(60)
        self.log = LogPanel()
        self.log.setMaximumHeight(60)

        # ── 레이아웃 / Layout ─────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(ctrl_group)
        layout.addWidget(self._inner_tabs, stretch=1)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        pass

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        ch = result.get("channel", "?")
        points = result.get("embeddings_2d", [])
        labels = result.get("labels", [])
        paths = result.get("paths", [])
        self._all_channel_data[ch] = {
            "points": points,
            "labels": labels,
            "paths": paths,
        }
        self._embedding_paths = paths
        self.log.append(f"✅ [{ch}] {len(points)}개 포인트 완료")
        self._render_scatter()
        self._render_purity(ch, labels)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── Public API ────────────────────────────────────────────────────────────

    def start_embedding(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.log.append("⚠️  이미 실행 중입니다.")
            return

        sel = self.channel_box.currentText()
        if sel.startswith("전체"):
            self._all_channel_data = {}
            self._pending_channels = list(_CHANNELS)
        else:
            self._pending_channels = [sel]

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._run_next_channel()

    def _run_next_channel(self) -> None:
        if not self._pending_channels:
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            return

        ch = self._pending_channels.pop(0)
        ckpt = self._get_checkpoint(ch)
        self.worker = self.service.start_embedding(self.cfg, ch, ckpt)
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self._on_ch_done)
        self.worker.error_occurred.connect(
            lambda msg, c=ch: self.log.append(f"❌ [{c}] {msg.splitlines()[0]}")
        )
        self.worker.start()

    def _on_ch_done(self, result: dict) -> None:
        self.on_worker_finished(result)
        self._run_next_channel()

    def stop_embedding(self) -> None:
        self._pending_channels = []
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.log.append("중지됨.")

    def save_label_correction(self, path: str, new_level: int) -> None:
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.labels_dir.glob("labels_v*.csv"))
        max_v = max((int(p.stem.split("_v")[-1]) for p in existing), default=-1)
        out = self.labels_dir / f"labels_v{max_v + 1}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "level"])
            writer.writerow([path, str(new_level)])
        self.log.append(f"✅ Saved {out.name}  ({Path(path).name} → L{new_level})")

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_scatter_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        hint = QLabel(
            "채널 선택 후 추출하면 t-SNE 산점도가 여기에 표시됩니다.\n"
            "'전체 비교' 모드: 4채널을 같은 차트에 색으로 구분하여 오버레이."
        )
        hint.setWordWrap(True)
        self._scatter_chart = PlotlyWidget()
        self._scatter_chart.setMinimumHeight(360)
        self._scatter_label = QLabel("선택됨: (없음)")
        v.addWidget(hint)
        v.addWidget(self._scatter_chart, stretch=1)
        v.addWidget(self._scatter_label)
        return w

    def _build_purity_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        hint = QLabel(
            "레벨 순도: 각 레벨의 임베딩 클러스터가 얼마나 응집되어 있는지(intra-cluster distance) 표시."
        )
        hint.setWordWrap(True)
        self._purity_chart = PlotlyWidget()
        self._purity_chart.setMinimumHeight(300)
        v.addWidget(hint)
        v.addWidget(self._purity_chart, stretch=1)
        return w

    def _build_similar_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        hint = QLabel(
            "산점도에서 포인트 클릭 후 이 탭으로 이동하면 가장 유사한 이미지 N개를 표시합니다."
        )
        hint.setWordWrap(True)

        ctrl = QHBoxLayout()
        self._k_spin = QSpinBox()
        self._k_spin.setRange(1, 20)
        self._k_spin.setValue(5)
        search_btn = QPushButton("🔍  유사 이미지 검색")
        search_btn.clicked.connect(self._search_similar)
        ctrl.addWidget(QLabel("K ="))
        ctrl.addWidget(self._k_spin)
        ctrl.addWidget(search_btn)
        ctrl.addStretch()

        self._similar_scroll = QScrollArea()
        self._similar_scroll.setWidgetResizable(True)
        self._similar_widget = QWidget()
        self._similar_layout = QHBoxLayout(self._similar_widget)
        self._similar_layout.setSpacing(8)
        self._similar_scroll.setWidget(self._similar_widget)

        v.addWidget(hint)
        v.addLayout(ctrl)
        v.addWidget(self._similar_scroll)
        return w

    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        self._selected_label = QLabel("선택됨: (없음)")
        self.level_box = QSpinBox()
        self.level_box.setRange(0, self.cfg.get("data", {}).get("num_levels", 6) - 1)
        self.level_box.setMaximumWidth(80)

        save_btn = QPushButton("💾  라벨 교정 저장")
        save_btn.clicked.connect(self._save_correction)

        browse_btn = QPushButton("📂  이미지 직접 선택…")
        browse_btn.clicked.connect(self._browse_for_correction)

        row = QHBoxLayout()
        row.addWidget(QLabel("New Level:"))
        row.addWidget(self.level_box)
        row.addWidget(save_btn)
        row.addWidget(browse_btn)
        row.addStretch()

        self._preview = ImageViewer()
        self._preview.setFixedHeight(160)

        v.addWidget(
            QLabel("<b>산점도 포인트를 클릭하거나 이미지를 직접 선택하세요</b>")
        )
        v.addWidget(self._selected_label)
        v.addWidget(self._preview)
        v.addLayout(row)
        return w

    # ── Private — rendering ───────────────────────────────────────────────────

    def _render_scatter(self) -> None:
        if not self._all_channel_data:
            return
        try:
            import plotly.graph_objects as go

            traces = []
            for ch, d in self._all_channel_data.items():
                pts = d["points"]
                lbs = d["labels"]
                color = _CH_COLORS.get(ch, "#888")
                traces.append(
                    go.Scatter(
                        x=[p[0] for p in pts],
                        y=[p[1] for p in pts],
                        mode="markers",
                        name=f"Ch {ch}",
                        marker={
                            "color": lbs if len(self._all_channel_data) == 1 else color,
                            "colorscale": (
                                "Viridis" if len(self._all_channel_data) == 1 else None
                            ),
                            "size": 6,
                            "opacity": 0.75,
                        },
                        text=[f"L{l}" for l in lbs],
                        hovertemplate="%{text}<extra>Ch " + ch + "</extra>",
                    )
                )
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="t-SNE Embedding Projection",
                template="plotly_dark",
                height=360,
                legend={"orientation": "h"},
            )
            self._scatter_chart.set_figure(fig)
        except Exception as exc:
            self.log.append(f"Scatter render error: {exc}")

    def _render_purity(self, ch: str, labels: list[int]) -> None:
        """Render intra-cluster distance bar chart from embedding data."""
        if ch not in self._all_channel_data:
            return
        try:
            import math

            import plotly.graph_objects as go

            pts = self._all_channel_data[ch]["points"]
            lbs = labels
            num_levels = self.cfg.get("data", {}).get("num_levels", 6)
            purity: dict[int, float] = {}

            for lvl in range(num_levels):
                lvl_pts = [pts[i] for i, l in enumerate(lbs) if l == lvl]
                if len(lvl_pts) < 2:
                    purity[lvl] = 0.0
                    continue
                cx = sum(p[0] for p in lvl_pts) / len(lvl_pts)
                cy = sum(p[1] for p in lvl_pts) / len(lvl_pts)
                dists = [
                    math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in lvl_pts
                ]
                purity[lvl] = sum(dists) / len(dists)

            fig = go.Figure(
                data=go.Bar(
                    x=[f"L{k}" for k in sorted(purity)],
                    y=[purity[k] for k in sorted(purity)],
                    marker_color="#60a5fa",
                )
            )
            fig.update_layout(
                title=f"레벨별 Intra-Cluster Distance [{ch}] (낮을수록 응집)",
                xaxis_title="Level",
                yaxis_title="Mean Distance",
                template="plotly_dark",
                height=280,
                margin={"t": 40},
            )
            self._purity_chart.set_figure(fig)
        except Exception as exc:
            self.log.append(f"Purity render error: {exc}")

    def _search_similar(self) -> None:
        if not self._selected_path or not self._embedding_paths:
            self.log.append("⚠️  먼저 산점도에서 포인트를 선택하세요.")
            return
        try:
            import math

            k = self._k_spin.value()
            # Find embedding index of selected path
            if self._selected_path not in self._embedding_paths:
                return
            sel_idx = self._embedding_paths.index(self._selected_path)

            # Use first available channel data
            ch = next(iter(self._all_channel_data))
            pts = self._all_channel_data[ch]["points"]
            sx, sy = pts[sel_idx]
            dists = [
                (math.sqrt((p[0] - sx) ** 2 + (p[1] - sy) ** 2), i)
                for i, p in enumerate(pts)
            ]
            dists.sort()
            top_k = [self._embedding_paths[i] for _, i in dists[1 : k + 1]]

            # Clear and refill thumbnail grid
            for i in reversed(range(self._similar_layout.count())):
                w = self._similar_layout.itemAt(i).widget()
                if w:
                    w.setParent(None)

            for p in top_k:
                iv = ImageViewer()
                iv.setFixedSize(120, 120)
                iv.load_image(p)
                self._similar_layout.addWidget(iv)
            self._similar_layout.addStretch()
        except Exception as exc:
            self.log.append(f"Similar search error: {exc}")

    # ── Private — corrections ─────────────────────────────────────────────────

    def _on_point_clicked(self, index: int) -> None:
        if 0 <= index < len(self._embedding_paths):
            self._selected_path = self._embedding_paths[index]
            name = Path(self._selected_path).name
            self._scatter_label.setText(f"선택됨: {name}")
            self._selected_label.setText(f"선택됨: {name}")
            self._preview.load_image(self._selected_path)

    def _save_correction(self) -> None:
        path = self._selected_path
        if not path:
            self.log.append("⚠️  이미지를 먼저 선택하세요.")
            return
        self.save_label_correction(path, self.level_box.value())

    def _browse_for_correction(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(self.labels_dir),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self._selected_path = path
            self._selected_label.setText(f"선택됨: {Path(path).name}")
            self._preview.load_image(path)

    def _get_checkpoint(self, channel: str = "Y") -> str:
        from gui.workers._ckpt_utils import auto_find_checkpoint

        if self._settings_tab is not None and hasattr(
            self._settings_tab, "get_checkpoint_path"
        ):
            ckpt = self._settings_tab.get_checkpoint_path()
            if ckpt:
                return ckpt
        return auto_find_checkpoint(self.cfg, channel)
