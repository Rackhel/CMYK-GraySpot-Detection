"""
evaluation/evaluator.py

책임 / Responsibility: Evaluator 조율자 (Orchestrator) — 상태 초기화 + save_report 진입점
Responsibility: Evaluator Orchestrator — state initialisation + save_report entry point

이 파일은 4개 Mixin 을 조합하여 Evaluator 공개 인터페이스를 구성한다.
This file composes 4 Mixins to form the public Evaluator interface.

    InferenceMixin  → load_labels, _run_single_channel, run
    MetricsMixin    → compute, get_misclassified
    ExportMixin     → save_csv, save_json
    ChartsMixin     → _build_dashboard, _build_per_class_chart, _build_mae_heatmap,
                      _build_mismatch_scatter, _build_confidence_dist, _build_phase3_decision

외부 API / External API (Contract §7 에 동결됨 / frozen per Contract §7):
    from evaluation.evaluator import Evaluator

    ev = Evaluator(model, labeled_dir, labels_csv, output_dir, device)
    results = ev.run(channels=['C', 'K'])
    metrics = ev.compute(results)
    ev.save_report(results, metrics, experiment_name='baseline')
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import torch
import torch.nn as nn

try:
    from utils.logger import LoggerMixin, get_logger
except ImportError:
    import logging

    class LoggerMixin:                          # type: ignore[no-redef]
        @property
        def logger(self):
            if not hasattr(self, '_logger'):
                self._logger = logging.getLogger(self.__class__.__name__)
            return self._logger

    def get_logger(name=None):                  # type: ignore[misc]
        return logging.getLogger(name or 'evaluation')

from evaluation.metrics import (
    NUM_LEVELS,
    CONF_THRESH_AUTO,
    CONF_THRESH_WARN,
    CONF_THRESH_MANUAL,
)
from evaluation.confusion import plot_all_channels
from evaluation.evaluator_inference import InferenceMixin
from evaluation.evaluator_metrics import MetricsMixin
from evaluation.evaluator_export import ExportMixin
from evaluation.evaluator_charts import ChartsMixin

_DEFAULT_CONF_AUTO   = CONF_THRESH_AUTO
_DEFAULT_CONF_WARN   = CONF_THRESH_WARN
_DEFAULT_CONF_MANUAL = CONF_THRESH_MANUAL


class Evaluator(InferenceMixin, MetricsMixin, ExportMixin, ChartsMixin, LoggerMixin):
    """
    전체 평가 파이프라인 조율자.
    Full evaluation pipeline orchestrator.

    상태를 초기화하고 save_report() 로 전체 파이프라인을 조율한다.
    Initialises state and orchestrates the full pipeline via save_report().

    Args:
        model       : 평가할 PyTorch 모델 (model.eval() 상태) / PyTorch model (must be in model.eval())
        labeled_dir : data_set/labeled/ 경로 / Path to data_set/labeled/
        labels_csv  : data_set/labels_v0.csv 경로 / Path to labels_v0.csv
        output_dir  : 리포트 저장 경로 / Report output directory
        device      : 연산 디바이스 / Compute device
        image_size  : 이미지 크기 (기본 128) / Image size (default 128)
        batch_size  : DataLoader 배치 크기 / DataLoader batch size
        num_levels  : Level 수 (0~5) / Number of levels
        cfg         : 신뢰도 임계값 및 swing 임계값 주입 / Confidence and swing threshold injection
    """

    def __init__(
        self,
        model       : nn.Module,
        labeled_dir : Path,
        labels_csv  : Path,
        output_dir  : Path,
        device      : torch.device,
        image_size  : int = 128,
        batch_size  : int = 32,
        num_levels  : int = NUM_LEVELS,
        cfg         : Optional[dict] = None,
    ):
        self.model       = model
        self.labeled_dir = Path(labeled_dir)
        self.labels_csv  = Path(labels_csv)
        self.output_dir  = Path(output_dir)
        self.device      = device
        self.image_size  = image_size
        self.batch_size  = batch_size
        self.num_levels  = num_levels
        self.cfg         = cfg or {}

        ct = self.cfg.get("inference", {}).get("confidence_thresholds", {})
        self.conf_thresh_auto   = float(ct.get("auto_accept",    _DEFAULT_CONF_AUTO))
        self.conf_thresh_warn   = float(ct.get("warn_threshold", _DEFAULT_CONF_WARN))
        self.conf_thresh_manual = float(ct.get("manual_review",  _DEFAULT_CONF_MANUAL))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Evaluator initialized / 초기화 완료  output_dir={self.output_dir}')

    # ------------------------------------------------------------------
    # Full report orchestration / 전체 리포트 조율
    # ------------------------------------------------------------------

    def save_report(
        self,
        results         : Dict[str, dict],
        metrics         : Dict[str, dict],
        experiment_name : str = 'eval',
        channels        : List[str] = None,
        open_browser    : bool = False,
        checkpoint_path : Optional[str] = None,
    ) -> Path:
        """
        전체 HTML 리포트를 생성하고 저장한다. (PRD 8.2.3)
        Generates and saves the full HTML report. (PRD 8.2.3)

        생성 파일 목록 / Generated files:
            confusion/cm_{channel}.html     — 채널별 혼동 행렬
            confusion/cm_overall.html       — 전체 혼동 행렬
            eval_dashboard.html             — Gauge + Bar 대시보드
            per_class_metrics.html          — 클래스별 F1
            mae_heatmap.html                — MAE 히트맵
            misclassified_scatter.html      — 오분류 scatter
            confidence_distribution.html    — 신뢰도 분포
            evaluation_results_{name}.csv
            misclassified_{name}.csv
            metrics_summary_{name}.json

        Returns:
            Path to eval_dashboard.html
        """
        if channels is None:
            channels = [c for c in ['Y', 'M', 'C', 'K'] if c in results]

        self.logger.info(f'Generating report / 리포트 생성 중... experiment={experiment_name}')

        def _save(fig: go.Figure, name: str) -> Path:
            path = self.output_dir / name
            fig.write_html(str(path), include_plotlyjs='cdn')
            self.logger.info(f'Saved / 저장: {path}')
            return path

        # 1. Confusion matrices / 혼동 행렬
        plot_all_channels(results, self.output_dir / 'confusion', channels, normalize=True)

        # 2. Dashboard / 대시보드
        dash_path = _save(self._build_dashboard(metrics, channels), 'eval_dashboard.html')

        # 3. Per-class F1
        _save(self._build_per_class_chart(metrics), 'per_class_metrics.html')

        # 4. MAE heatmap / MAE 히트맵
        _save(self._build_mae_heatmap(results, channels), 'mae_heatmap.html')

        # 5. Misclassified / 오분류
        df_miss   = self.get_misclassified(results, channels)
        miss_path = self.output_dir / f'misclassified_{experiment_name}.csv'
        df_miss.to_csv(miss_path, index=False, encoding='utf-8-sig')
        self.logger.info(f'Saved / 저장: {miss_path}')
        _save(self._build_mismatch_scatter(df_miss), 'misclassified_scatter.html')

        # 6. Confidence distribution / 신뢰도 분포
        _save(self._build_confidence_dist(results, channels), 'confidence_distribution.html')

        # 7. CSV + JSON
        self.save_csv(results, experiment_name, channels)
        self.save_json(metrics, experiment_name, channels, checkpoint_path)

        # 8. Phase 3 decision / Phase 3 판단
        decision_text = self._build_phase3_decision(metrics, channels)
        self.logger.info('\n' + decision_text)

        if open_browser:
            webbrowser.open(dash_path.resolve().as_uri())

        self.logger.info(f'Report complete / 리포트 완료: {self.output_dir}')
        return dash_path
