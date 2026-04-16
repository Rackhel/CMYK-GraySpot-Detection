"""
evaluation/evaluator.py

전체 평가 파이프라인 — 모델 추론 → 지표 계산 → HTML 리포트 생성.
Full evaluation pipeline — model inference -> metric computation -> HTML report.

PRD Section 5.6 에서 정의한 evaluation/ 모듈의 최상위 진입점이다.
Top-level entry point for the evaluation/ module defined in PRD Section 5.6.

의존성 / Dependencies:
    - utils/logger.py  : LoggerMixin, get_logger (R6 담당)
    - evaluation/metrics.py, confusion.py : 이 패키지 내부
    - cv2, numpy, torch, plotly

data/dataset.py (R1 담당) 와의 관계 / Relationship with data/dataset.py (R1):
    R1 의 CMYKDataset 은 wide-format CSV + image_dir 기반으로 설계되어 있다.
    R1's CMYKDataset is designed for wide-format CSV + image_dir.
    evaluator 는 long-format + labeled/{color}/{level}/ 경로 구조를 사용하므로
    내부에 _EvalDataset 을 별도로 구현한다.
    Since evaluator uses long-format + labeled/{color}/{level}/ path structure,
    it implements _EvalDataset internally.
    S3 이후 R1 과 협의하여 통합 여부를 결정한다.            
    Integration with R1's dataset will be decided after S3.

사용 예 / Usage example:
    from evaluation.evaluator import Evaluator

    ev = Evaluator(model, labeled_dir, labels_csv, output_dir, device)
    results = ev.run(channels=['C', 'K'])
    metrics = ev.compute(results)
    ev.save_report(results, metrics, experiment_name='baseline')
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots

# ── Internal modules / 내부 모듈 ──────────────────────────────────────────
from utils import get_logger, LoggerMixin
from .metrics import (
from torch.utils.data import Dataset, DataLoader

# utils/logger.py 연동 / Integration with utils/logger.py
# sys.path 에 src/ 가 있다고 가정한다 (scripts/evaluate.py 또는 conftest.py 에서 추가)
# Assumes src/ is in sys.path (added by scripts/evaluate.py or conftest.py)
try:
    from utils.logger import LoggerMixin, get_logger
except ImportError:
    # logger.py 가 아직 경로에 없을 경우 fallback
    # Fallback if logger.py is not yet in path
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
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    TARGET_MAE,
    CONF_THRESH_AUTO,
    CONF_THRESH_WARN,
    CONF_THRESH_MANUAL,
    compute_all_channels,
    check_targets,
    print_summary,
)
from evaluation.confusion import plot_confusion_matrix, plot_all_channels


# ---------------------------------------------------------------------------
# Internal Dataset / 내부 Dataset
# ---------------------------------------------------------------------------

class _EvalDataset(Dataset):
    """
    평가 전용 Dataset. evaluator.py 내부에서만 사용한다.
    Evaluation-only Dataset. Used internally by evaluator.py only.

    R1 의 CMYKDataset 과의 차이 / Difference from R1's CMYKDataset:
        CMYKDataset  : wide-format CSV + image_dir (단일 디렉토리)
        _EvalDataset : long-format DataFrame + labeled/{color}/{level}/ (계층 구조)
        CMYKDataset  : wide-format CSV + single image_dir
        _EvalDataset : long-format DataFrame + labeled/{color}/{level}/ (hierarchical)

    이미지 로딩 / Image loading: cv2, IMAGE_SIZE=128 (02/03/05 notebook 과 동일)
    Image loading: cv2, IMAGE_SIZE=128 (same as 02/03/05 notebooks)
    """

    def __init__(self, df: pd.DataFrame, patch_dir: Path, image_size: int):
        self.df         = df.reset_index(drop=True)
        self.patch_dir  = Path(patch_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        color = row['color']
        fname = row['filename']
        level = int(row['level'])

        img_path = self.patch_dir / color / str(level) / fname

        if not img_path.exists():
            raise FileNotFoundError(
                f'Image not found / 이미지 없음: {img_path}'
            )

        # cv2 loading — 02/03/05 notebook 방식과 동일
        # cv2 loading — identical to 02/03/05 notebooks
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0

        # (H, W, 3) -> (3, H, W) tensor
        tensor = torch.tensor(img).permute(2, 0, 1).float()
        return tensor, level, fname


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator(LoggerMixin):
    """
    전체 평가 파이프라인을 관리하는 클래스.
    Class that manages the full evaluation pipeline.

    LoggerMixin 을 상속하여 self.logger 를 자동으로 갖는다.
    Inherits LoggerMixin so self.logger is available automatically.

    PRD Section 5.6:
        - run()         : 모델 추론 / Run model inference
        - compute()     : 지표 계산 / Compute metrics
        - save_report() : HTML 리포트 생성 / Generate HTML report
        - save_csv()    : CSV 결과 저장 / Save CSV results
        - save_json()   : JSON 지표 저장 / Save JSON metrics

    Args:
        model       : 평가할 PyTorch 모델 / PyTorch model to evaluate
        labeled_dir : data_set/labeled/ 경로 / Path to data_set/labeled/
        labels_csv  : data_set/labels_v0.csv 경로 / Path to labels_v0.csv
        output_dir  : 리포트 저장 경로 / Report output directory
        device      : 연산 디바이스 / Compute device
        image_size  : 이미지 크기 — 02/03/05 와 동일하게 128
                      Image size — 128, same as 02/03/05 notebooks
        batch_size  : DataLoader 배치 크기 / DataLoader batch size
        num_levels  : Level 수 (0~5) / Number of levels
    """

    LEVEL_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#1a1a2e']
    CMYK_COLORS  = {'Y': '#f5e642', 'M': '#e91e8c', 'C': '#00b4d8', 'K': '#444444'}

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
    ):
        self.model       = model
        self.labeled_dir = Path(labeled_dir)
        self.labels_csv  = Path(labels_csv)
        self.output_dir  = Path(output_dir)
        self.device      = device
        self.image_size  = image_size
        self.batch_size  = batch_size
        self.num_levels  = num_levels

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Evaluator initialized / 초기화 완료  output_dir={self.output_dir}')

    # ------------------------------------------------------------------
    # Label loading / 라벨 로드
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_color(fname: str) -> Optional[str]:
        """
        파일명에서 색상 코드를 추출한다.
        Extracts color code from filename.
        e.g. 'scan_001_C_0007.png' -> 'C'
        """
        for part in Path(fname).stem.split('_'):
            if part in ('Y', 'M', 'C', 'K'):
                return part
        return None

    def load_labels(self) -> pd.DataFrame:
        """
        labels_v0.csv 를 long-format DataFrame 으로 변환한다.
        Converts labels_v0.csv to a long-format DataFrame.

        파일명 색상과 일치하는 라벨만 포함한다.
        Only labels matching the color in the filename are included.
        e.g. scan_001_C_0007.png -> C 라벨만 유효 / Only C label valid

        R1 의 CMYKDataset 이 wide-format 을 그대로 읽는 것과 달리,
        evaluator 는 채널별 독립 추론을 위해 long-format 으로 변환한다.
        Unlike R1's CMYKDataset which reads wide-format directly,
        evaluator converts to long-format for per-channel inference.
        """
        df       = pd.read_csv(self.labels_csv)
        cols_map = {'Y': 'Y', 'M': 'M', 'C': 'C', 'K': 'K'}

        records = []
        skipped = 0
        for _, row in df.iterrows():
            color = self._extract_color(str(row['filename']))
            if color is None:
                skipped += 1
                continue
            col = cols_map.get(color)
            if col is None or col not in df.columns:
                skipped += 1
                continue
            records.append({
                'filename': row['filename'],
                'color'   : color,
                'level'   : int(row[col]),
            })

        long_df = pd.DataFrame(records)
        self.logger.info(
            f'Labels loaded / 라벨 로드: {len(long_df)} rows  (skipped / 제외: {skipped})'
        )
        return long_df

    # ------------------------------------------------------------------
    # Inference / 추론
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_single_channel(
        self,
        df_color : pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        단일 채널 추론을 실행한다.
        Runs inference for a single channel.

        Returns:
            y_true, y_pred, confidences, filenames
        """
        self.model.eval()
        ds = _EvalDataset(df_color, self.labeled_dir, self.image_size)
        loader = DataLoader(
            ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = 0,  # 0 = Windows/macOS 안전 / safe on Windows/macOS
            pin_memory  = (self.device.type == 'cuda'),
        )

        all_true, all_pred, all_conf, all_fnames = [], [], [], []

        for batch_imgs, batch_labels, batch_fnames in loader:
            batch_imgs  = batch_imgs.to(self.device, non_blocking=True)
            logits      = self.model(batch_imgs)
            probs       = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            all_true.extend(batch_labels.numpy())
            all_pred.extend(preds.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())
            all_fnames.extend(batch_fnames)

        return (
            np.array(all_true),
            np.array(all_pred),
            np.array(all_conf),
            all_fnames,
        )

    def run(self, channels: List[str] = None) -> Dict[str, dict]:
        """
        지정한 채널에 대해 모델 추론을 실행한다.
        Runs model inference for the specified channels.

        Args:
            channels : 처리할 채널 목록 / Channel list (default: ['Y','M','C','K'])

        Returns:
            results : {
                'Y': {'y_true', 'y_pred', 'confidences', 'filenames'},
                ...
            }
        """
        if channels is None:
            channels = ['Y', 'M', 'C', 'K']

        df_labels = self.load_labels()
        results: Dict[str, dict] = {}

        for color in channels:
            df_color = df_labels[df_labels['color'] == color].reset_index(drop=True)

            if len(df_color) == 0:
                self.logger.warning(f'[{color}] No samples / 샘플 없음 — skipping')
                continue

            y_true, y_pred, confs, fnames = self._run_single_channel(df_color)
            results[color] = {
                'y_true'      : y_true,
                'y_pred'      : y_pred,
                'confidences' : confs,
                'filenames'   : fnames,
            }

            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_true, y_pred)
            self.logger.info(f'[{color}] {len(y_true):4d} samples | Accuracy: {acc:.4f}')

        self.logger.info(
            f'Inference complete / 추론 완료 — channels: {list(results.keys())}'
        )
        return results

    # ------------------------------------------------------------------
    # Metrics / 지표
    # ------------------------------------------------------------------

    def compute(
        self,
        results  : Dict[str, dict],
        channels : List[str] = None,
    ) -> Dict[str, dict]:
        """
        추론 결과로부터 지표를 계산한다.
        Computes metrics from inference results.

        Args:
            results  : run() 반환값 / Return value of run()
            channels : 처리할 채널 / Channels to process

        Returns:
            metrics : compute_all_channels() 반환값
        """
        metrics = compute_all_channels(results, channels, self.num_levels)
        print_summary(metrics, channels=[c for c in metrics if c != 'overall'])
        return metrics

    # ------------------------------------------------------------------
    # Misclassified samples / 오분류 샘플
    # ------------------------------------------------------------------

    def get_misclassified(
        self,
        results  : Dict[str, dict],
        channels : List[str] = None,
    ) -> pd.DataFrame:
        """
        오분류 샘플 목록을 DataFrame 으로 반환한다.
        Returns misclassified samples as a DataFrame.
        """
        if channels is None:
            channels = list(results.keys())

        records = []
        for color in channels:
            if color not in results:
                continue
            yt     = results[color]['y_true']
            yp     = results[color]['y_pred']
            confs  = results[color]['confidences']
            fnames = results[color]['filenames']

            for i in np.where(yt != yp)[0]:
                records.append({
                    'filename'  : fnames[i],
                    'color'     : color,
                    'true_level': int(yt[i]),
                    'pred_level': int(yp[i]),
                    'confidence': round(float(confs[i]), 4),
                    'correct'   : False,
                    'error_gap' : int(abs(yt[i] - yp[i])),
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values(['error_gap', 'confidence'], ascending=[False, True])
        return df

    # ------------------------------------------------------------------
    # Save helpers / 저장 유틸리티
    # ------------------------------------------------------------------

    def save_csv(
        self,
        results         : Dict[str, dict],
        experiment_name : str = 'eval',
        channels        : List[str] = None,
    ) -> Path:
        """
        샘플별 예측 결과를 CSV 로 저장한다. (PRD 8.2.2)
        Saves per-sample predictions as CSV. (PRD 8.2.2)
        """
        if channels is None:
            channels = list(results.keys())

        rows = []
        for color in channels:
            if color not in results:
                continue
            yt     = results[color]['y_true']
            yp     = results[color]['y_pred']
            confs  = results[color]['confidences']
            fnames = results[color]['filenames']
            for i in range(len(yt)):
                rows.append({
                    'filename'  : fnames[i],
                    'color'     : color,
                    'true_level': int(yt[i]),
                    'pred_level': int(yp[i]),
                    'confidence': round(float(confs[i]), 4),
                    'correct'   : bool(yt[i] == yp[i]),
                    'error_gap' : int(abs(yt[i] - yp[i])),
                })

        path = self.output_dir / f'evaluation_results_{experiment_name}.csv'
        # UTF-8 BOM: Windows Excel 한글 깨짐 방지 / Prevents Korean garbling in Windows Excel
        pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
        self.logger.info(f'CSV saved / 저장: {path}  ({len(rows)} rows)')
        return path

    def save_json(
        self,
        metrics         : Dict[str, dict],
        experiment_name : str = 'eval',
        channels        : List[str] = None,
        checkpoint_path : Optional[str] = None,
    ) -> Path:
        """
        집계된 지표를 JSON 으로 저장한다. (PRD 8.2.2)
        Saves aggregated metrics as JSON. (PRD 8.2.2)
        """
        if channels is None:
            channels = [k for k in metrics if k != 'overall']

        targets = check_targets(metrics, channels)

        export = {
            'meta': {
                'experiment' : experiment_name,
                'checkpoint' : checkpoint_path,
                'image_size' : self.image_size,
                'num_levels' : self.num_levels,
            },
            'targets': {
                'overall_accuracy'  : TARGET_OVERALL_ACC,
                'per_color_accuracy': TARGET_PER_COLOR_ACC,
                'per_class_f1'      : TARGET_PER_CLASS_F1,
                'mae'               : TARGET_MAE,
            },
            'global': {
                'accuracy': round(metrics['overall']['accuracy'], 4),
                'macro_f1': round(metrics['overall']['macro_f1'], 4),
                'mae'     : round(metrics['overall']['mae'],      4),
                'acc_pass': targets['overall']['acc_pass'],
                'f1_pass' : targets['overall']['f1_pass'],
                'mae_pass': targets['overall']['mae_pass'],
                'all_pass': targets['overall']['all_pass'],
            },
            'by_color': {
                color: {
                    'accuracy': round(metrics[color]['accuracy'], 4),
                    'macro_f1': round(metrics[color]['macro_f1'], 4),
                    'mae'     : round(metrics[color]['mae'],      4),
                    'acc_pass': targets[color]['acc_pass'],
                    'all_pass': targets[color]['all_pass'],
                }
                for color in channels if color in metrics
            },
            'per_class_overall': [
                {
                    'level'    : pc['level'],
                    'precision': round(pc['precision'], 4),
                    'recall'   : round(pc['recall'],    4),
                    'f1'       : round(pc['f1'],        4),
                    'f1_pass'  : bool(pc['f1'] >= TARGET_PER_CLASS_F1),
                }
                for pc in metrics['overall']['per_class']
            ],
        }

        path = self.output_dir / f'metrics_summary_{experiment_name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        self.logger.info(f'JSON saved / 저장: {path}')
        return path

    # ------------------------------------------------------------------
    # Chart builders / 차트 생성
    # ------------------------------------------------------------------

    def _build_dashboard(self, metrics: Dict[str, dict], channels: List[str]) -> go.Figure:
        """
        Gauge + Bar 평가 대시보드를 생성한다.
        Builds a Gauge + Bar evaluation dashboard.
        """
        m_all     = metrics['overall']
        valid_chs = [c for c in channels if c in metrics]

        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'bar', 'colspan': 3}, None, None],
            ],
            subplot_titles=[
                'Overall Accuracy',
                'Overall Macro F1',
                'Overall MAE (lower is better / 낮을수록 좋음)',
                'Per-Color Accuracy / 색상별 정확도',
            ],
            vertical_spacing=0.18,
        )

        fig.add_trace(go.Indicator(
            mode='gauge+number',
            value=round(m_all['accuracy'] * 100, 2),
            number={'suffix': '%', 'font': {'size': 30}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar' : {'color': '#50e3c2'},
                'threshold': {'line': {'color': '#ff7aa2', 'width': 3},
                              'value': TARGET_OVERALL_ACC * 100},
                'bgcolor': '#0b1220',
            },
            title={'text': f'Target >= {TARGET_OVERALL_ACC:.0%}'},
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode='gauge+number',
            value=round(m_all['macro_f1'], 4),
            number={'font': {'size': 30}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar' : {'color': '#66d9ff'},
                'threshold': {'line': {'color': '#ff7aa2', 'width': 3},
                              'value': TARGET_PER_CLASS_F1},
                'bgcolor': '#0b1220',
            },
            title={'text': f'Target >= {TARGET_PER_CLASS_F1:.2f}'},
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode='gauge+number',
            value=round(m_all['mae'], 4),
            number={'font': {'size': 30}},
            gauge={
                'axis': {'range': [0, 3]},
                'bar' : {'color': '#c792ea'},
                'threshold': {'line': {'color': '#ffb347', 'width': 3},
                              'value': TARGET_MAE},
                'bgcolor': '#0b1220',
            },
            title={'text': f'Target <= {TARGET_MAE:.2f}'},
        ), row=1, col=3)

        acc_vals = [metrics[c]['accuracy'] * 100 for c in valid_chs]
        fig.add_trace(go.Bar(
            x=valid_chs,
            y=acc_vals,
            marker_color=[self.CMYK_COLORS[c] for c in valid_chs],
            text=[f'{v:.2f}%' for v in acc_vals],
            textposition='outside',
            name='Accuracy',
        ), row=2, col=1)

        if valid_chs:
            fig.add_shape(
                type='line', x0=-0.5, x1=len(valid_chs) - 0.5,
                y0=TARGET_PER_COLOR_ACC * 100, y1=TARGET_PER_COLOR_ACC * 100,
                line=dict(color='#ff7aa2', dash='dash'),
                xref='x', yref='y', row=2, col=1,
            )
            fig.add_annotation(
                x=valid_chs[-1], y=TARGET_PER_COLOR_ACC * 100,
                text=f'Target {TARGET_PER_COLOR_ACC:.0%}',
                showarrow=False, yshift=10, row=2, col=1,
            )

        fig.update_layout(
            title=dict(text='Grayspot Evaluation Dashboard / 평가 대시보드', font=dict(size=17)),
            template='plotly_dark',
            font=dict(family='Segoe UI', size=12),
            height=750, showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def _build_per_class_chart(self, metrics: Dict[str, dict]) -> go.Figure:
        """클래스별 F1 막대 차트 / Per-class F1 bar chart."""
        pc     = metrics['overall']['per_class']
        levels = [f"Level {d['level']}" for d in pc]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='F1',        x=levels, y=[d['f1']        for d in pc], marker_color='#1976D2'))
        fig.add_trace(go.Bar(name='Precision', x=levels, y=[d['precision'] for d in pc], marker_color='#388E3C'))
        fig.add_trace(go.Bar(name='Recall',    x=levels, y=[d['recall']    for d in pc], marker_color='#F57C00'))
        fig.add_hline(
            y=TARGET_PER_CLASS_F1, line_dash='dash', line_color='#ff7aa2',
            annotation_text=f'F1 Target >= {TARGET_PER_CLASS_F1:.2f}',
        )
        fig.update_layout(
            title=dict(text='Per-Class Metrics (Overall) / 클래스별 지표', font=dict(size=15)),
            barmode='group',
            yaxis=dict(range=[0, 1.15], title='Score'),
            xaxis=dict(title='Level'),
            template='plotly_dark',
            font=dict(family='Segoe UI', size=12),
            legend=dict(orientation='h', y=1.1),
            height=480,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def _build_mae_heatmap(
        self,
        results  : Dict[str, dict],
        channels : List[str],
    ) -> go.Figure:
        """(color x level) MAE 히트맵 / MAE heatmap by (color x level)."""
        valid_chs    = [c for c in channels if c in results]
        level_names  = [f'Level {i}' for i in range(self.num_levels)]
        mae_matrix   = np.full((len(valid_chs), self.num_levels), np.nan)
        count_matrix = np.zeros((len(valid_chs), self.num_levels), dtype=int)

        for ci, color in enumerate(valid_chs):
            yt = results[color]['y_true']
            yp = results[color]['y_pred']
            for lv in range(self.num_levels):
                mask = yt == lv
                if mask.sum() > 0:
                    mae_matrix[ci, lv]   = float(np.mean(np.abs(yt[mask] - yp[mask])))
                    count_matrix[ci, lv] = int(mask.sum())

        annot = [
            [
                f'{mae_matrix[r, c]:.2f}\n(n={count_matrix[r, c]})'
                if not np.isnan(mae_matrix[r, c]) else 'N/A'
                for c in range(self.num_levels)
            ]
            for r in range(len(valid_chs))
        ]

        fig = go.Figure(go.Heatmap(
            z=mae_matrix, x=level_names, y=valid_chs,
            text=annot, texttemplate='%{text}',
            colorscale='YlOrRd', zmin=0, zmax=2.0,
            colorbar=dict(title='MAE'),
            hovertemplate='Color: %{y}<br>Level: %{x}<br>MAE: %{z:.3f}<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text=f'MAE per (Color x True Level) — Target <= {TARGET_MAE}', font=dict(size=15)),
            xaxis=dict(title='True Level / 실제 레벨'),
            yaxis=dict(title='Color Channel / 색상 채널'),
            template='plotly_dark',
            font=dict(family='Segoe UI', size=12),
            height=360,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def _build_mismatch_scatter(self, df_miss: pd.DataFrame) -> go.Figure:
        """오분류 scatter / Misclassified samples scatter."""
        if len(df_miss) == 0:
            return go.Figure()

        fig = px.scatter(
            df_miss,
            x='true_level', y='pred_level',
            color='color', color_discrete_map=self.CMYK_COLORS,
            size='error_gap', size_max=20,
            hover_data=['filename', 'color', 'true_level', 'pred_level', 'confidence', 'error_gap'],
            title='Misclassified Samples / 오분류 샘플 — True vs Predicted Level',
            labels={
                'true_level': 'True Level / 실제 레벨',
                'pred_level': 'Predicted Level / 예측 레벨',
                'color'     : 'CMYK Channel',
            },
            template='plotly_dark',
            width=680, height=540,
        )
        fig.add_trace(go.Scatter(
            x=[0, self.num_levels - 1], y=[0, self.num_levels - 1],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            name='Correct boundary / 정답 경계',
            showlegend=True,
        ))
        fig.update_layout(
            font=dict(family='Segoe UI', size=12),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def _build_confidence_dist(
        self,
        results  : Dict[str, dict],
        channels : List[str],
    ) -> go.Figure:
        """신뢰도 분포 히스토그램 / Confidence distribution histogram."""
        valid_chs = [c for c in channels if c in results]
        n_rows    = (len(valid_chs) + 1) // 2

        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=[f'[{c}]' for c in valid_chs],
            horizontal_spacing=0.10,
            vertical_spacing=0.18,
        )
        bins = dict(start=0, end=1, size=0.04)

        for i, color in enumerate(valid_chs):
            r  = i // 2 + 1
            c  = i % 2  + 1
            yt = results[color]['y_true']
            yp = results[color]['y_pred']
            cf = results[color]['confidences']

            fig.add_trace(go.Histogram(
                x=cf[yt == yp], xbins=bins,
                name='Correct / 정답', marker_color='#4fc3f7',
                opacity=0.70, showlegend=(i == 0), legendgroup='correct',
            ), row=r, col=c)

            fig.add_trace(go.Histogram(
                x=cf[yt != yp], xbins=bins,
                name='Wrong / 오답', marker_color='#ef5350',
                opacity=0.70, showlegend=(i == 0), legendgroup='wrong',
            ), row=r, col=c)

            for thresh, col_color in [
                (CONF_THRESH_AUTO,   'green'),
                (CONF_THRESH_WARN,   'orange'),
                (CONF_THRESH_MANUAL, 'red'),
            ]:
                fig.add_vline(
                    x=thresh, line_dash='dash',
                    line_color=col_color, line_width=1.5,
                    row=r, col=c,
                )

        fig.update_layout(
            title=dict(
                text='Confidence Distribution / 신뢰도 분포 — Correct vs Wrong',
                font=dict(size=15),
            ),
            barmode='overlay',
            template='plotly_dark',
            font=dict(family='Segoe UI', size=12),
            height=640,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def _build_phase3_decision(
        self,
        metrics  : Dict[str, dict],
        channels : List[str],
    ) -> str:
        """
        PRD 3.3.2 피드백 복귀 판단 텍스트를 생성한다.
        Generates PRD 3.3.2 feedback-loop decision text.
        """
        targets     = check_targets(metrics, channels)
        decisions   = []
        overall_mae = metrics['overall']['mae']
        overall_acc = metrics['overall']['accuracy']
        overall_mf1 = metrics['overall']['macro_f1']

        # 검사 1: 색상별 정확도 < 0.80 -> Phase 0
        # Check 1: per-color accuracy < 0.80 -> Phase 0
        for color in channels:
            if color not in metrics:
                continue
            acc = metrics[color]['accuracy']
            if acc < 0.80:
                decisions.append(
                    f'[{color}] Accuracy {acc:.3f} < 0.80'
                    ' -> Phase 0 (retrain representation / 표현 재학습)'
                )

        # 검사 2: 클래스별 F1 < 0.70 -> Phase 1
        # Check 2: per-class F1 < 0.70 -> Phase 1
        for pc in metrics['overall']['per_class']:
            if pc['f1'] < 0.70:
                decisions.append(
                    f"Level {pc['level']} F1={pc['f1']:.3f} < 0.70"
                    ' -> Phase 1 (review level boundary / 레벨 경계 재검토)'
                )

        # 검사 3: MAE > 0.80 -> Phase 0
        # Check 3: MAE > 0.80 -> Phase 0
        if overall_mae > 0.80:
            decisions.append(
                f'Overall MAE {overall_mae:.3f} > 0.80'
                ' -> Phase 0 (representation learning retry / 표현 학습 재시도)'
            )

        lines = [
            '=== Phase 3 Feedback Decision / Phase 3 피드백 복귀 판단 ===',
        ]

        all_color_ok = all(
            targets.get(c, {}).get('acc_pass', False) for c in channels
        )
        if targets['overall']['all_pass'] and all_color_ok:
            lines.append('All targets met -- TERMINATE Swing / 모든 목표 달성 -- Swing 종료')
        elif not decisions:
            lines.append('No critical failures -- continue / 심각한 실패 없음 -- 계속 진행')
        else:
            lines.append('Action required / 조치 필요:')
            lines.extend(f'  {d}' for d in decisions)

        lines += [
            '',
            f'  Overall Accuracy : {overall_acc:.4f}  (target >= {TARGET_OVERALL_ACC})',
            f'  Overall Macro F1 : {overall_mf1:.4f}  (target >= {TARGET_PER_CLASS_F1})',
            f'  Overall MAE      : {overall_mae:.4f}  (target <= {TARGET_MAE})',
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Full report / 전체 리포트
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
            confidence_distribution.html   — 신뢰도 분포
            evaluation_results_{name}.csv
            misclassified_{name}.csv
            metrics_summary_{name}.json

        Args:
            results         : run() 반환값
            metrics         : compute() 반환값
            experiment_name : 실험 이름 (파일명 접미사) / Experiment name (filename suffix)
            channels        : 처리할 채널 / Channels to process
            open_browser    : 대시보드를 브라우저로 자동 열기 / Auto-open dashboard
            checkpoint_path : 체크포인트 경로 (JSON 메타데이터용)

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
