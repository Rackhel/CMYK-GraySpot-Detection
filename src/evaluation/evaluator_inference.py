"""
evaluation/evaluator_inference.py

책임 / Responsibility: 라벨 로딩 + 모델 추론 (run)
Responsibility: Label loading + model inference (run)

SRP 준수: 이 모듈은 "데이터 로딩과 모델 추론"만 담당한다.
SRP compliant: this module handles only "data loading and model inference".

_EvalDataset은 data/dataset.py 에서 관리된다 (데이터 계층 단일 책임).
_EvalDataset is managed in data/dataset.py (single responsibility for the data layer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from data.dataset import _EvalDataset  # 평가 전용 Dataset — 데이터 계층에서 관리


# ---------------------------------------------------------------------------
# InferenceMixin
# ---------------------------------------------------------------------------


class InferenceMixin:
    """
    라벨 로딩과 모델 추론을 담당하는 Mixin.
    Mixin responsible for label loading and model inference.

    사용하는 self 속성 / Consumed self attributes (provided by Evaluator.__init__):
        self.model, self.labeled_dir, self.labels_csv,
        self.device, self.image_size, self.batch_size, self.logger
    """

    @staticmethod
    def _extract_color(fname: str) -> Optional[str]:
        """파일명에서 CMYK 색상 코드를 추출한다. / Extracts CMYK color code from filename."""
        for part in Path(fname).stem.split("_"):
            if part in ("Y", "M", "C", "K"):
                return part
        return None

    def load_labels(self) -> pd.DataFrame:
        """
        CSV 파일을 읽어 long-format DataFrame 으로 반환한다.
        Reads a CSV file and returns it as a long-format DataFrame.

        두 가지 CSV 형식을 자동 감지한다 / Auto-detects two CSV formats:

          [형식 A / Format A] labels_master.csv (long-format) — 권장 / Recommended
            columns: filepath, channel, level
            → filepath 에서 파일명을 추출해 filename 컬럼 생성
            → channel 컬럼을 color 컬럼으로 복사

          [형식 B / Format B] labels_v0.csv (wide-format) — 레거시 / Legacy
            columns: filename, C, M, Y, K
            → 파일명에서 색상 코드를 추출해 long-format 으로 변환

        Returns:
            pd.DataFrame — columns: ["filename", "color", "level"]
        """
        df = pd.read_csv(self.labels_csv)

        # ── 형식 감지 / Format detection ──────────────────────────────────────
        is_long_format = "filepath" in df.columns and "channel" in df.columns

        if is_long_format:
            return self._load_labels_long(df)
        else:
            return self._load_labels_wide(df)

    def _load_labels_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        long-format CSV (labels_master.csv) 처리.
        Processes long-format CSV (labels_master.csv).

        columns: filepath, channel, level
        """
        records = []
        skipped = 0

        for _, row in df.iterrows():
            filepath = str(row["filepath"])
            color = str(row["channel"]).strip().upper()
            if color not in ("Y", "M", "C", "K"):
                skipped += 1
                continue
            try:
                level = int(row["level"])
            except (ValueError, TypeError):
                skipped += 1
                continue

            filename = Path(filepath).name
            records.append({
                "filename": filename,
                "color": color,
                "level": level,
            })

        long_df = pd.DataFrame(records)
        self.logger.info(
            f"Labels loaded (long-format) / 라벨 로드: "
            f"{len(long_df)} rows  (skipped / 제외: {skipped})"
        )
        return long_df

    def _load_labels_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        wide-format CSV (labels_v0.csv / labels_cmyk.csv) 처리.
        Processes wide-format CSV (labels_v0.csv / labels_cmyk.csv).

        columns: filename, C, M, Y, K
        """
        cols_map = {"Y": "Y", "M": "M", "C": "C", "K": "K"}
        records = []
        skipped = 0

        for _, row in df.iterrows():
            color = self._extract_color(str(row["filename"]))
            if color is None:
                skipped += 1
                continue
            col = cols_map.get(color)
            if col is None or col not in df.columns:
                skipped += 1
                continue
            records.append({
                "filename": row["filename"],
                "color": color,
                "level": int(row[col]),
            })

        long_df = pd.DataFrame(records)
        self.logger.info(
            f"Labels loaded (wide-format) / 라벨 로드: "
            f"{len(long_df)} rows  (skipped / 제외: {skipped})"
        )
        return long_df

    @torch.no_grad()
    def _run_single_channel(
        self,
        df_color: pd.DataFrame,
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        all_true, all_pred, all_conf, all_fnames = [], [], [], []

        for batch_imgs, batch_labels, batch_fnames in loader:
            batch_imgs = batch_imgs.to(self.device, non_blocking=True)
            logits = self.model(batch_imgs)
            probs = F.softmax(logits, dim=1)
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
            channels = ["Y", "M", "C", "K"]

        df_labels = self.load_labels()
        results: Dict[str, dict] = {}

        for color in channels:
            df_color = df_labels[df_labels["color"] == color].reset_index(drop=True)

            if len(df_color) == 0:
                self.logger.warning(f"[{color}] No samples / 샘플 없음 — skipping")
                continue

            y_true, y_pred, confs, fnames = self._run_single_channel(df_color)
            results[color] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "confidences": confs,
                "filenames": fnames,
            }

            acc = accuracy_score(y_true, y_pred)
            self.logger.info(
                f"[{color}] {len(y_true):4d} samples | Accuracy: {acc:.4f}"
            )

        self.logger.info(
            f"Inference complete / 추론 완료 — channels: {list(results.keys())}"
        )
        return results
