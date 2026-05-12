"""
evaluation/evaluator_export.py

책임 / Responsibility: CSV / JSON 결과 저장
Responsibility: Save results as CSV and JSON

SRP 준수: 이 모듈은 "파일 직렬화 및 저장"만 담당한다.
SRP compliant: this module handles only "file serialization and export".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluation.metrics import (
    TARGET_MAE,
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    check_targets,
)


class ExportMixin:
    """
    CSV / JSON 저장을 담당하는 Mixin.
    Mixin responsible for CSV and JSON export.

    사용하는 self 속성 / Consumed self attributes (provided by Evaluator.__init__):
        self.output_dir, self.image_size, self.num_levels, self.logger
    """

    def save_csv(
        self,
        results: Dict[str, dict],
        experiment_name: str = "eval",
        channels: List[str] = None,
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
            yt = results[color]["y_true"]
            yp = results[color]["y_pred"]
            confs = results[color]["confidences"]
            fnames = results[color]["filenames"]
            for i in range(len(yt)):
                rows.append(
                    {
                        "filename": fnames[i],
                        "color": color,
                        "true_level": int(yt[i]),
                        "pred_level": int(yp[i]),
                        "confidence": round(float(confs[i]), 4),
                        "correct": bool(yt[i] == yp[i]),
                        "error_gap": int(abs(yt[i] - yp[i])),
                    }
                )

        path = self.output_dir / f"evaluation_results_{experiment_name}.csv"
        # UTF-8 BOM: Windows Excel 한글 깨짐 방지 / Prevents Korean garbling in Windows Excel
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        self.logger.info(f"CSV saved / 저장: {path}  ({len(rows)} rows)")
        return path

    def save_json(
        self,
        metrics: Dict[str, dict],
        experiment_name: str = "eval",
        channels: List[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Path:
        """
        집계된 지표를 JSON 으로 저장한다. (PRD 8.2.2)
        Saves aggregated metrics as JSON. (PRD 8.2.2)
        """
        if channels is None:
            channels = [k for k in metrics if k != "overall"]

        targets = check_targets(metrics, channels)

        export = {
            "meta": {
                "experiment": experiment_name,
                "checkpoint": checkpoint_path,
                "image_size": self.image_size,
                "num_levels": self.num_levels,
            },
            "targets": {
                "overall_accuracy": TARGET_OVERALL_ACC,
                "per_color_accuracy": TARGET_PER_COLOR_ACC,
                "per_class_f1": TARGET_PER_CLASS_F1,
                "mae": TARGET_MAE,
            },
            "global": {
                "accuracy": round(metrics["overall"]["accuracy"], 4),
                "macro_f1": round(metrics["overall"]["macro_f1"], 4),
                "mae": round(metrics["overall"]["mae"], 4),
                "acc_pass": targets["overall"]["acc_pass"],
                "f1_pass": targets["overall"]["f1_pass"],
                "mae_pass": targets["overall"]["mae_pass"],
                "all_pass": targets["overall"]["all_pass"],
            },
            "by_color": {
                color: {
                    "accuracy": round(metrics[color]["accuracy"], 4),
                    "macro_f1": round(metrics[color]["macro_f1"], 4),
                    "mae": round(metrics[color]["mae"], 4),
                    "acc_pass": targets[color]["acc_pass"],
                    "all_pass": targets[color]["all_pass"],
                }
                for color in channels
                if color in metrics
            },
            "per_class_overall": [
                {
                    "level": pc["level"],
                    "precision": round(pc["precision"], 4),
                    "recall": round(pc["recall"], 4),
                    "f1": round(pc["f1"], 4),
                    "f1_pass": bool(pc["f1"] >= TARGET_PER_CLASS_F1),
                }
                for pc in metrics["overall"]["per_class"]
            ],
        }

        path = self.output_dir / f"metrics_summary_{experiment_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON saved / 저장: {path}")
        return path
