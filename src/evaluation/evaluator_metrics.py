"""
evaluation/evaluator_metrics.py

책임 / Responsibility: 지표 계산 위임 + 오분류 샘플 추출
Responsibility: Metric computation delegation + misclassified sample extraction

SRP 준수: 이 모듈은 "지표 계산과 오분류 분석"만 담당한다.
SRP compliant: this module handles only "metric computation and misclassification analysis".
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from evaluation.metrics import compute_all_channels, print_summary


class MetricsMixin:
    """
    지표 계산과 오분류 샘플 추출을 담당하는 Mixin.
    Mixin responsible for metric computation and misclassified sample extraction.

    사용하는 self 속성 / Consumed self attributes (provided by Evaluator.__init__):
        self.num_levels, self.logger
    """

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
