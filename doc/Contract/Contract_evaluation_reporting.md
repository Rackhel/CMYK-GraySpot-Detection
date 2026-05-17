---
type: contract
domain: evaluation_reporting
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Contract] Evaluation & Reporting — 평가 및 리포팅 경계 계약

> **목적**: `Evaluator`, `build_evaluation_summary()`, `determine_swing_feedback()`, `generate_baseline_report()`의 입출력 계약을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) (평가/리포팅 정의)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (Hard/Soft 판단 기준)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

평가 파이프라인의 입출력 타입, 반환 구조, 생성 파일 목록을 정의한다.

---

## 2. 모듈 트리

> ✅ **리팩토링 완료 (2026-05-12)**: `evaluator.py`(~950줄)를 4 Mixin + Orchestrator 패턴으로 분해하여 SRP·ISP 위반 해결.

```
evaluation/
├── evaluator_inference.py  — _EvalDataset, InferenceMixin  (추론 전담)
├── evaluator_metrics.py    — MetricsMixin                  (지표 계산 전담)
├── evaluator_export.py     — ExportMixin                   (CSV/JSON 저장 전담)
├── evaluator_charts.py     — ChartsMixin                   (차트 7종 + Phase 3 판단)
└── evaluator.py            — Evaluator                     (Orchestrator)
```

> 외부 API (`from evaluation.evaluator import Evaluator`) 및 §3–§12 계약은 변경 없음.

---

## 3. `Evaluator` 입력 계약

```python
evaluator = Evaluator(
    model       = model,            # nn.Module, model.eval() 상태
    labeled_dir = Path("data_set/labeled"),
    labels_csv  = Path("data_set/labels_v0.csv"),
    output_dir  = Path("outputs/reports"),
    device      = device,
    cfg         = cfg,              # swing_thresholds + confidence_thresholds 주입
)
```

| 입력 | 타입 | 제약 |
| --- | --- | --- |
| `model` | `nn.Module` | 반드시 `model.eval()` 상태 |
| 이미지 배치 | `Tensor (B, 3, 128, 128)` | BGR float32, ImageNet-normalized — 학습과 동일 |
| 레이블 | `Tensor (B,)` | int [0, 5] |

---

## 4. `Evaluator.run()` 출력 계약

```python
results = evaluator.run(channels=["Y", "M", "C", "K"])
```

| 출력 키 | 타입 | 형상 | 범위 |
| --- | --- | --- | --- |
| `"y_true"` | `np.ndarray` | `(N,)` | int [0, 5] |
| `"y_pred"` | `np.ndarray` | `(N,)` | int [0, 5] |
| `"confidences"` | `np.ndarray` | `(N,)` | float [0.0, 1.0] |

---

## 5. `build_evaluation_summary()` 계약

```python
from evaluation import build_evaluation_summary

summary = build_evaluation_summary(
    results  = {"Y": {"y_true": ..., "y_pred": ...}, ...},
    channels = ["Y", "M", "C", "K"],
    meta     = {"backbone": "efficientnet_b0"},
    cfg      = cfg,   # evaluation.swing_thresholds → summary.targets 주입
)
```

| 출력 | 타입 | 보장 |
| --- | --- | --- |
| `summary.overall` | `ChannelMetrics` | 전 채널 집계 지표 |
| `summary.by_channel[ch]` | `ChannelMetrics` | 채널별 지표 |
| `summary.targets["swing_*_retry"]` | `float` | cfg 주입값 (없으면 기본값) |

---

## 6. `ChannelMetrics` 구조

```python
@dataclass
class ChannelMetrics:
    accuracy : float       # 전체 정확도
    macro_f1 : float       # Macro-averaged F1
    mae      : float       # Mean Absolute Error (ordinal)
    n_samples: int
    per_class: List[PerClassMetric]  # level별 precision/recall/f1/support

    @property
    def acc_pass(self) -> bool: ...  # accuracy >= TARGET_PER_COLOR_ACC (0.85)
    def f1_pass (self) -> bool: ...  # macro_f1  >= TARGET_PER_CLASS_F1  (0.80)
    def mae_pass(self) -> bool: ...  # mae       <= TARGET_MAE            (0.50)
```

---

## 7. `determine_swing_feedback()` 계약

```python
from evaluation import determine_swing_feedback

feedback = determine_swing_feedback(summary, channels=None)
# channels: List[str] | None — None 이면 summary.by_channel.keys() 전체 사용
```

**반환 타입**: `dict`

```python
{
    "terminate": bool,        # True = 모든 목표 달성 → Swing 종료
    "decisions": List[str],   # 조치 필요 항목 목록 (비어있으면 계속 진행)
}
```

| `terminate` | `decisions` | 조건 | 다음 단계 |
| --- | --- | --- | --- |
| `True` | `[]` | 모든 채널 및 목표 달성 | Swing 종료 |
| `False` | 채움 | per-color accuracy < `swing_acc_retry` (0.80) | Phase 0 재시작 |
| `False` | 채움 | per-class F1 < `swing_f1_retry` (0.70) | Phase 1 레벨 경계 재검토 |
| `False` | 채움 | overall MAE > `swing_mae_retry` (0.80) | Phase 0 표현학습 재시도 |

> **임계값 출처**: `cfg["evaluation"]["swing_thresholds"]` 에서 읽음. 기본값은 위 괄호 안 값.

---

## 8. `MetricsMixin.compute()` 계약

```python
metrics = evaluator.compute(results, channels=None)
```

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `results` (입력) | `Dict[str, dict]` | `run()` 반환값 — 채널별 `y_true/y_pred/confidences/filenames` |
| `channels` (입력) | `Optional[List[str]]` | None 이면 `results.keys()` 전체 사용 |
| 반환 | `Dict[str, dict]` | `compute_all_channels()` 반환값 — `"overall"` 키 포함 |

---

## 9. `MetricsMixin.get_misclassified()` 계약

```python
df = evaluator.get_misclassified(results, channels=None)
```

| 반환 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `filename` | `str` | 파일명 |
| `color` | `str` | 채널 (`Y`/`M`/`C`/`K`) |
| `true_level` | `int` | 실제 레벨 [0-5] |
| `pred_level` | `int` | 예측 레벨 [0-5] |
| `confidence` | `float` | Max-softmax 신뢰도 [0.0-1.0] |
| `correct` | `bool` | 항상 `False` — 오분류 필터 결과 |

---

## 10. `ExportMixin.save_report()` 계약

```python
dashboard_path = evaluator.save_report(
    results,
    metrics,
    experiment_name="eval",
    channels=None,
    open_browser=False,
    checkpoint_path=None,
) -> Path
```

**생성 파일** (모두 `output_dir/` 아래):

| 파일 | 설명 |
| --- | --- |
| `confusion/cm_{channel}.html` | 채널별 정규화 혼동행렬 |
| `confusion/cm_overall.html` | 전체 혼동행렬 |
| `eval_dashboard.html` | Gauge + Bar 대시보드 (반환 경로) |
| `per_class_metrics.html` | 클래스별 F1 바 차트 |
| `mae_heatmap.html` | MAE 히트맵 |
| `misclassified_scatter.html` | 오분류 scatter |
| `confidence_distribution.html` | 신뢰도 분포 |
| `evaluation_results_{name}.csv` | 채널별 지표 CSV |
| `misclassified_{name}.csv` | 오분류 샘플 CSV |
| `metrics_summary_{name}.json` | JSON 요약 |

---

## 11. `summary_to_dict()` 계약

```python
from evaluation.metrics import summary_to_dict

d = summary_to_dict(summary)  # → dict (JSON-직렬화 가능)
```

```python
{
    "overall": {"accuracy": float, "macro_f1": float, "mae": float,
                "n_samples": int, "per_class": [{"level", "precision", "recall", "f1", "support"}, ...]},
    "by_channel": {"Y": {...}, "M": {...}, "C": {...}, "K": {...}},
    "meta": dict,
    "targets": dict,
}
```

---

## 12. `generate_baseline_report()` 계약

```python
from reporting.html_report import generate_baseline_report

path = generate_baseline_report(
    summary      : EvaluationSummary,
    results      : Dict[str, dict],
    output_path  : str | Path = "outputs/reports/baseline.html",
    channels     : List[str]  = ["Y", "M", "C", "K"],
    open_browser : bool       = False,
    logger       = None,
) -> Path
```

| 항목 | 설명 |
| --- | --- |
| 반환 | 생성된 HTML 파일의 절대 경로 |
| 산출물 | 단일 독립 HTML (Plotly CDN 내장) |
| 필수 입력 | `summary` (EvaluationSummary), `results` (run() 반환값) |

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | 평가/리포팅 정의 (What) |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 소비자 |
| [Contract_inference_boundary.md](Contract_inference_boundary.md) | 추론 경계 계약 |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-FF01 정의 |
