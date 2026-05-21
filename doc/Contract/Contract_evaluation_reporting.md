---
type: contract
domain: evaluation_reporting
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Evaluation & Reporting — 평가 및 리포팅 경계 계약 / Evaluation & Reporting Boundary Contract

> **목적 / Purpose**: `Evaluator`, `build_evaluation_summary()`, `determine_swing_feedback()`, `generate_baseline_report()`의 입출력 계약을 정의한다. / Defines I/O contracts for `Evaluator`, `build_evaluation_summary()`, `determine_swing_feedback()`, and `generate_baseline_report()`.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) (평가/리포팅 정의 / Evaluation/Reporting definition)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (Hard/Soft 판단 기준 / Hard/Soft decision criteria)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

평가 파이프라인의 입출력 타입, 반환 구조, 생성 파일 목록을 정의한다.

Defines input/output types, return structures, and generated file lists for the evaluation pipeline.

---

## 2. 모듈 트리 / Module Tree

> ✅ **리팩토링 완료 (2026-05-12) / Refactoring Complete**: `evaluator.py`(~950줄)를 4 Mixin + Orchestrator 패턴으로 분해하여 SRP·ISP 위반 해결. / Decomposed `evaluator.py` (~950 lines) into 4 Mixin + Orchestrator pattern to resolve SRP·ISP violations.

```
data/
└── dataset.py              — _EvalDataset                  (평가 전용 Dataset — 데이터 계층 / Eval-only Dataset — data layer)

evaluation/
├── evaluator_inference.py  — InferenceMixin                (추론 전담; _EvalDataset 은 data/dataset.py 에서 import / inference only; _EvalDataset imported from data/dataset.py)
├── evaluator_metrics.py    — MetricsMixin                  (지표 계산 전담 / metrics computation only)
├── evaluator_export.py     — ExportMixin                   (CSV/JSON 저장 전담 / CSV/JSON export only)
├── evaluator_charts.py     — ChartsMixin                   (차트 7종 + Phase 3 판단 / 7 charts + Phase 3 decision)
└── evaluator.py            — Evaluator                     (Orchestrator)
```

> 외부 API (`from evaluation.evaluator import Evaluator`) 및 §3–§12 계약은 변경 없음.
> / External API (`from evaluation.evaluator import Evaluator`) and §3–§12 contracts remain unchanged.

---

## 3. `Evaluator` 입력 계약 / Evaluator Input Contract

```python
evaluator = Evaluator(
    model       = model,            # nn.Module, model.eval() 상태 / model.eval() state
    labeled_dir = Path("data_set/labeled"),
    labels_csv  = Path("data_set/labels_master.csv"),   # ← Canonical (long-format)
    output_dir  = Path("outputs/reports"),
    device      = device,
    cfg         = cfg,              # swing_thresholds + confidence_thresholds 주입 / injected
)
```

> **CSV 형식 자동 감지 / CSV format auto-detection**: `labels_csv` 에는 `labels_master.csv` (long-format: `filepath`, `channel`, `level`) 또는 레거시 wide-format (`filename`, `C`, `M`, `Y`, `K`) 을 모두 전달할 수 있다. `InferenceMixin.load_labels()` 가 형식을 자동 감지한다.
> Both `labels_master.csv` (long-format) and legacy wide-format CSV files are accepted; `InferenceMixin.load_labels()` auto-detects the format.

| 입력 / Input | 타입 / Type | 제약 / Constraint |
| --- | --- | --- |
| `model` | `nn.Module` | 반드시 `model.eval()` 상태 / Must be in `model.eval()` state |
| 이미지 배치 / Image batch | `Tensor (B, 3, 128, 128)` | BGR float32, ImageNet-normalized — 학습과 동일 / Same as training |
| 레이블 / Labels | `Tensor (B,)` | int [0, 5] |

---

## 4. `Evaluator.run()` 출력 계약 / Evaluator.run() Output Contract

```python
results = evaluator.run(channels=["Y", "M", "C", "K"])
```

| 출력 키 / Output Key | 타입 / Type | 형상 / Shape | 범위 / Range |
| --- | --- | --- | --- |
| `"y_true"` | `np.ndarray` | `(N,)` | int [0, 5] |
| `"y_pred"` | `np.ndarray` | `(N,)` | int [0, 5] |
| `"confidences"` | `np.ndarray` | `(N,)` | float [0.0, 1.0] |

---

## 5. `build_evaluation_summary()` 계약 / build_evaluation_summary() Contract

```python
from evaluation import build_evaluation_summary

summary = build_evaluation_summary(
    results  = {"Y": {"y_true": ..., "y_pred": ...}, ...},
    channels = ["Y", "M", "C", "K"],
    meta     = {"backbone": "efficientnet_b0"},
    cfg      = cfg,   # evaluation.swing_thresholds → summary.targets 주입 / injected
)
```

| 출력 / Output | 타입 / Type | 보장 / Guarantee |
| --- | --- | --- |
| `summary.overall` | `ChannelMetrics` | 전 채널 집계 지표 / Aggregated metrics across all channels |
| `summary.by_channel[ch]` | `ChannelMetrics` | 채널별 지표 / Per-channel metrics |
| `summary.targets["swing_*_retry"]` | `float` | cfg 주입값 (없으면 기본값) / cfg-injected value (defaults if absent) |

---

## 6. `ChannelMetrics` 구조 / ChannelMetrics Structure

```python
@dataclass
class ChannelMetrics:
    accuracy : float       # 전체 정확도 / Overall accuracy
    macro_f1 : float       # Macro-averaged F1
    mae      : float       # Mean Absolute Error (ordinal)
    n_samples: int
    per_class: List[PerClassMetric]  # level별 precision/recall/f1/support / per-level metrics

    @property
    def acc_pass(self) -> bool: ...  # accuracy >= TARGET_PER_COLOR_ACC (0.85)
    def f1_pass (self) -> bool: ...  # macro_f1  >= TARGET_PER_CLASS_F1  (0.80)
    def mae_pass(self) -> bool: ...  # mae       <= TARGET_MAE            (0.50)
```

---

## 7. `determine_swing_feedback()` 계약 / determine_swing_feedback() Contract

```python
from evaluation import determine_swing_feedback

feedback = determine_swing_feedback(summary, channels=None)
# channels: List[str] | None — None 이면 summary.by_channel.keys() 전체 사용
# / None uses all keys from summary.by_channel
```

**반환 타입 / Return Type**: `dict`

```python
{
    "terminate": bool,        # True = 모든 목표 달성 → Swing 종료 / True = all targets met → end Swing
    "decisions": List[str],   # 조치 필요 항목 목록 (비어있으면 계속 진행) / action items (empty = continue)
}
```

| `terminate` | `decisions` | 조건 / Condition | 다음 단계 / Next Step |
| --- | --- | --- | --- |
| `True` | `[]` | 모든 채널 및 목표 달성 / All channels and targets met | Swing 종료 / End Swing |
| `False` | 채움 / Filled | per-color accuracy < `swing_acc_retry` (0.80) | Phase 0 재시작 / Restart Phase 0 |
| `False` | 채움 / Filled | per-class F1 < `swing_f1_retry` (0.70) | Phase 1 레벨 경계 재검토 / Review Phase 1 level boundaries |
| `False` | 채움 / Filled | overall MAE > `swing_mae_retry` (0.80) | Phase 0 표현학습 재시도 / Retry Phase 0 representation learning |

> **임계값 출처 / Threshold Source**: `cfg["evaluation"]["swing_thresholds"]` 에서 읽음. 기본값은 위 괄호 안 값. / Read from cfg. Default values shown in parentheses above.

---

## 8. `MetricsMixin.compute()` 계약 / MetricsMixin.compute() Contract

```python
metrics = evaluator.compute(results, channels=None)
```

| 항목 / Item | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `results` (입력 / Input) | `Dict[str, dict]` | `run()` 반환값 — 채널별 `y_true/y_pred/confidences/filenames` / return value — per-channel data |
| `channels` (입력 / Input) | `Optional[List[str]]` | None 이면 `results.keys()` 전체 사용 / None uses all keys from results |
| 반환 / Return | `Dict[str, dict]` | `compute_all_channels()` 반환값 — `"overall"` 키 포함 / includes `"overall"` key |

---

## 9. `MetricsMixin.get_misclassified()` 계약 / MetricsMixin.get_misclassified() Contract

```python
df = evaluator.get_misclassified(results, channels=None)
```

| 반환 컬럼 / Return Column | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `filename` | `str` | 파일명 / Filename |
| `color` | `str` | 채널 / Channel (`Y`/`M`/`C`/`K`) |
| `true_level` | `int` | 실제 레벨 / True level [0-5] |
| `pred_level` | `int` | 예측 레벨 / Predicted level [0-5] |
| `confidence` | `float` | Max-softmax 신뢰도 / confidence [0.0-1.0] |
| `correct` | `bool` | 항상 `False` — 오분류 필터 결과 / Always `False` — misclassification filter result |

---

## 10. `ExportMixin.save_report()` 계약 / ExportMixin.save_report() Contract

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

**생성 파일 / Generated Files** (모두 `output_dir/` 아래 / all under `output_dir/`):

| 파일 / File | 설명 / Description |
| --- | --- |
| `confusion/cm_{channel}.html` | 채널별 정규화 혼동행렬 / Per-channel normalized confusion matrix |
| `confusion/cm_overall.html` | 전체 혼동행렬 / Overall confusion matrix |
| `eval_dashboard.html` | Gauge + Bar 대시보드 (반환 경로) / Gauge + Bar dashboard (returned path) |
| `per_class_metrics.html` | 클래스별 F1 바 차트 / Per-class F1 bar chart |
| `mae_heatmap.html` | MAE 히트맵 / MAE heatmap |
| `misclassified_scatter.html` | 오분류 scatter / Misclassification scatter plot |
| `confidence_distribution.html` | 신뢰도 분포 / Confidence distribution |
| `evaluation_results_{name}.csv` | 채널별 지표 CSV / Per-channel metrics CSV |
| `misclassified_{name}.csv` | 오분류 샘플 CSV / Misclassified samples CSV |
| `metrics_summary_{name}.json` | JSON 요약 / JSON summary |

---

## 11. `summary_to_dict()` 계약 / summary_to_dict() Contract

```python
from evaluation.metrics import summary_to_dict

d = summary_to_dict(summary)  # → dict (JSON-직렬화 가능 / JSON-serializable)
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

## 12. `generate_baseline_report()` 계약 / generate_baseline_report() Contract

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

| 항목 / Item | 설명 / Description |
| --- | --- |
| 반환 / Return | 생성된 HTML 파일의 절대 경로 / Absolute path of the generated HTML file |
| 산출물 / Output | 단일 독립 HTML (Plotly CDN 내장) / Single standalone HTML (Plotly CDN embedded) |
| 필수 입력 / Required Input | `summary` (EvaluationSummary), `results` (run() 반환값 / return value) |

---

## 13. Swing Efficiency 계약 / Swing Efficiency Contract

### compute_swing_efficiency()

```python
from evaluation.swing_efficiency import compute_swing_efficiency, SwingEfficiencyReport

report: SwingEfficiencyReport = compute_swing_efficiency(
    baseline_acc=0.72,
    cycle_acc=0.81,
    n_labels_changed=45,
    cycle=1,
    cfg=cfg,
)
# report.swing_decision: "pass" | "retry_phase2" | "retry_phase0"
# report.efficiency_ratio: delta_acc / n_labels_changed
```

### should_early_stop()

```python
from evaluation.swing_efficiency import should_early_stop

stop = should_early_stop(current_report, previous_report)
# True → Cycle 2 조기 종료, Cycle 1 결과 채택
# / True → Early stop Cycle 2, adopt Cycle 1 results
```

---

## 체크리스트 / Checklist

- [ ] `Evaluator.run()` 반환 dict에 `y_true`, `y_pred`, `confidences` 키 포함 확인 / Verify `Evaluator.run()` return dict includes `y_true`, `y_pred`, `confidences` keys
- [ ] `build_evaluation_summary()` 호출 시 `cfg["evaluation"]["swing_thresholds"]` 주입 확인 / Verify `cfg["evaluation"]["swing_thresholds"]` is injected when calling `build_evaluation_summary()`
- [ ] `determine_swing_feedback()` 반환 `terminate` 플래그 기반 Swing 종료 로직 확인 / Verify Swing termination logic based on `terminate` flag in `determine_swing_feedback()` return
- [ ] `save_report()` 후 `outputs/reports/` 하위 파일 생성 확인 / Verify files are created under `outputs/reports/` after `save_report()`
- [ ] `compute_swing_efficiency()` 반환 `swing_decision` 값이 `"pass"` | `"retry_phase2"` | `"retry_phase0"` 중 하나인지 확인 / Verify `swing_decision` value is one of `"pass"` | `"retry_phase2"` | `"retry_phase0"`
- [ ] `should_early_stop()` 조기 종료 조건 (`efficiency_ratio < previous * 0.5`) 단위 테스트 확인 / Verify unit test for early stop condition (`efficiency_ratio < previous * 0.5`)

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | 평가/리포팅 정의 (What) / Evaluation/reporting definition |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 소비자 / Checkpoint consumer |
| [Contract_inference_boundary.md](Contract_inference_boundary.md) | 추론 경계 계약 / Inference boundary contract |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-FF01 정의 / SSOT-FF01 definition |
