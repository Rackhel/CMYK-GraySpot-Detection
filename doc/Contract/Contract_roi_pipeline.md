---
type: contract
domain: roi_pipeline
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] ROI Pipeline — ROI 추출 및 라벨 정제 구현 계약 / ROI Extraction and Label Refinement Implementation Contract

> **목적 / Purpose**: `ROIExtractor`, `LabelRefiner` 공개 API와 입출력 계약을 정의한다. / Defines the public API and I/O contracts for ROIExtractor and LabelRefiner.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-18

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md)를 최종 판결자로 따른다.
> / This document does not redefine semantic definitions in SSOT documents. For semantic interpretation, follow SSOT_ROI_Pipeline.md as the final authority.

---

## 1. ROIExtractor API 계약 / ROIExtractor API Contract

```python
# src/data/roi_extractor.py

class ROIExtractor:
    def __init__(self, cfg: dict) -> None:
        # roi.mode, roi.fixed_coords, roi.auto_threshold 읽기 / read config keys
        ...

    def extract_patches(
        self,
        image_path: str | Path,
        channel: str,          # "C", "M", "Y", "K"
        level: int,            # 1–6 (directory label, 1-based)
        save_dir: str | Path | None = None,
    ) -> List[np.ndarray]:
        # Returns: List of (128, 128, 3) BGR uint8 patches
        ...

    def split_cmyk(self, bgr_image: np.ndarray) -> Dict[str, np.ndarray]:
        # Returns: {"C": arr, "M": arr, "Y": arr, "K": arr}
        # Each arr: (H, W) float32 [0,1]
        ...
```

### 1.1 사전 조건 / Preconditions

| 조건 / Condition | 설명 / Description |
| --- | --- |
| 입력 이미지 / Input image | BGR uint8 (cv2.imread 기본 / default) |
| config 키 필수 / Required config keys | `roi.mode`, `data.image_size` |
| channel | `"C"`, `"M"`, `"Y"`, `"K"` 중 하나 / one of |

### 1.2 사후 조건 / Postconditions

| 조건 / Condition | 설명 / Description |
| --- | --- |
| 반환 패치 크기 / Return patch size | 정확히 / Exactly `(cfg["data"]["image_size"], cfg["data"]["image_size"], 3)` |
| 반환 타입 / Return type | `List[np.ndarray]`, dtype=uint8 |
| 빈 반환 / Empty return | ROI 감지 실패 시 `[]` 반환 (예외 아님) / Returns `[]` on ROI detection failure (not an exception) |

### 1.3 CMYK 분리 수식 / CMYK Splitting Formula (코드 레벨 / Code Level)

```python
def split_cmyk(self, bgr_image: np.ndarray) -> Dict[str, np.ndarray]:
    img_float = bgr_image.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    c_ch = 1.0 - r
    m_ch = 1.0 - g
    y_ch = 1.0 - b
    k_ch = np.minimum(np.minimum(c_ch, m_ch), y_ch)
    return {"C": c_ch, "M": m_ch, "Y": y_ch, "K": k_ch}
```

### 1.4 단채널 → 3채널 변환 / Single-Channel to 3-Channel Conversion

```python
# 단채널 float [0,1] → BGR uint8 3채널 (모델 입력 호환)
# / single-channel float → 3-channel BGR uint8 (model input compatible)
ch_uint8 = (channel_arr * 255).astype(np.uint8)
patch_3ch = cv2.merge([ch_uint8, ch_uint8, ch_uint8])  # BGR 3채널 / BGR 3-channel
```

---

## 2. LabelRefiner API 계약 / LabelRefiner API Contract

```python
# src/data/label_refiner.py

class LabelRefiner:
    def __init__(self, cfg: dict) -> None: ...

    def compute_priority_score(
        self,
        embeddings: np.ndarray,   # (N, D) float32 — GrayspotModel feature 출력 / output
        labels: List[int],        # (N,) int — 0-based level
        paths: List[str],         # (N,) str — 이미지 경로 / image paths
    ) -> pd.DataFrame:
        # Returns DataFrame: columns=[path, true_label, priority_score, cluster_label]
        # priority_score: float [0, 1], 높을수록 검토 우선순위 높음 / higher = higher review priority
        ...

    def compute_clustering_quality(
        self,
        embeddings: np.ndarray,
        labels: List[int],
    ) -> Dict[str, float]:
        # Returns: {"ari": float, "silhouette": float}
        ...

    def get_review_queue(
        self,
        priority_df: pd.DataFrame,
        top_ratio: float = 0.2,
    ) -> pd.DataFrame:
        # Returns: priority_score 상위 top_ratio 비율 행만 반환 / returns top top_ratio fraction rows
        ...

    def save_labels(
        self,
        original_csv: str | Path,
        corrections: Dict[str, int],  # {path: new_level (0-based)}
        output_path: str | Path,
    ) -> None:
        # corrections를 반영한 새 labels_vN.csv 저장 / save new labels_vN.csv with corrections applied
        ...
```

### 2.1 clustering_quality 판단 기준 / clustering_quality Decision Criteria

| 지표 / Metric | 정제 완료 기준 / Refinement Completion Criterion |
| --- | --- |
| ARI | ≥ 0.6 |
| Silhouette Score | ≥ 0.4 |
| 판단 / Decision | 둘 중 하나 이상 만족 시 정제 완료 / Refinement complete when at least one is satisfied |

---

## 3. evaluate.py 스크립트 계약 / evaluate.py Script Contract

```python
# src/scripts/evaluate.py
# CLI: python -m src.scripts.evaluate --channel Y [--checkpoint path/to/best.pt]

def main(argv: List[str] | None = None) -> None:
    # 1. load_config()
    # 2. 채널별 Evaluator 실행 / Run Evaluator per channel
    # 3. outputs/reports/ 에 JSON + HTML 저장 / Save JSON + HTML to outputs/reports/
    # 4. 콘솔에 메트릭 출력 / Print metrics to console
```

| 인자 / Argument | 필수 / Required | 설명 / Description |
| --- | --- | --- |
| `--channel` | ✅ | `Y`, `M`, `C`, `K`, `all` |
| `--checkpoint` | ❌ | 기본값 / Default: `outputs/models/best_{channel}.pt` |
| `--output-dir` | ❌ | 기본값 / Default: `outputs/reports/` |

---

## 4. swing_efficiency.py API 계약 / swing_efficiency.py API Contract

```python
# src/evaluation/swing_efficiency.py

@dataclass
class SwingEfficiencyReport:
    cycle: int
    baseline_acc: float
    cycle_acc: float
    delta_acc: float                  # cycle_acc - baseline_acc
    n_labels_changed: int
    efficiency_ratio: float           # delta_acc / n_labels_changed (0/0 → 0.0, not exception)
    swing_decision: str               # "pass" | "retry_phase2" | "retry_phase0"

def compute_swing_efficiency(
    baseline_acc: float,
    cycle_acc: float,
    n_labels_changed: int,
    cycle: int,
    cfg: dict,
) -> SwingEfficiencyReport:
    # swing_thresholds는 cfg["evaluation"]["swing_thresholds"]에서 읽음 / read from cfg
    ...

def should_early_stop(
    current_report: SwingEfficiencyReport,
    previous_report: SwingEfficiencyReport,
) -> bool:
    # current.efficiency_ratio < previous.efficiency_ratio * 0.5 → True
    ...
```

---

## 5. onnx_export.py API 계약 / onnx_export.py API Contract

```python
# src/inference/onnx_export.py

def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    cfg: dict,
    opset_version: int = 17,
) -> Path:
    # 1. GrayspotModel 로드 / load (phase=2)
    # 2. torch.onnx.export(model, dummy_input, output_path, ...)
    # 3. onnx.checker.check_model() 검증 / validation
    # Returns: output_path (Path)
```

| 항목 / Item | 값 / Value |
| --- | --- |
| dummy_input shape | `(1, 3, 128, 128)` float32 |
| output shape | `(1, 6)` float32 logits |
| opset | 17 |
| 검증 / Validation | `onnx.checker.check_model()` 필수 / required |
| 오류 시 / On error | `RuntimeError` 발생 / raised (파일 미저장 / file not saved) |

---

## 6. 체크리스트 / Checklist

- [ ] `extract_patches()` 반환 패치가 항상 `(image_size, image_size, 3)` uint8인지 확인 / Verify extract_patches() always returns (image_size, image_size, 3) uint8
- [ ] `split_cmyk()` 수식이 SSOT_ROI_Pipeline.md §2와 일치하는지 확인 / Verify split_cmyk() formula matches SSOT_ROI_Pipeline.md §2
- [ ] `compute_priority_score()` 반환 DataFrame 컬럼 검증 / Validate compute_priority_score() return DataFrame columns
- [ ] `export_to_onnx()` 후 `onnx.checker` 통과 여부 확인 / Verify onnx.checker passes after export_to_onnx()
- [ ] `evaluate.py` 실행 시 리포트 파일 생성 확인 / Confirm report files are created when evaluate.py runs

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | ROI 추출 및 라벨 정제 정의 (What) / ROI extraction and label refinement definition |
| [Contract_data_pipeline.md](Contract_data_pipeline.md) | 하위 데이터 파이프라인 계약 / Downstream data pipeline contract |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | Swing Efficiency 연동 / Swing Efficiency integration |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | ONNX 산출물 스키마 / ONNX artifact schema |
