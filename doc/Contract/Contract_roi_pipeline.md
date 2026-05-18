---
type: contract
domain: roi_pipeline
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] ROI Pipeline — ROI 추출 및 라벨 정제 구현 계약

> **목적**: `ROIExtractor`, `LabelRefiner` 공개 API와 입출력 계약을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-18

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md)를 최종 판결자로 따른다.

---

## 1. ROIExtractor API 계약

```python
# src/data/roi_extractor.py

class ROIExtractor:
    def __init__(self, cfg: dict) -> None:
        # roi.mode, roi.fixed_coords, roi.auto_threshold 읽기
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

### 1.1 사전 조건

| 조건 | 설명 |
| --- | --- |
| 입력 이미지 | BGR uint8 (cv2.imread 기본) |
| config 키 필수 | `roi.mode`, `data.image_size` |
| channel | `"C"`, `"M"`, `"Y"`, `"K"` 중 하나 |

### 1.2 사후 조건

| 조건 | 설명 |
| --- | --- |
| 반환 패치 크기 | 정확히 `(cfg["data"]["image_size"], cfg["data"]["image_size"], 3)` |
| 반환 타입 | `List[np.ndarray]`, dtype=uint8 |
| 빈 반환 | ROI 감지 실패 시 `[]` 반환 (예외 아님) |

### 1.3 CMYK 분리 수식 (코드 레벨)

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

### 1.4 단채널 → 3채널 변환

```python
# 단채널 float [0,1] → BGR uint8 3채널 (모델 입력 호환)
ch_uint8 = (channel_arr * 255).astype(np.uint8)
patch_3ch = cv2.merge([ch_uint8, ch_uint8, ch_uint8])  # BGR 3채널
```

---

## 2. LabelRefiner API 계약

```python
# src/data/label_refiner.py

class LabelRefiner:
    def __init__(self, cfg: dict) -> None: ...

    def compute_priority_score(
        self,
        embeddings: np.ndarray,   # (N, D) float32 — GrayspotModel feature 출력
        labels: List[int],        # (N,) int — 0-based level
        paths: List[str],         # (N,) str — 이미지 경로
    ) -> pd.DataFrame:
        # Returns DataFrame: columns=[path, true_label, priority_score, cluster_label]
        # priority_score: float [0, 1], 높을수록 검토 우선순위 높음
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
        # Returns: priority_score 상위 top_ratio 비율 행만 반환
        ...

    def save_labels(
        self,
        original_csv: str | Path,
        corrections: Dict[str, int],  # {path: new_level (0-based)}
        output_path: str | Path,
    ) -> None:
        # corrections를 반영한 새 labels_vN.csv 저장
        ...
```

### 2.1 clustering_quality 판단 기준

| 지표 | 정제 완료 기준 |
| --- | --- |
| ARI | ≥ 0.6 |
| Silhouette Score | ≥ 0.4 |
| 판단 | 둘 중 하나 이상 만족 시 정제 완료 |

---

## 3. evaluate.py 스크립트 계약

```python
# src/scripts/evaluate.py
# CLI: python -m src.scripts.evaluate --channel Y [--checkpoint path/to/best.pt]

def main(argv: List[str] | None = None) -> None:
    # 1. load_config()
    # 2. 채널별 Evaluator 실행
    # 3. outputs/reports/ 에 JSON + HTML 저장
    # 4. 콘솔에 메트릭 출력
```

| 인자 | 필수 | 설명 |
| --- | --- | --- |
| `--channel` | ✅ | `Y`, `M`, `C`, `K`, `all` |
| `--checkpoint` | ❌ | 기본값: `outputs/models/best_{channel}.pt` |
| `--output-dir` | ❌ | 기본값: `outputs/reports/` |

---

## 4. swing_efficiency.py API 계약

```python
# src/evaluation/swing_efficiency.py

@dataclass
class SwingEfficiencyReport:
    cycle: int
    baseline_acc: float
    cycle_acc: float
    delta_acc: float                  # cycle_acc - baseline_acc
    n_labels_changed: int
    efficiency_ratio: float           # delta_acc / n_labels_changed (0이면 inf 처리)
    swing_decision: str               # "pass" | "retry_phase2" | "retry_phase0"

def compute_swing_efficiency(
    baseline_acc: float,
    cycle_acc: float,
    n_labels_changed: int,
    cycle: int,
    cfg: dict,
) -> SwingEfficiencyReport:
    # swing_thresholds는 cfg["evaluation"]["swing_thresholds"]에서 읽음
    ...

def should_early_stop(
    current_report: SwingEfficiencyReport,
    previous_report: SwingEfficiencyReport,
) -> bool:
    # current.efficiency_ratio < previous.efficiency_ratio * 0.5 → True
    ...
```

---

## 5. onnx_export.py API 계약

```python
# src/inference/onnx_export.py

def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    cfg: dict,
    opset_version: int = 17,
) -> Path:
    # 1. GrayspotModel 로드 (phase=2)
    # 2. torch.onnx.export(model, dummy_input, output_path, ...)
    # 3. onnx.checker.check_model() 검증
    # Returns: output_path (Path)
```

| 항목 | 값 |
| --- | --- |
| dummy_input shape | `(1, 3, 128, 128)` float32 |
| output shape | `(1, 6)` float32 logits |
| opset | 17 |
| 검증 | `onnx.checker.check_model()` 필수 |
| 오류 시 | `RuntimeError` 발생 (파일 미저장) |

---

## 6. 체크리스트

- [ ] `extract_patches()` 반환 패치가 항상 `(image_size, image_size, 3)` uint8인지 확인
- [ ] `split_cmyk()` 수식이 SSOT_ROI_Pipeline.md §2와 일치하는지 확인
- [ ] `compute_priority_score()` 반환 DataFrame 컬럼 검증
- [ ] `export_to_onnx()` 후 `onnx.checker` 통과 여부 확인
- [ ] `evaluate.py` 실행 시 리포트 파일 생성 확인

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | ROI 추출 및 라벨 정제 정의 (What) |
| [Contract_data_pipeline.md](Contract_data_pipeline.md) | 하위 데이터 파이프라인 계약 |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | Swing Efficiency 연동 |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | ONNX 산출물 스키마 |
