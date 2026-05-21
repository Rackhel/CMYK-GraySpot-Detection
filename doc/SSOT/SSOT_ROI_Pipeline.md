---
type: ssot
domain: roi_pipeline
status: Active
last_updated: 2026-05-21
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Data_Pipeline.md"
  - "SSOT_Evaluation_Reporting.md"
---

# [SSOT] ROI Pipeline — ROI 추출 및 라벨 정제 / ROI Extraction and Label Refinement

> **목적 / Purpose**: 원본 스캔 이미지에서 ROI 패치를 추출하는 파이프라인과 임베딩 기반 라벨 정제 흐름을 정의한다. / Define the pipeline for extracting ROI patches from raw scan images and the embedding-based label refinement flow.
> **역할 / Role**: What — ROI 추출 계약과 라벨 정제 기준을 선언한다. / Declare the ROI extraction contract and label refinement criteria.

---

## 1. ROI 추출 파이프라인 / ROI Extraction Pipeline

```
원본 스캔 이미지 (RGB, PNG/JPG)
  → cv2.imread()               # BGR uint8, HWC
  → CMYK 채널 분리             # §2 수식 적용 / Apply §2 formula
  → ROI 저장: data_set/roi/lvlX_..._CH.png
  → 채널별 독립 라벨링         # 시각 검사 후 roi_labels.csv 기록
  → ROIExtractor.extract_patches_from_roi()
      │  가로 정규화: 폭 ≥ 128 → 중앙 크롭, 폭 < 128 → reflect 패딩
      │  세로 슬라이딩: stride=128, 저분산 패치(std < min_std) 제거
      └─ 128×128×3 BGR uint8 패치 목록
  → 저장: data_set/labeled/{channel}/{level}/{roi_stem}_{idx:04d}.png
  → labels_master.csv 갱신
```

> 리사이즈(resize) 는 사용하지 않는다. 중앙 크롭 + 슬라이딩 윈도우 방식으로 패치를 추출한다.
> Resize is NOT used. Patches are extracted via center-crop width normalization + sliding window.

---

## 2. CMYK 채널 분리 수식 / CMYK Channel Splitting Formula

| 채널 / Channel | 수식 (정규화 입력 기준, [0,1]) / Formula (normalized input, [0,1]) |
| --- | --- |
| C (Cyan) | `1 - R` |
| M (Magenta) | `1 - G` |
| Y (Yellow) | `1 - B` |
| K (Key/Black) | `min(C, M, Y)` |

> 분리 후 각 채널은 독립 그레이스케일 이미지로 저장된다. 단, 모델 입력은 3채널(BGR)이므로 단채널을 3채널로 복제한다. / After splitting, each channel is saved as an independent grayscale image. However, since the model input requires 3 channels (BGR), the single channel is replicated to 3 channels.

> **⚠️ 채널별 독립 라벨링 / Per-Channel Independent Labeling**: CMYK 분리 후 각 채널 이미지를 **개별적으로** 육안 검사하여 레벨을 부여해야 한다. 스캔 파일명의 레벨(`lvlX_`)을 4개 채널에 동일하게 복사해서는 안 된다. / After CMYK splitting, each channel image must be **individually** inspected and labeled. Do NOT copy the scan filename level (`lvlX_`) uniformly to all 4 channels.

---

## 3. ROIExtractor 클래스 / ROIExtractor Class

| 항목 / Item | 값 / Value |
| --- | --- |
| 클래스 / Class | `src/data/roi_extractor.py::ROIExtractor` |
| 공개 메서드 / Public Methods | `split_cmyk(image)`, `extract_patches(path, channel, level)`, `extract_patches_from_roi(path)` |
| 반환 / Return | `List[np.ndarray]` — `(128, 128, 3)` BGR uint8 패치 목록 / BGR patch list |
| ROI 모드 / ROI Mode | `fixed` (config 좌표 / config coordinates) / `auto` (크롭 없이 전체 이미지) |
| config 키 / config Keys | `roi.mode`, `roi.fixed_coords`, `roi.min_std`, `data.image_size` |

#### 메서드 시그니처 / Method Signatures

```python
ROIExtractor(cfg: dict)
  # cfg["roi"]["mode"]              — "fixed" | "auto"
  # cfg["roi"]["fixed_coords"]      — [x0, y0, x1, y1]  (mode="fixed" 시 필수)
  # cfg["roi"].get("min_std", 5.0)  — 저분산 패치 제거 임계값 / Low-variance filter threshold
  # cfg["data"]["image_size"]       — 패치 크기 (int, 기본 128) / Patch size

.split_cmyk(image: np.ndarray) -> dict[str, np.ndarray]
  # BGR uint8 → CMYK float32 [0,1] 채널 dict

.extract_patches(image_path, channel: str, level: int) -> list[np.ndarray]
  # 전체 스캔 이미지: CMYK 분리 → 지정 채널 → 패치 추출
  # Full scan: CMYK split → specified channel → extract patches

.extract_patches_from_roi(image_path) -> list[np.ndarray]
  # 이미 분리된 ROI 이미지(lvlX_..._CH.png)에서 직접 패치 추출
  # Pre-split ROI image → extract patches directly (used by prepare_dataset.py)
```

### 3.1 ROI 모드 / ROI Mode

| 모드 / Mode | 설명 / Description | config |
| --- | --- | --- |
| `fixed` | `roi.fixed_coords: [x0, y0, x1, y1]`로 지정한 고정 영역 크롭 / Fixed region crop by `[x0, y0, x1, y1]` | `roi.mode: fixed` |
| `auto` | 크롭 없이 전체 이미지 사용 / Use full image without cropping | `roi.mode: auto` |

---

## 4. LabelRefiner 클래스 / LabelRefiner Class

| 항목 / Item | 값 / Value |
| --- | --- |
| 클래스 / Class | `src/data/label_refiner.py::LabelRefiner` |
| 주요 메서드 / Main Method | `compute_priority_score(embeddings, labels)` |
| 반환 / Return | `pd.DataFrame` — columns: [path, true_label, priority_score, cluster_label] |
| 정제 완료 기준 / Refinement Completion Criteria | ARI ≥ 0.6, Silhouette Score ≥ 0.4 |
| 검토 우선순위 / Review Priority | Priority Score 상위 20% 샘플 / Top 20% samples by Priority Score |

### 4.1 Priority Score 계산 / Priority Score Calculation

| 기준 / Criterion | 가중치 / Weight |
| --- | --- |
| 클러스터 경계 근접도 (클러스터 중심까지 거리) / Cluster boundary proximity (distance to cluster center) | 0.5 |
| 예측 신뢰도 역수 (낮은 confidence) / Inverse prediction confidence (low confidence) | 0.3 |
| 라벨 불일치 빈도 (이전 버전과 차이) / Label inconsistency frequency (difference from previous version) | 0.2 |

---

## 4.5 채널별 라벨링 워크플로우 / Per-Channel Labeling Workflow

ROI 파일의 `lvlX_` 는 스캔 단위 레벨이다. 채널별 독립 라벨링을 위해 `data_set/roi_labels.csv` 를 사용한다.

The `lvlX_` in ROI filenames is the scan-level label. Use `data_set/roi_labels.csv` for per-channel independent labels.

```csv
roi_filename,level
lvl3_Scanned Documents (113)_3_1_C,3
lvl3_Scanned Documents (113)_3_1_M,1
lvl3_Scanned Documents (113)_3_1_Y,0
lvl3_Scanned Documents (113)_3_1_K,0
```

- `roi_filename`: ROI 파일 스템 (확장자 제외) / ROI file stem (no extension)
- `level`: 시각 검사로 부여한 채널별 레벨 / Channel-specific level from visual inspection
- 파일이 없으면 `prepare_dataset.py` 가 파일명의 `lvlX_` 를 fallback 으로 사용
- If absent, `prepare_dataset.py` falls back to the `lvlX_` level in the filename

---

## 5. 라벨 버전 관리 / Label Version Management

| 버전 / Version | 파일명 / Filename | 위치 / Location | 형식 / Format | 생성 시점 / Creation Point | 상태 / Status |
| --- | --- | --- | --- | --- | --- |
| v0 | `labels_v0.csv` | `data_set/` | wide-format | S1 초기 라벨링 / S1 initial labeling | ⚠️ Legacy |
| v0b | `labels_cmyk.csv` | `data_set/` | wide-format | S2 2차 라벨링 / S2 second batch | ⚠️ Legacy |
| master | `labels_master.csv` | `data_set/` | long-format | S4 통합 정규화 / S4 unified normalization | ✅ **현행 Canonical** |

> `labels_master.csv` (내부 명칭 labels_v2) 는 long-format `(filepath, channel, level)` 정규 라벨 파일이다. 이 파일이 현행 학습 및 평가의 단일 라벨 소스다.
> `labels_master.csv` (internally labels_v2) is the canonical label file in long-format `(filepath, channel, level)`. It is the single label source for all training and evaluation.

> **Dataset status (2026-05-21)**: ✅ 재구성 완료. labels_master.csv 6,080행 (채널당 1,520장, L0:330 L1:330 L2:330 L3:265 L4:165 L5:100).
> ✅ Reconstruction complete. labels_master.csv has 6,080 rows (1,520 per channel, L0:330 L1:330 L2:330 L3:265 L4:165 L5:100).

### 5.1 labels_master.csv 스키마 / labels_master.csv Schema

| 컬럼 / Column | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `filepath` | str | 프로젝트 루트 기준 상대 경로 / Relative path from project root |
| `channel` | str | C / M / Y / K |
| `level` | int | 0–5 (Grayspot 결함 수준) / Defect level |

---

## 6. Phase 1 실행 조건 / Phase 1 Execution Conditions

| 조건 / Condition | 기준 / Criteria |
| --- | --- |
| Phase 0 완료 후 진입 / Entry after Phase 0 completion | `phase0_backbone_{ch}_{tag}.pt` 존재 / exists |
| 정제 완료 판단 / Refinement completion judgment | ARI ≥ 0.6 또는 / or Silhouette Score ≥ 0.4 |
| 최대 검토 샘플 수 / Maximum review sample count | Priority Score 상위 20% (전체의 20%만 / only 20% of total) |

---

## 7. 체크리스트 / Checklist

- [ ] ROI 좌표 변경 시 config.yaml `roi.fixed_coords` 업데이트 / Update config.yaml `roi.fixed_coords` when ROI coordinates change
- [ ] CMYK 분리 수식 변경 시 §2 동기화 / Sync §2 when CMYK splitting formula changes
- [ ] 라벨 버전 추가 시 §5 테이블 업데이트 / Update §5 table when a new label version is added
- [x] labels_master.csv (v2) 생성 완료 — §5 반영 / labels_master.csv (v2) created — reflected in §5
- [x] 채널별 독립 라벨링 정책 확정 (2026-05-21) — §2 경고 반영 / Per-channel independent labeling policy confirmed — reflected in §2 warning
- [x] ROIExtractor 구현 완료 (2026-05-21) — §3 반영 / ROIExtractor implemented — reflected in §3
- [x] roi_labels.csv 오버라이드 메커니즘 추가 (2026-05-21) — §4.5 신규 / roi_labels.csv override added — new §4.5
- [x] prepare_dataset.py 자동화 파이프라인 완성 (2026-05-21) / prepare_dataset.py full pipeline automation complete
- [x] 데이터셋 재구성 완료 (2026-05-21) — 6,080장 / Dataset reconstruction complete — 6,080 images
- [ ] Priority Score 가중치 변경 시 §4.1 동기화 / Sync §4.1 when Priority Score weights change

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | Hard SSOT 값 (image_size=128) / Hard SSOT values (image_size=128) |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | 하위 전처리 파이프라인 / Downstream preprocessing pipeline |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | ARI/Silhouette 지표 정의 / ARI/Silhouette metric definitions |
