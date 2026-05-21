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
  → cv2.imread()          # BGR uint8, HWC
  → CMYK 채널 분리        # §2 수식 적용 / Apply §2 formula
  → ROI 좌표 적용         # 고정 좌표 또는 자동 감지 / Fixed coordinates or auto-detection
  → cv2.resize(128,128)   # 패치 크기 (Hard SSOT) / Patch size (Hard SSOT)
  → 저장: data_set/labeled/{channel}/{level}/*.png
```

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
| 주요 메서드 / Main Method | `extract_patches(image_path, channel, level, cfg)` |
| 반환 / Return | `List[np.ndarray]` — (128, 128, 3) BGR 패치 목록 / BGR patch list |
| ROI 모드 / ROI Mode | `fixed` (config 좌표 / config coordinates) / `auto` (엣지 감지 / edge detection) |
| config 키 / config Keys | `roi.mode`, `roi.fixed_coords`, `roi.auto_threshold` |

### 3.1 ROI 모드 / ROI Mode

| 모드 / Mode | 설명 / Description | config |
| --- | --- | --- |
| `fixed` | `roi.fixed_coords: [x, y, w, h]`로 지정한 고정 영역 / Fixed region specified by `roi.fixed_coords: [x, y, w, h]` | `roi.mode: fixed` |
| `auto` | Canny 엣지 + 컨투어로 자동 감지 / Auto-detection via Canny edge + contour | `roi.mode: auto` |

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

## 5. 라벨 버전 관리 / Label Version Management

| 버전 / Version | 파일명 / Filename | 위치 / Location | 형식 / Format | 생성 시점 / Creation Point | 상태 / Status |
| --- | --- | --- | --- | --- | --- |
| v0 | `labels_v0.csv` | `data_set/` | wide-format | S1 초기 라벨링 / S1 initial labeling | ⚠️ Legacy |
| v0b | `labels_cmyk.csv` | `data_set/` | wide-format | S2 2차 라벨링 / S2 second batch | ⚠️ Legacy |
| master | `labels_master.csv` | `data_set/` | long-format | S4 통합 정규화 / S4 unified normalization | ✅ **현행 Canonical** |

> `labels_master.csv` (내부 명칭 labels_v2) 는 long-format `(filepath, channel, level)` 정규 라벨 파일이다. 이 파일이 현행 학습 및 평가의 단일 라벨 소스다.
> `labels_master.csv` (internally labels_v2) is the canonical label file in long-format `(filepath, channel, level)`. It is the single label source for all training and evaluation.

> **Dataset status (2026-05-21)**: OLD labeled 데이터 삭제 후 초기화됨 (0 rows). 채널별 독립 재라벨링 진행 중.
> Old labeled data cleared; 0 rows. Per-channel independent re-labeling in progress.

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
- [ ] 신규 ROI (288장/채널) 채널별 재라벨링 완료 후 §5 상태 업데이트 / Update §5 status after per-channel re-labeling of new ROI images
- [ ] Priority Score 가중치 변경 시 §4.1 동기화 / Sync §4.1 when Priority Score weights change

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | Hard SSOT 값 (image_size=128) / Hard SSOT values (image_size=128) |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | 하위 전처리 파이프라인 / Downstream preprocessing pipeline |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | ARI/Silhouette 지표 정의 / ARI/Silhouette metric definitions |
