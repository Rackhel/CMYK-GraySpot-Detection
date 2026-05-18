---
type: ssot
domain: roi_pipeline
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Data_Pipeline.md"
  - "SSOT_Evaluation_Reporting.md"
---

# [SSOT] ROI Pipeline — ROI 추출 및 라벨 정제

> **목적**: 원본 스캔 이미지에서 ROI 패치를 추출하는 파이프라인과 임베딩 기반 라벨 정제 흐름을 정의한다.
> **역할**: What — ROI 추출 계약과 라벨 정제 기준을 선언한다.

---

## 1. ROI 추출 파이프라인

```
원본 스캔 이미지 (RGB, PNG/JPG)
  → cv2.imread()          # BGR uint8, HWC
  → CMYK 채널 분리        # §2 수식 적용
  → ROI 좌표 적용         # 고정 좌표 또는 자동 감지
  → cv2.resize(128,128)   # 패치 크기 (Hard SSOT)
  → 저장: data_set/labeled/{channel}/{level}/*.png
```

---

## 2. CMYK 채널 분리 수식

| 채널 | 수식 (정규화 입력 기준, [0,1]) |
| --- | --- |
| C (Cyan) | `1 - R` |
| M (Magenta) | `1 - G` |
| Y (Yellow) | `1 - B` |
| K (Key/Black) | `min(C, M, Y)` |

> 분리 후 각 채널은 독립 그레이스케일 이미지로 저장된다. 단, 모델 입력은 3채널(BGR)이므로 단채널을 3채널로 복제한다.

---

## 3. ROIExtractor 클래스

| 항목 | 값 |
| --- | --- |
| 클래스 | `src/data/roi_extractor.py::ROIExtractor` |
| 주요 메서드 | `extract_patches(image_path, channel, level, cfg)` |
| 반환 | `List[np.ndarray]` — (128, 128, 3) BGR 패치 목록 |
| ROI 모드 | `fixed` (config 좌표) / `auto` (엣지 감지) |
| config 키 | `roi.mode`, `roi.fixed_coords`, `roi.auto_threshold` |

### 3.1 ROI 모드

| 모드 | 설명 | config |
| --- | --- | --- |
| `fixed` | `roi.fixed_coords: [x, y, w, h]`로 지정한 고정 영역 | `roi.mode: fixed` |
| `auto` | Canny 엣지 + 컨투어로 자동 감지 | `roi.mode: auto` |

---

## 4. LabelRefiner 클래스

| 항목 | 값 |
| --- | --- |
| 클래스 | `src/data/label_refiner.py::LabelRefiner` |
| 주요 메서드 | `compute_priority_score(embeddings, labels)` |
| 반환 | `pd.DataFrame` — columns: [path, true_label, priority_score, cluster_label] |
| 정제 완료 기준 | ARI ≥ 0.6, Silhouette Score ≥ 0.4 |
| 검토 우선순위 | Priority Score 상위 20% 샘플 |

### 4.1 Priority Score 계산

| 기준 | 가중치 |
| --- | --- |
| 클러스터 경계 근접도 (클러스터 중심까지 거리) | 0.5 |
| 예측 신뢰도 역수 (낮은 confidence) | 0.3 |
| 라벨 불일치 빈도 (이전 버전과 차이) | 0.2 |

---

## 5. 라벨 버전 관리

| 버전 | 파일명 | 생성 시점 |
| --- | --- | --- |
| v0 | `data/labels/labels_v0.csv` | S1 초기 라벨링 |
| v1 | `data/labels/labels_v1.csv` | S3 Phase 1 정제 |
| v2 | `data/labels/labels_v2.csv` | S4 Phase 1 재정제 |

### 5.1 labels_vN.csv 스키마

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `path` | str | 패치 파일 경로 |
| `channel` | str | C / M / Y / K |
| `level` | int | 0–5 (0-based) |
| `version` | int | 라벨 버전 번호 |
| `reviewer` | str | 검토자 또는 `auto` |

---

## 6. Phase 1 실행 조건

| 조건 | 기준 |
| --- | --- |
| Phase 0 완료 후 진입 | `phase0_backbone_{ch}_{tag}.pt` 존재 |
| 정제 완료 판단 | ARI ≥ 0.6 또는 Silhouette Score ≥ 0.4 |
| 최대 검토 샘플 수 | Priority Score 상위 20% (전체의 20%만) |

---

## 7. 체크리스트

- [ ] ROI 좌표 변경 시 config.yaml `roi.fixed_coords` 업데이트
- [ ] CMYK 분리 수식 변경 시 §2 동기화
- [ ] 라벨 버전 추가 시 §5 테이블 업데이트
- [ ] Priority Score 가중치 변경 시 §4.1 동기화

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | Hard SSOT 값 (image_size=128) |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | 하위 전처리 파이프라인 |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | ARI/Silhouette 지표 정의 |
