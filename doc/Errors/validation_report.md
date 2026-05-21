# CMYK Grayspot Detection System
# Dataset Validation Report — 데이터셋 검증 보고서

Version: 3.1.0
Date: 2026-05-21
Label Source: `data_set/labels_master.csv`
Status: **✅ PRD v2 Complete — 6,080 rows / PRD v2 완료 — 6,080행**

---

# 1. Validation Objective — 검증 목적

This report documents the current validation status of the CMYK Grayspot Detection System dataset.

본 보고서는 CMYK Grayspot Detection System 데이터셋의 현재 검증 현황을 기록한다.

---

# 2. Dataset Reset Summary — 데이터셋 초기화 요약

2026-05-21: Old labeled dataset (8,969 images) was cleared and `labels_master.csv` was reset to headers only.

2026-05-21: OLD labeled 데이터셋(8,969장) 삭제 완료, `labels_master.csv` 헤더만 유지하여 초기화.

**이유 / Reason:**

- Old labeled data was from previous pipeline with incorrect augmentation multipliers / 이전 파이프라인의 잘못된 증강 배율 적용 데이터
- M channel had 6,273 images vs new target of 1,520 / M 채널 6,273장 vs 새 목표 1,520장 불일치
- Channels were not independently labeled — all channels inherited the same level from raw filename / 채널 독립 라벨링 미적용 — 파일명 레벨을 전 채널에 복사

---

# 3. New PRD Targets — 새 PRD 목표 (v2)

PRD Section 6.3 v2: 채널당 총 1,400~1,600장, 레벨 비율 유지 (×3.3)

| Level | Target per Channel / 채널당 목표 | v1 목표 |
|---|---|---|
| 0 | 330 | 100 |
| 1 | 330 | 100 |
| 2 | 330 | 100 |
| 3 | 265 | 80 |
| 4 | 165 | 50 |
| 5 | 100 | 30 |
| **합계 / Total** | **1,520** | 460 |

> 모든 목표는 채널(C/M/Y/K)별 독립 적용 / All targets applied independently per channel.

---

# 4. Current Dataset Status — 현재 데이터셋 현황

| Component / 구성요소 | Count / 수량 | Status / 상태 |
|---|---|---|
| Raw scans (new) / 새 스캔 | 288 jpeg (lvlX_...) | ✅ 수집 완료 |
| Raw scans (old) / 이전 스캔 | 190 jpeg (scan_XXX_) | ⚠️ 재사용 여부 미결정 |
| ROI files / ROI 파일 | 1,152 png (288 × 4ch) | ✅ CMYK 분리 완료 |
| Labeled patches (original) / 원본 패치 | **5,300** | ✅ 완료 |
| Labeled patches (augmented) / 증강 패치 | **780** | ✅ 완료 |
| **Labeled patches (total) / 전체** | **6,080** | ✅ PRD v2 달성 |
| labels_master.csv rows / 라벨 행 수 | **6,080** | ✅ 완료 |

**채널×레벨 분포 / Channel × Level distribution:**

| Ch | L0 | L1 | L2 | L3 | L4 | L5 | 합계 |
|---|---|---|---|---|---|---|---|
| C | 330 | 330 | 330 | 265 | 165 | 100 | 1,520 |
| M | 330 | 330 | 330 | 265 | 165 | 100 | 1,520 |
| Y | 330 | 330 | 330 | 265 | 165 | 100 | 1,520 |
| K | 330 | 330 | 330 | 265 | 165 | 100 | 1,520 |

> ⚠️ 현재 라벨링은 파일명 기반(스캔 단위). 채널별 독립 시각 검사 후 `data_set/roi_labels.csv`로 오버라이드 가능.
> Current labeling is filename-based (scan-level). Per-channel visual inspection results can override via `data_set/roi_labels.csv`.

---

# 5. Completed Actions — 완료된 작업

| # | Action / 작업 | Status |
|---|---|---|
| 1 | OLD labeled data (8,969장) 삭제 및 labels_master.csv 초기화 | ✅ 완료 |
| 2 | ROI 파일에서 패치 추출 (`prepare_dataset.py`) | ✅ 완료 |
| 3 | PRD v2 목표 달성 (`augment_dataset.py` 자동 호출) | ✅ 완료 |
| 4 | ROIExtractor 클래스 구현 및 TDD 테스트 10개 통과 | ✅ 완료 |
| 5 | roi_labels.csv 채널별 독립 라벨링 오버라이드 지원 추가 | ✅ 완료 |

**전체 파이프라인 단일 명령 / Full pipeline single command:**
```bash
python -m src.scripts.prepare_dataset
```

---

# 6. Labeling Policy — 라벨링 정책

**Required / 필수**: Each CMYK channel must be labeled **independently** for each scan.

각 스캔의 CMYK 채널은 **독립적으로** 라벨링해야 한다.

```
예시 / Example:
  scan_file.jpeg → ROI split:
    Y: inspect Y channel image → assign level (e.g., 0)
    M: inspect M channel image → assign level (e.g., 2)
    C: inspect C channel image → assign level (e.g., 3)
    K: inspect K channel image → assign level (e.g., 1)
```

ROI files are at: `data_set/roi/lvlX_..._CH.png`

---

# 7. Validation Checks — 검증 항목 (재구성 완료 후 수행)

- Missing files / 누락 파일
- Duplicate filepaths / 중복 경로
- Channel×level distribution vs PRD v2 targets / 채널×레벨 분포 vs PRD v2 목표
- labels_master.csv integrity / labels_master.csv 무결성

---

## See Also — 관련 문서

| Document / 문서 | Relation / 관계 |
|---|---|
| [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) | 데이터 파이프라인 SSOT |
| [augmentation_policy.md](../Guideline/augmentation_policy.md) | PRD v2 증강 정책 |
| [Data_Relabeling_Strategy.md](../Guideline/Data_Relabeling_Strategy.md) | 재라벨링 전략 및 프로세스 |
| `src/scripts/augment_dataset.py` | 증강 + CSV 갱신 (PRD v2 목표 반영) |
