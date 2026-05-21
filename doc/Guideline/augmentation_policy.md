# CMYK Grayspot Detection System
# Augmentation Policy — 증강 정책

Version: 2.2.0
Date: 2026-05-21

---

# 1. Objective — 목적

This document defines the official augmentation policy used in the CMYK Grayspot Detection System dataset pipeline.

본 문서는 CMYK Grayspot Detection System 데이터셋 파이프라인에서 사용하는 공식 증강 정책을 정의한다.

The augmentation system is designed to:

증강 시스템의 설계 목적:

- balance shortage classes / 부족 클래스 균형 보정
- improve dataset consistency / 데이터셋 일관성 향상
- avoid uncontrolled oversampling / 비통제 오버샘플링 방지
- preserve realistic print-defect characteristics / 실제 인쇄 결함 특성 보존

---

# 2. Augmentation Philosophy — 증강 철학

Augmentation is NOT applied globally.

증강은 전역적으로 적용하지 않는다.

The system uses controlled augmentation only for classes that fail PRD minimum sample requirements.

PRD 최소 샘플 요건에 미달하는 클래스에만 제어된 증강을 적용한다.

This prevents:

이를 통해 방지하는 것:

- artificial dataset inflation / 인위적 데이터셋 팽창
- excessive duplication / 과도한 중복
- model bias toward oversampled classes / 오버샘플 클래스 편향
- unrealistic synthetic data generation / 비현실적 합성 데이터 생성

---

# 3. PRD Minimum Targets — PRD 최소 목표 수량

PRD Section 6.3 기준 (v2) / Based on PRD Section 6.3 (v2):

채널당 총 목표 1,400~1,600장, 레벨 비율은 기존 유지 (×3.3 스케일)

Per-channel total target: 1,400–1,600 samples. Level ratio unchanged from v1 (scaled ×3.3).

| Level | Minimum Samples / 최소 샘플 수 | v1 (이전) |
|---|---|---|
| 0 | 330 | 100 |
| 1 | 330 | 100 |
| 2 | 330 | 100 |
| 3 | 265 | 80 |
| 4 | 165 | 50 |
| 5 | 100 | 30 |
| **합계 / Total** | **1,520** | 460 |

> 모든 목표는 채널별 독립 적용 / All targets are applied independently per channel (C/M/Y/K).

---

# 4. Augmentation Trigger Rule — 증강 발동 규칙

Augmentation is applied only when:

다음 조건이 충족될 때만 증강이 적용된다:

```text
current_count < target_count
현재 샘플 수 < 목표 샘플 수
```

If the target count is already satisfied, augmentation is skipped.

목표 수량이 이미 달성된 경우 증강을 건너뛴다.

---

# 5. Allowed Transformations — 허용 변환

The following transformations are allowed:

다음 변환이 허용된다:

- horizontal mirror / 수평 뒤집기
- 90° rotation / 90° 회전
- 180° rotation / 180° 회전
- 270° rotation / 270° 회전

These transformations preserve defect semantics while increasing sample diversity.

이 변환들은 결함 의미를 보존하면서 샘플 다양성을 높인다.

---

# 6. Forbidden Transformations — 금지 변환

The following transformations are prohibited:

다음 변환은 금지된다:

- extreme geometric distortion / 극단적 기하학적 왜곡
- unrealistic scaling / 비현실적 스케일링
- heavy blur / 강한 블러
- synthetic noise injection / 합성 노이즈 주입
- aggressive color modification / 강한 색상 변형

These operations may alter real print-defect characteristics.

이 변환들은 실제 인쇄 결함 특성을 훼손할 수 있다.

---

# 7. Channel Policy — 채널별 정책

모든 채널(C/M/Y/K)에 동일한 PRD v2 목표와 증강 정책이 적용된다.

The same PRD v2 targets and augmentation policy apply to all channels (C/M/Y/K).

## Labeling Rule — 라벨링 원칙

각 채널은 **독립적으로** 라벨링되어야 한다. 같은 스캔에서 나온 Y/M/C/K 채널이라도 각각 다른 결함 레벨을 가질 수 있다.

Each channel must be labeled **independently**. Y/M/C/K channels from the same scan can each have different defect levels.

---

## Y / M / C / K Channels

Augmentation is applied to any (channel, level) pair below the PRD v2 target.

PRD v2 목표에 미달하는 (채널, 레벨) 쌍에 증강이 적용된다.

All four channels use the same target table (§3) and the same allowed transforms (§5).

4개 채널 모두 동일한 목표표(§3)와 허용 변환(§5)을 사용한다.

---

# 8. Duplicate Prevention — 중복 방지

All augmented samples must use unique filenames.

모든 증강 샘플은 고유 파일명을 사용해야 한다.

UUID-based naming is recommended.

UUID 기반 명명 방식을 권장한다.

Example / 예시:

```text
aug_a1b2c3d4.png
```

This prevents filepath collisions and duplicate dataset entries.

이를 통해 경로 충돌과 중복 데이터셋 항목을 방지한다.

Implementation: `src/scripts/augment_dataset.py` uses `uuid.uuid4().hex[:8]` for unique filenames and updates `data_set/labels_master.csv` automatically.

구현: `src/scripts/augment_dataset.py` 가 `uuid.uuid4().hex[:8]` 로 고유 파일명을 생성하고 `data_set/labels_master.csv` 를 자동 갱신한다.

---

# 9. Validation Requirement — 검증 요건

After augmentation, dataset validation must confirm:

증강 후 다음 항목을 데이터셋 검증으로 확인해야 한다:

- no duplicate filepaths / 중복 파일경로 없음
- no missing files / 누락 파일 없음
- valid label mappings / 유효한 라벨 매핑
- correct channel distributions / 채널 분포 정상
- correct level distributions / 레벨 분포 정상

`augment_dataset.py` 가 실행 전 CSV 기준으로 channel×level 카운트를 자동 확인한다.
`augment_dataset.py` automatically checks channel×level counts from the CSV before augmenting.

---

# 10. Data Pipeline Script — 데이터 파이프라인 스크립트

| Script / 스크립트 | Command / 명령 | Role / 역할 |
|---|---|---|
| `augment_dataset.py` | `python -m src.scripts.augment_dataset` | PRD 미달 레벨 증강 + CSV 갱신 |

---

# 11. Final Policy Status — 최종 정책 현황

The augmentation pipeline is now:

증강 파이프라인 현황:

- controlled / 통제됨
- traceable / 추적 가능
- reproducible / 재현 가능
- PRD-compliant / PRD 준수
- dataset-safe / 데이터셋 안전

This policy is considered the canonical augmentation standard for the CMYK dataset pipeline.

본 정책은 CMYK 데이터셋 파이프라인의 공식 증강 표준으로 간주된다.

---

## See Also — 관련 문서

| Document / 문서 | Relation / 관계 |
|---|---|
| [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) | 증강 정책 SSOT 정의 / Augmentation policy SSOT definition |
| [validation_report.md](../Errors/validation_report.md) | 데이터셋 검증 결과 / Dataset validation results |
| [Data_Audit.md](../Errors/Data_Audit.md) | 채널별 현황 감사 / Per-channel audit |
