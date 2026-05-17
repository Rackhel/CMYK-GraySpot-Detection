---
type: ssot
domain: core
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Artifacts.md"
  - "SSOT_Config_Resolution.md"
  - "SSOT_GlobalVariables.md"
  - "SSOT_Validation_Codes.md"
---

# SSOT Core — 핵심 원칙 및 용어 정의 / Core Principles and Terminology

CMYK Grayspot Detection System 의 SSOT 핵심 원칙, 용어, 문서 목록의 기준 문서.

This document defines the core SSOT principles, terminology, and document inventory for the CMYK Grayspot Detection System.

> **목적 / Purpose**: SSOT 핵심 원칙, 용어, 아티팩트 의미 공간을 정의
> **역할 / Role**: "What" — 의미/원칙 정의 (구현 세부는 개별 SSOT 문서 참조)
> **근거 / Basis**: `src/` 런타임 코드 정밀 분석 기준 (config 선언이 아닌 실제 코드 동작)

---

## 1. SSOT 정의 / Definition

**SSOT(Single Source of Truth)**는 데이터/파라미터/아티팩트의 **의미(semantic)를 정의하는 유일 출처**이며, 다른 컴포넌트는 이를 **재정의·우회·대체 없이 그대로 소비**해야 한다.

**SSOT** is the single authoritative source defining the **semantics** of data, parameters, and artifacts. All other components must consume it **without redefining, bypassing, or substituting**.

### 1.1 역할 범위 / Scope

| 구분 / Category | 포함 / Included | 제외 / Excluded |
|------|------|------|
| ✅ SSOT | 학습 Phase 정의, 모델 구조 파라미터, 데이터 분할 규칙, 평가 목표 / Training phase definitions, model structure parameters, data split rules, evaluation targets | — |
| ❌ 비 SSOT / Non-SSOT | — | HTML 리포트 포맷, 디버깅 로그, GUI 레이아웃 / HTML report format, debug logs, GUI layout |

> SSOT는 **의미를 고정하는 최소 단위**까지만 책임진다.
> SSOT is responsible only for fixing the **minimum semantic unit**.

### 1.2 코드 vs Config 우선순위 / Priority

이 프로젝트에서 SSOT 판정 기준 / SSOT authority ranking in this project:

| 순위 / Rank | 원천 / Source | 근거 / Reason |
|------|------|------|
| 1 | **런타임 코드** (실제 실행되는 값) | 코드가 실제 동작을 결정 / Code determines actual behavior |
| 2 | `config/config.json` 소비되는 키 | 코드에서 읽는 값만 유효 / Only consumed keys are valid |
| 3 | — | 미소비 config 키 = Dead Config, SSOT 아님 / Unconsumed = Dead Config |

---

## 2. 핵심 용어 / Glossary

| 용어 / Term | 정의 / Definition | 참조 / Reference |
|------|------|------|
| **Phase** | 학습 단계 (Phase 0: Contrastive, Phase 2: Supervised) / Training stage | §4, SSOT_Training_Pipeline |
| **Channel** | CMYK 인쇄 색상 (Y, M, C, K) — 채널별 독립 모델 / CMYK color channel, independent model per channel | SSOT_Data_Pipeline |
| **Level** | Grayspot 결함 수준 (0~5, 6-class ordinal) / Defect severity level | SSOT_Data_Pipeline |
| **Hard SSOT** | 변경 시 하위 결과가 무효화되는 필수 파라미터 / Parameter change invalidates all downstream results | §6 |
| **Soft SSOT** | 변경 시 간접 영향만 주는 설정 값 / Change affects only performance or speed | §6 |
| **Dead Config** | config.json에 선언되었으나 코드에서 소비되지 않는 키 / Key declared in config but never consumed | SSOT_Config_Resolution |
| **Fail-Fast** | SSOT 위반 발견 즉시 실패. 우회/임시 생성 금지 / Fail immediately on violation, no fallbacks | §5 |
| **backbone_tag** | backbone 이름의 단축 식별자 (`efficientnet_b0` → `effb0`) / Short identifier for backbone name | SSOT_Artifacts |
| **Channel Invariant** | 동일 channel끼리만 Phase 0 → Phase 2 연결 허용 / Only same-channel backbones may flow into Phase 2 | SSOT_Artifacts |

---

## 3. 상태 아이콘 범례 / Icon Legend

| 아이콘 / Icon | 의미 / Meaning | 설명 / Description |
|--------|------|------|
| 🟢 | **소비됨 / Consumed** | config에서 선언되고 코드에서 실제로 사용 / Declared in config and actually consumed by code |
| 🔴 | **미연결 / Dead** | config에서 선언되었으나 코드에서 미소비 (Dead Config) / Declared but never read by code |
| 🟡 | **하드코딩 / Hardcoded** | config 키 없이 코드에 직접 고정된 값 / Value fixed directly in code without config key |
| ⚠️ | **불일치 / Mismatch** | 선언과 구현이 다름 (잠재적 버그) / Declaration differs from implementation |

---

## 4. 프로젝트 정체성 / Project Identity

| 항목 / Item | 값 / Value |
|------|---|
| 프로젝트명 / Project | CMYK Grayspot Detection Pipeline |
| 목적 / Purpose | CMYK 인쇄물의 Grayspot 결함 수준(Level 0~5) 분류 / Classify Grayspot defect level (0~5) from CMYK prints |
| 입력 / Input | 패치 이미지 (128×128 px, 3채널) / Patch image (128×128 px, 3 channels) |
| 출력 / Output | Level 0~5 분류 (6-class) / 6-class classification |
| 학습 방식 / Training | Phase 0 (Contrastive) → Phase 2 (Supervised Classification) |
| Backbone | EfficientNet-B0 (기본 / default) / ResNet50 (대안 / alternative) |
| 프레임워크 / Framework | PyTorch 2.x |
| 데이터 구조 / Data layout | `data_set/labeled/{channel}/{level}/*.png` |
| 채널 / Channels | Y, M, C, K (4색분판 / 4-color separation) |
| 성능 목표 / Performance | Overall Acc ≥ 90%, Macro F1 ≥ 0.85, Per-class F1 ≥ 0.80, MAE ≤ 0.50 |
| 설정 파일 / Config file | `src/config/config.json` |

---

## 5. SOLID 설계 원칙 / SOLID Design Principles

이 프로젝트의 모든 Python 코드는 SOLID 5원칙을 **설계 기준**으로 사용한다.
All Python code in this project uses SOLID principles as **design criteria**.

---

### 5.1 S — 단일 책임 원칙 (SRP) / Single Responsibility Principle

> **하나의 모듈/클래스/함수는 변경 이유(reason to change)가 하나여야 한다.**
> **Each module, class, and function must have exactly one reason to change.**

| 단위 / Unit | 원칙 / Rule | 위반 예시 / Bad Example | 준수 예시 / Good Example |
|---|---|---|---|
| 모듈 / Module | 단일 관심사만 다룸 / Handle a single concern only | `utils.py` — 로깅 + 설정 로딩 + 모델 빌드 혼재 / logging + config loading + model build mixed | `logger.py` / `utils_config.py` / `utils_model.py` 분리 / separated |
| 클래스 / Class | 단일 역할 수행 / Perform a single role | `Trainer` 가 학습 + 평가 + 저장 동시 담당 / `Trainer` handles train + evaluate + save simultaneously | `Phase0Trainer` (학습만 / train only), `Evaluator` (평가만 / evaluate only) |
| 함수 / Function | 한 가지 작업만 수행 / Perform a single task | `run()` 이 전처리 + 학습 + 저장 + 리포트 생성 / `run()` does preprocess + train + save + report | 각 단계를 분리된 함수로 위임 / Delegate each step to a separate function |

**SRP 위반 판단 기준 / SRP violation test**: 함수/클래스 설명에 "그리고(and)"가 필요하면 책임이 두 개다. / If describing a function/class requires the word "and", it has more than one responsibility.

---

### 5.2 O — 개방-폐쇄 원칙 (OCP) / Open-Closed Principle

> **클래스/함수는 확장에는 열려 있고, 수정에는 닫혀 있어야 한다.**
> **Classes and functions must be open for extension but closed for modification.**

기존 코드를 수정하지 않고 새로운 기능을 추가할 수 있어야 한다. 조건 분기(`if/elif`)로 구현된 확장 지점은 OCP 위반이다.
New functionality should be addable without modifying existing code. Conditional branching (`if/elif`) for extension points is an OCP violation.

| 항목 / Item | 현재 패턴 / Current Pattern | OCP 준수 패턴 / OCP-Compliant Pattern |
|---|---|---|
| Backbone 추가 / Add backbone | `build_backbone()` 내부 `if/elif` 수정 필요 / Must modify internal `if/elif` | Registry dict에 등록만 추가 / Register entry in registry dict |
| Optuna 탐색 공간 / Optuna search space | `search_space.py` 내 `if backbone_name == "resnet50"` 분기 수정 / Must modify backbone branch | backbone별 독립 search_space 함수 등록 / Register per-backbone function |
| Head 구조 추가 / Add head variant | `ClassifierHead.__init__` 내 `if mid_dim is not None` 분기 / `if mid_dim` branch inside | Head 클래스를 상속·확장 / Inherit and extend head class |

---

### 5.3 L — 리스코프 치환 원칙 (LSP) / Liskov Substitution Principle

> **파생(서브)클래스는 기반(베이스)클래스를 완전히 대체할 수 있어야 한다.**
> **Derived classes must be fully substitutable for their base class.**

서브클래스가 베이스클래스의 계약(전제 조건, 후행 조건, 불변 조건)을 약화시키면 LSP 위반이다.
A subclass violates LSP if it weakens the base class's contract (preconditions, postconditions, invariants).

**준수 규칙 / Rules**:

| 규칙 / Rule | 설명 / Description |
|---|---|
| `nn.Module` forward 계약 유지 / Honor `nn.Module` forward contract | `forward(x: Tensor) → Tensor` — 입출력 shape 계약 반드시 유지 / Always maintain I/O shape contract |
| Softmax 위치 고정 / Fixed Softmax position | `ClassifierHead.forward()` 는 raw logits만 반환. 서브클래스도 동일. / Always return raw logits; subclasses must do the same |
| Trainer 계약 유지 / Honor Trainer contract | `Phase0Trainer` / `Phase2Trainer` 를 상속할 경우 `train()` 반환 타입 `List[dict]` 유지 필수 / If subclassing trainers, `train()` must return `List[dict]` |


---

### 5.4 I — 인터페이스 분리 원칙 (ISP) / Interface Segregation Principle

> **클라이언트는 사용하지 않는 메서드에 의존하도록 강요받아서는 안 된다.**
> **No client should be forced to depend on methods it does not use.**

하나의 큰 인터페이스보다 목적별로 작은 인터페이스 여러 개가 낫다. / Many small purpose-specific interfaces are better than one large general-purpose interface.

**준수 가이드라인 / Compliance Guidelines**:

| 원칙 / Rule | 설명 / Description |
|---|---|
| Trainer는 필요한 모델 메서드만 호출 / Trainers call only needed model methods | `Phase0Trainer` → `model.forward()` 만 호출. `switch_to_phase2()` 절대 호출 금지 / Call only `model.forward()`; never call `switch_to_phase2()` |
| 평가·추론은 eval 인터페이스만 의존 / Evaluator/Predictor depend only on eval interface | `Evaluator`, `GrayspotPredictor` → `model.eval()` + `model.forward()` 만 사용 / Use only `model.eval()` + `model.forward()` |
| 설정 주입 분리 / Separate config injection | 모듈은 자신이 필요한 cfg 키만 접근. 전체 cfg를 내부 저장하되 필요한 키만 읽음 / Access only needed cfg keys — store full cfg but read selectively |


---

### 5.5 D — 의존 역전 원칙 (DIP) / Dependency Inversion Principle

> **고수준 모듈은 저수준 모듈에 직접 의존해서는 안 된다. 둘 다 추상화에 의존해야 한다.**
> **High-level modules must not depend directly on low-level modules. Both should depend on abstractions.**

구체 클래스(concrete class)가 아닌 추상화(프로토콜·인터페이스·base class)에 의존해야 외부에서 구현체를 교체할 수 있다. / Depending on abstractions rather than concrete classes allows substituting implementations externally.

**준수 규칙 / Rules**:

| 규칙 / Rule | 설명 / Description |
|---|---|
| Trainer → Model: `nn.Module` 타입만 수령 / Accept only `nn.Module` | `Phase0Trainer(model, ...)` — `model`은 `nn.Module`이면 무엇이든 교체 가능 / Any `nn.Module` is acceptable |
| Evaluator → Model: `model.eval()` + `model(x)` 만 의존 / Depend only on eval interface | Evaluator는 model 내부 구조를 알지 못함 / Evaluator has no knowledge of model internals |
| Config 주입 / Config injection | 모든 컴포넌트는 `cfg: dict` 를 외부에서 주입받음 — 내부 파일 로드 금지 / All components receive `cfg: dict` from outside — no internal file loading |


---

## 6. 코딩 컨벤션 / Coding Conventions

이 프로젝트의 모든 Python 코드는 아래 세 원칙을 **의무적으로 준수**한다.
All Python code in this project **must** follow the three principles below.

---

### 6.1 snake_case 명명 규칙 / snake_case Naming Convention

> **모든 Python 식별자는 snake_case를 사용한다. 단, 클래스명은 PascalCase.**
> **All Python identifiers use snake_case. Class names use PascalCase.**

| 식별자 종류 / Identifier | 규칙 / Rule | 예시 / Example |
|---|---|---|
| 변수 / Variable | snake_case | `learning_rate`, `val_acc`, `model_dir` |
| 함수 / Function | snake_case | `load_config()`, `get_nested()`, `build_model()` |
| 모듈/파일 / Module | snake_case | `utils_config.py`, `grayspot_model.py` |
| 클래스 / Class | PascalCase | `GrayspotModel`, `Phase2Trainer`, `EvaluationSummary` |
| 상수 / Constant | UPPER_SNAKE_CASE | `NUM_LEVELS`, `TARGET_MAE`, `CONF_THRESH_AUTO` |
| config 키 / Config key | snake_case (JSON) | `"learning_rate"`, `"num_levels"`, `"data_root"` |

**금지 / Prohibited**: camelCase 변수(`learningRate`), 단어 축약(`lr_rt`), 밑줄 과잉(`__val__`) / camelCase variables, word abbreviations, excessive underscores

---

### 6.2 명시적 명명 / Explicit Naming

> **모든 객체는 역할과 의미를 즉시 알 수 있는 이름을 가진다. 축약·단문자·매직 넘버 금지.**
> **Every object must have a name that immediately conveys its role. No abbreviations, single letters, or magic numbers.**

| 규칙 / Rule | 위반 예시 / Bad | 준수 예시 / Good |
|---|---|---|
| 변수명이 역할을 설명함 / Variable name describes its role | `x`, `tmp`, `d`, `res` | `image_tensor`, `val_loader`, `best_val_acc` |
| 함수명이 동작을 설명함 / Function name describes its action | `process()`, `run()`, `do()` | `augment_contrastive()`, `save_backbone()` |
| 매직 넘버 금지 / No magic numbers | `if score > 0.8` | `if score > cfg["evaluation"]["targets"]["overall_accuracy"]` |
| 불리언은 서술어 형식 / Booleans use predicate form | `flag`, `check` | `is_skipped`, `has_early_stopped`, `use_oversample` |
| 루프 변수도 의미 있게 / Loop variables must be meaningful | `for i in channels` | `for channel in channels` |

**예외 / Exception**: 수학 공식 내 관례적 단문자 (`i`, `j` 인덱스, `z` 임베딩 벡터) 는 허용하되 주석으로 의미 명시. / Conventional single-letter variables in math formulas (`i`, `j` for indices, `z` for embedding vectors) are allowed but must be explained in comments.

---

## 7. Fail-Fast 정책 / Fail-Fast Policy

### 7.1 핵심 선언 / Core Declaration

> **SSOT 위반은 즉시 실패해야 한다. 우회(fallback), 임시 생성, 추측 금지.**
> **SSOT violations must fail immediately. No fallbacks, no temporary creation, no guessing.**

### 7.2 즉시 오류 발생 조건 / Immediate Failure Conditions

| 조건 / Condition | SSOT 코드 / Code | 설명 / Description |
|------|-----------|------|
| 필수 아티팩트 누락 / Missing artifact | `SSOT-FF01` | Phase 0 backbone 없이 Phase 2 시작 / Phase 2 started without Phase 0 backbone |
| Phase 정의 불일치 / Phase violation | `SSOT-PH01` | Phase 0 없이 Phase 2 직행 (backbone 미학습) / Jump directly to Phase 2 without trained backbone |
| 색상 공간 불일치 / Color space mismatch | `SSOT-CS01` | 학습(BGR) ↔ 평가(RGB) 불일치 / Train (BGR) vs. evaluation (RGB) mismatch |
| 정규화 기준 불일치 / Normalization mismatch | `SSOT-NM01` | pretrained backbone이 기대하는 ImageNet norm 미적용 / ImageNet normalization not applied to pretrained backbone |
| config 키 미존재 / Missing config key | `SSOT-CF01` | 코드가 참조하는 config 키가 json에 없음 / Config key referenced in code is absent from config.json |
| Dead Config 감지 / Dead config detected | `SSOT-CF02` | config 키 선언은 있으나 코드에서 미소비 / Config key declared but not consumed by code |
| Seed 재현 실패 / Seed reproducibility | `SSOT-SD01` | 동일 seed에서 다른 결과 (비결정론적 분할) / Same seed produces different results |

### 7.3 Fail-Fast 등급 체계 / Severity Levels

| 등급 / Level | 이름 / Name | 동작 / Action | 예시 / Example |
|------|------|------|------|
| **Level 0** | Panic | 시스템 즉시 중단 / Immediate system halt | 모델 파일 손상, CUDA OOM / Corrupt model file, CUDA OOM |
| **Level 1** | Error | 현재 실행 중단, 명시적 오류 반환 / Abort current run with explicit error | SSOT-FF01, SSOT-PH01, SSOT-CS01 |
| **Level 2** | Warning + Continue | 경고 로그 + 실행 계속 / Log warning and continue | Dead Config 키 감지 (SSOT-CF02) / Dead config key detected |
| **Level 3** | Info | 기록만, 실행 계속 / Log only, continue execution | 권장값과 다른 설정 / Settings different from recommended values |

### 7.4 금지 행위 / Prohibited Actions

| 위반 유형 / Violation | 설명 / Description |
|-----------|------|
| ❌ Fallback 생성 / Create fallback | SSOT 아티팩트 없으면 임시 대체 생성 금지 / No temporary substitutes for missing artifacts |
| ❌ 하드코딩 우회 / Hardcode bypass | config 키 대신 리터럴 사용 금지 / No string literals instead of config keys |
| ❌ Phase 건너뛰기 / Skip phase | Phase 0 → Phase 2 순서 위반 금지 / Phase ordering must be respected |
| ❌ 색상 공간 무시 / Ignore color space | BGR/RGB 불일치를 묵인하고 진행 금지 / Never silently accept BGR/RGB mismatch |

---

## 8. Hard / Soft SSOT 분류 / Classification

| 구분 / Category | 예시 / Examples | 변경 영향 / Impact | 변경 시 조치 / Action |
|------|------|----------|-------------|
| **Hard SSOT** | `num_levels=6`, `image_size=128`, `backbone`, Phase 순서, 색상 공간 | 하위 결과 전체 무효 / All downstream results invalidated | 재학습 필수 / Retrain required |
| **Soft SSOT** | `batch_size`, `num_workers`, 로깅 레벨, 리포트 포맷 | 간접 영향 (성능·속도) / Indirect effect on performance or speed | 재학습 선택 / Retrain optional |

> 📌 **판단 원칙 / Decision Principle**: "기존 체크포인트를 그대로 쓸 수 있는가?"
> — 쓸 수 없으면 Hard, 쓸 수 있으면 Soft.
> — If the existing checkpoint can no longer be used as-is, it's Hard. Otherwise it's Soft.

### 8.1 판단 기준 / Decision Criteria

> **Hard SSOT 기준 / Hard SSOT when:**
>
> 1. 값 변경 시 모델 **구조**가 달라지는 경우 (`num_levels`, `backbone`, `hidden_dim`) / Model structure changes
> 2. 값 변경 시 **입력 분포**가 달라지는 경우 (`image_size`, 정규화 방식, 색상 순서) / Input distribution changes
> 3. 값 변경 시 **학습 결과 의미가 달라지는** 경우 (Phase 순서, loss function) / Semantic meaning of results changes
>
> **Soft SSOT 기준 / Soft SSOT when:**
>
> 1. 값 변경 시 **성능만** 달라지는 하이퍼파라미터 (`lr`, `weight_decay`, `batch_size`) / Only performance changes
> 2. **실행 환경에만** 영향을 미치는 파라미터 (`num_workers`, `pin_memory`, `device`) / Only execution environment affected

---

## 9. SSOT 문서 목록 / Document Index

모든 SSOT 문서는 `doc/` 폴더에 위치한다. / All SSOT documents reside in the `doc/` folder.

| 문서 / Document | 역할 / Role | 핵심 관심사 / Key Concerns |
|------|------|------------|
| [SSOT_Core.md](SSOT_Core.md) | 이 문서 — 핵심 원칙 / This document — core principles | SSOT 정의, SOLID 원칙, 코딩 컨벤션, Fail-Fast, Hard/Soft 분류 / SSOT definition, SOLID principles, coding conventions, Fail-Fast, Hard/Soft classification |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | 데이터 로딩, 분할, 전처리, 증강 / Data loading, split, preprocessing, augmentation | 입력 분포, 분할 재현성 / Input distribution, split reproducibility |
| [SSOT_Model_Architecture.md](SSOT_Model_Architecture.md) | 모델 구조, backbone, head 정의 / Model structure, backbone, head | 레이어 구성, feature dim / Layer structure, feature dim |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | 학습 루프, optimizer, loss 정의 / Training loop, optimizer, loss | Phase 순서, 실제 학습 파라미터 / Phase ordering, actual training parameters |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | 평가 지표, 목표값, 리포트 / Evaluation metrics, targets, reports | 지표 정의, 합격 기준 / Metric definitions, pass thresholds |
| [SSOT_Config_Resolution.md](SSOT_Config_Resolution.md) | config 키별 소비 현황 대조표 / Config key consumption cross-reference | 선언 vs 실제 소비 / Declared vs. actually consumed |
| [SSOT_Artifacts.md](SSOT_Artifacts.md) | 산출물 파일명 패턴 및 스키마 / Artifact filename patterns and schemas | 모델 파일, 리포트 경로 / Model files, report paths |
| [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) | 검증 에러 코드 정의 / Validation error code definitions | Fail-Fast 코드 목록 / Fail-Fast code list |
| [SSOT_GlobalVariables.md](SSOT_GlobalVariables.md) | 전역 변수 / 하드코딩 정책 / Global variables and hardcoding policy | Hard/Soft 분류 전체 목록 / Full Hard/Soft classification list |

### 9.1 문서 계층도 / Document Hierarchy

```
SSOT_Core.md                    ← 최상위 원칙 (본 문서) / Top-level principles (this document)
├── SSOT_Config_Resolution.md   ← config 키 전체 대조표 / Full config key cross-reference
├── SSOT_GlobalVariables.md     ← Hard/Soft 변수 목록 / Hard/Soft variable list
├── SSOT_Validation_Codes.md    ← 에러 코드 정의 / Error code definitions
├── SSOT_Model_Architecture.md  ← 모델 구조 / Model architecture
├── SSOT_Data_Pipeline.md       ← 데이터 파이프라인 / Data pipeline
├── SSOT_Training_Pipeline.md   ← 학습 파이프라인 / Training pipeline
├── SSOT_Evaluation_Reporting.md← 평가 및 보고 / Evaluation and reporting
└── SSOT_Artifacts.md           ← 산출물 스키마 / Artifact schema
```

---

## 체크리스트 / Checklist

- [ ] 새 config 키 추가 시 `SSOT_Config_Resolution.md` 업데이트 / Update when adding new config key
- [ ] 새 에러 코드 추가 시 `SSOT_Validation_Codes.md` 업데이트 / Update when adding new error code
- [ ] Hard SSOT 값 변경 시 모든 체크포인트 재학습 / Retrain all checkpoints when Hard SSOT value changes
- [ ] 모델 구조 변경 시 `SSOT_Model_Architecture.md` 동기화 / Sync `SSOT_Model_Architecture.md` on model structure change
- [ ] Dead Config 발견 시 제거 또는 연결 / Remove or connect Dead Config when found

---

## 10. 관련 문서 / Related Documents

| 문서 / Document | 경로 / Path | 역할 / Role |
|------|------|------|
| 의존성 그래프 / Dependency graph | `src/config/dependencies.json` | 모듈 의존성 레이어 구조 / Module dependency layer structure |
| 설정 파일 / Config file | `src/config/config.json` | 런타임 파라미터 SSOT / Runtime parameter SSOT |
| Config 로딩 인터페이스 / Config loading interface | `src/utils/utils_config.py` | `load_config()` → `dict`, `validate_config()`, `create_directories()`, `get_nested()` |
| 모델 유틸리티 / Model utilities | `src/utils/utils_model.py` | `set_seed()`, `backbone_tag()`, `build_model()` |
| 튜닝 유틸리티 / Tuning utilities | `src/tuning/optuna_utils.py` | `normalize_channel()`, `load_best_params()`, `save_best_params()`, `save_trials_summary()`, `apply_phase2_params()` |
| 탐색 공간 / Search space | `src/tuning/search_space.py` | `get_phase2_search_space(trial, cfg=None)` — backbone-aware HPO |
| Markdown 가이드 / Markdown guide | `doc/Markdown_guide.md` | 문서 작성 스타일 가이드 / Document writing style guide |
| CI 설정 / CI Setup | `doc/CI_Setup.md` | GitHub Actions workflow 및 파이프라인 규칙 / GitHub Actions workflow and pipeline rules — 파이프라인 트리거, 테스트 매트릭스, 형식 정책 / Pipeline triggers, test matrix, formatting policy |
| 인터페이스 계약 / Interface Contract | `doc/Contract.md` | 모듈 간 공개 API 계약 — PRD → SSOT 아래, BDD 위 / Public API contracts between modules (sits below PRD→SSOT, above BDD) |
| 행위 명세 / BDD | `doc/BDD.md` | Gherkin 시나리오 — 액터 관점 동작 명세 (PRD → SSOT/Contract → BDD → TDD → Code) / Gherkin scenarios from actor perspective |
| 설계 연쇄 체계 결정 / Design chain ADR | `doc/ADR_Design_Doc_Chain.md` | PRD → SSOT/Contract → BDD → TDD → Code 채택 근거 / Rationale for adopting the design document chain |
| 데이터셋 현황 감사 / Dataset audit | `doc/Data_Audit.md` | PRD 목표 대비 실제 데이터 수량, CSV-폴더 불일치, 라벨링 배치 이력 / Actual vs PRD target quantities, CSV-folder mismatch, labeling batch history |

---

## 11. Inference and Deployment / 추론 및 배포

### 11.1 Inference Pipeline / 추론 파이프라인

> **모든 추론은 `GrayspotPredictor` 클래스를 통해서만 수행된다. 직접 모델 로드 금지.**
> **All inference must be performed through the `GrayspotPredictor` class. Direct model loading is prohibited.**

#### Core Interface / 핵심 인터페이스

```
inference/
├── predictor_device.py    — DeviceMixin      (장치 감지·설정 / Device detection)
├── predictor_loader.py    — ModelLoaderMixin (모델 로딩·캐시 / Loading & cache)
├── predictor_inference.py — InferenceMixin   (추론 실행 + SSOT-NM01 / Inference + NM01)
└── predictor.py           — GrayspotPredictor (Orchestrator)
```

### 11.2 Deployment Requirements / 배포 요구사항

| 요구사항 / Requirement | 설명 / Description | 검증 방법 / Validation |
|------|------|------|
| **Model Artifacts** | Phase 2 체크포인트 파일 존재 / Phase 2 checkpoint files exist | `SSOT-FF01` — 필수 아티팩트 누락 / Missing artifact |
| **Config Consistency** | 추론 시 학습 config와 동일 / Same config as training | `SSOT-CF01` — config 키 불일치 / Config key mismatch |
| **Color Space** | BGR 입력 유지 / Maintain BGR input | `SSOT-CS01` — 색상 공간 불일치 / Color space mismatch |
| **Normalization** | ImageNet 정규화 적용 / Apply ImageNet normalization | `SSOT-NM01` — 정규화 불일치 / Normalization mismatch |

### 11.3 Docker Deployment / Docker 배포

멀티 스테이지 빌드를 사용하여 base 이미지 위에 inference 이미지를 구성한다. / Multi-stage build: inference image built on top of base image.

### 11.4 Performance Targets / 성능 목표

| 지표 / Metric | 목표값 / Target | 측정 방법 / Measurement |
|------|------|------|
| 추론 시간 / Inference time | < 100ms per image | Single image latency |
| 배치 처리량 / Batch throughput | > 50 images/sec | GPU batch processing |
| 메모리 사용 / Memory usage | < 2GB per model | Peak GPU memory |
| ONNX 호환성 / ONNX compatibility | opset 11+ | Export and runtime validation |

---

## See Also

| 문서 / Document | 관계 / Relation |
| --- | --- |
| [SSOT_Config_Resolution.md](SSOT_Config_Resolution.md) | config 키 소비 상세 / Config key consumption detail |
| [SSOT_GlobalVariables.md](SSOT_GlobalVariables.md) | Hard/Soft 분류 전체 / Full Hard/Soft classification |
| [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) | 에러 코드 체계 / Error code system |
| [Contract.md](../Contract/Contract.md) | 구현 세부 계약 / Implementation contracts |

