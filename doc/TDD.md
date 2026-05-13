---
title: TDD 전략 / Test-Driven Development Strategy
version: 0.2.0
last_updated: 2026-05-12
scope: CMYK Grayspot Detection System — 전체 모듈 / All modules
---

# TDD 전략 / Test-Driven Development Strategy

## 목차 / Table of Contents

1. [TDD 원칙](#1-tdd-원칙--tdd-principles)
2. [테스트 피라미드](#2-테스트-피라미드--test-pyramid)
3. [모듈별 테스트 전략](#3-모듈별-테스트-전략--per-module-strategy)
4. [파일 구조 및 네이밍](#4-파일-구조-및-네이밍--file-structure--naming)
5. [Fixture 전략](#5-fixture-전략--fixture-strategy)
6. [테스트 작성 규칙](#6-테스트-작성-규칙--test-writing-rules)
7. [실행 방법](#7-실행-방법--how-to-run)
8. [제외 범위](#8-제외-범위--out-of-scope)

---

## 1. TDD 원칙 / TDD Principles

### 1.1 Red → Green → Refactor 사이클 / Red → Green → Refactor Cycle

```
1. RED    — 실패하는 테스트를 먼저 작성한다. 구현 없이 테스트만. / Write a failing test first. No implementation yet.
2. GREEN  — 테스트를 통과하는 최소한의 코드만 작성한다. / Write the minimum code to make the test pass.
3. REFACTOR — 동작을 유지하면서 코드를 정리한다. 테스트는 GREEN 유지. / Refactor while keeping tests GREEN.
```

**이 프로젝트에서의 의미 / Meaning in this project:**
- 새 함수/클래스 구현 전 → 해당 함수의 단위 테스트를 `tests/unit/`에 먼저 작성 / Before implementing a new function/class → write its unit test in `tests/unit/` first
- 기존 코드 리팩토링 전 → 리팩토링 대상 함수의 테스트가 없으면 먼저 작성 / Before refactoring existing code → write a test first if none exists
- 버그 수정 전 → 버그를 재현하는 테스트를 먼저 작성 / Before fixing a bug → write a test that reproduces the bug first

### 1.2 이 프로젝트 TDD의 범위 / Scope of TDD in This Project

ML 프로젝트 특성상 **모든 것을 TDD로 할 수 없다.** 아래 기준으로 범위를 나눈다:
Due to the nature of ML projects, **not everything can be covered by TDD.** The scope is divided as follows:

| 적용 대상 / In-scope ✅ | 제외 대상 / Out-of-scope ❌ |
|------------|------------|
| 순수 계산 함수 (metrics, losses, config) / Pure computation functions | 모델 학습 루프 전체 / Full model training loop |
| 텐서 shape / dtype / range 보장 / Tensor shape / dtype / range guarantees | GPU/MPS 의존 통합 테스트 / GPU/MPS-dependent integration tests |
| Config 로딩 / 검증 로직 / Config loading / validation logic | 실제 이미지 파일 필요한 E2E / E2E tests requiring real image files |
| 데이터 전처리 / 증강 변환 / Data preprocessing / augmentation transforms | Optuna 튜닝 전체 실행 / Full Optuna tuning runs |
| 평가 지표 계산 / Evaluation metric computation | HTML 리포트 렌더링 시각 검증 / Visual verification of HTML report rendering |

---

## 2. 테스트 피라미드 / Test Pyramid

```
          ▲
         /E2E\          ← 현재 tests/ 의 test_training_phase*.py
        /──────\           실제 학습 + 저장 + 성능 확인 (Smoke)
       /  통합   \       ← tests/integration/ (DataLoader ↔ Trainer 등)
      /────────────\
     /   단위 테스트  \   ← tests/unit/ (순수 함수, 텐서 변환)  ← TDD 적용 대상
    /────────────────────\
```

### 계층별 역할 / Per-Layer Roles

| 계층 / Layer | 경로 / Path | 속도 / Speed | 역할 / Role |
|------|------|------|------|
| **Unit** | `src/tests/unit/` | < 1초/개 / < 1s each | 함수 단위 정확성, 엣지케이스 / Per-function correctness and edge cases |
| **Integration** | `src/tests/integration/` | 수 초 / Several seconds | 모듈 간 데이터 흐름 검증 / Cross-module data flow validation |
| **Smoke / E2E** | `src/tests/` (기존 / existing) | 수 분 / Several minutes | 전체 파이프라인 동작 확인 / Full pipeline operation check |

---

## 3. 모듈별 테스트 전략 / Per-Module Strategy

### 3.1 `utils/` — 기반 모듈 / Base Utilities

**파일 / Files:** `tests/unit/test_utils_config.py`, `tests/unit/test_utils_model.py`

| 함수 / Function | 테스트 항목 / Test Items |
|------|------------|
| `load_config()` | 정상 로드, 필수 키 존재, `system.device` 해석 / Normal load, required keys present, `system.device` resolution |
| `validate_config()` | 필수 키 누락 시 `KeyError` / `ValueError` / `KeyError` / `ValueError` on missing required keys |
| `get_nested()` | 중첩 키 정상 접근, 없는 키 `default` 반환 / Normal nested key access, returns `default` for missing keys |
| `_resolve_device()` | `"auto"` → `"mps"` (Mac), `"cuda"` 없으면 fallback / `"auto"` → `"mps"` (Mac), fallback if `"cuda"` unavailable |
| `backbone_tag()` | `"efficientnet_b0"` → `"effb0"`, 미등록 backbone 처리 / `"efficientnet_b0"` → `"effb0"`, handling unregistered backbones |
| `set_seed()` | 동일 seed → 동일 난수 재현 / Same seed → same random values reproduced |

```python
# 예시 / Example
def test_get_nested_missing_key_returns_default():
    cfg = {"a": {"b": 1}}
    assert get_nested(cfg, "a.c", default=0) == 0

def test_resolve_device_auto_returns_mps_on_mac():
    # torch.backends.mps.is_available() 가 True인 환경 가정 / Assumes torch.backends.mps.is_available() is True
    result = _resolve_device("auto")
    assert result in ("mps", "cuda", "cpu")  # 환경 의존, 타입만 확인 / Environment-dependent, check type only
```

---

### 3.2 `data/` — 전처리 / 증강 / Preprocessing / Augmentation

**파일 / Files:** `tests/unit/test_preprocessing.py`, `tests/unit/test_augmentation.py`

| 함수 / Function | 테스트 항목 / Test Items |
|------|------------|
| `preprocess(img)` | 출력 shape `(3, H, W)`, dtype `float32`, range `[0, 1]` / Output shape `(3, H, W)`, dtype `float32`, range `[0, 1]` |
| `augment_supervised(tensor)` | shape 보존, dtype 보존 / Shape preserved, dtype preserved |
| `augment_contrastive(tensor)` | `(v1, v2)` shape 동일, 두 view가 동일하지 않음 (다양성 보장) / `(v1, v2)` same shape, two views are not identical (diversity guaranteed) |

```python
# 예시 / Example
def test_preprocess_output_range():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    tensor = preprocess(img, size=128)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert tensor.shape == (3, 128, 128)

def test_augment_contrastive_produces_different_views():
    tensor = torch.rand(3, 128, 128)
    v1, v2 = augment_contrastive(tensor)
    assert v1.shape == v2.shape
    assert not torch.allclose(v1, v2)  # 두 view는 달라야 함 / The two views must differ
```

---

### 3.3 `models/` — 신경망 구성 요소 / Neural Network Components

**파일 / Files:** `tests/unit/test_models.py`

| 클래스/함수 / Class / Function | 테스트 항목 / Test Items |
|------------|------------|
| `build_backbone("efficientnet_b0")` | 출력 feature_dim `1280` / Output feature_dim `1280` |
| `build_backbone("resnet50")` | 출력 feature_dim `2048` / Output feature_dim `2048` |
| `ClassifierHead(in_dim=1280, mid_dim=None)` | EfficientNet-B0 직접 압축 — 출력 shape `(batch, 6)`, Linear 2개 / Direct compression — output shape `(batch, 6)`, 2 Linear layers |
| `ClassifierHead(in_dim=2048, mid_dim=512)` | ResNet-50 단계적 압축 — 출력 shape `(batch, 6)`, Linear 3개 / Staged compression — output shape `(batch, 6)`, 3 Linear layers |
| `ProjectionHead(in_dim, proj_dim=128)` | 출력 shape `(batch, 128)` / Output shape `(batch, 128)` |
| `GrayspotModel(cfg, phase=0)` | 출력 shape `(batch, projection_dim)` / Output shape `(batch, projection_dim)` |
| `GrayspotModel(cfg, phase=2)` (EffB0) | 출력 shape `(batch, 6)`, head Linear 2개 (mid_dim 없음) / output shape `(batch, 6)`, 2 Linear layers (no mid_dim) |
| `GrayspotModel(cfg, phase=2)` (Res50) | 출력 shape `(batch, 6)`, head Linear 3개 (mid_dim=512 반영) / output shape `(batch, 6)`, 3 Linear layers (mid_dim=512) |

```python
# 예시 / Example
def test_grayspot_model_phase0_output_shape(minimal_cfg):
    model = GrayspotModel(minimal_cfg, phase=0)
    dummy = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, minimal_cfg["phase0"]["projection_dim"])

def test_classifier_head_effb0_direct_compression():
    head = ClassifierHead(in_dim=1280, hidden_dim=256, num_classes=6, dropout=0.2, mid_dim=None)
    x = torch.randn(4, 1280)
    assert head(x).shape == (4, 6)

def test_classifier_head_resnet50_staged_compression():
    head = ClassifierHead(in_dim=2048, hidden_dim=256, num_classes=6, dropout=0.4, mid_dim=512)
    x = torch.randn(4, 2048)
    assert head(x).shape == (4, 6)

def test_grayspot_model_phase2_effb0_output_shape(minimal_cfg):
    minimal_cfg["model"]["backbone"] = "efficientnet_b0"
    model = GrayspotModel(minimal_cfg, phase=2)
    dummy = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 6)

def test_grayspot_model_phase2_resnet50_output_shape(minimal_cfg):
    minimal_cfg["model"]["backbone"] = "resnet50"
    model = GrayspotModel(minimal_cfg, phase=2)
    dummy = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 6)
```

---

### 3.4 `training/` — 손실 함수 / Loss Functions

**파일 / Files:** `tests/unit/test_losses.py`

| 클래스/함수 / Class / Function | 테스트 항목 / Test Items |
|------------|------------|
| `InfoNCELoss(temperature=0.1)` | 반환값 scalar, 완전 유사 pair → 낮은 loss / Returns scalar, perfectly similar pair → low loss |
| `InfoNCELoss` | 완전 무관 pair → 높은 loss (상대적 검증) / Completely dissimilar pair → high loss (relative verification) |
| `get_loss("infonce")` | `InfoNCELoss` 인스턴스 반환 / Returns `InfoNCELoss` instance |
| `get_loss("cross_entropy")` | `nn.CrossEntropyLoss` 인스턴스 반환 / Returns `nn.CrossEntropyLoss` instance |

```python
# 예시 / Example
def test_infonce_loss_is_scalar():
    loss_fn = InfoNCELoss(temperature=0.1)
    z1 = F.normalize(torch.randn(8, 128), dim=1)
    z2 = F.normalize(torch.randn(8, 128), dim=1)
    loss = loss_fn(z1, z2)
    assert loss.ndim == 0  # scalar

def test_infonce_similar_pairs_lower_loss():
    loss_fn = InfoNCELoss(temperature=0.1)
    z = F.normalize(torch.randn(8, 128), dim=1)
    similar_loss = loss_fn(z, z + torch.randn_like(z) * 0.01)
    random_loss  = loss_fn(z, F.normalize(torch.randn(8, 128), dim=1))
    assert similar_loss < random_loss
```

---

### 3.5 `evaluation/` — 지표 계산 / Metric Computation ← TDD 핵심 영역 / Core TDD Area

**파일 / Files:** `tests/unit/test_metrics.py`, `tests/unit/test_confusion.py`

| 함수 / Function | 테스트 항목 / Test Items |
|------|------------|
| `compute_metrics(y_true, y_pred, confidences)` | accuracy, f1, mae 값 정확성 / accuracy, f1, mae value correctness |
| `build_evaluation_summary(results, cfg=cfg)` | `swing_thresholds` cfg 주입 반영 / `swing_thresholds` cfg injection reflected |
| `determine_swing_feedback(summary)` | PASS / RETRY / MANUAL 분기 정확성 / PASS / RETRY / MANUAL branching correctness |
| `check_targets(summary)` | TARGET_OVERALL_ACC 기준 pass/fail 판정 / pass/fail decision based on TARGET_OVERALL_ACC |
| `plot_confusion_matrix()` | 반환 타입 (Figure), 예외 없이 실행 / Return type (Figure), runs without exceptions |

```python
# 예시 / Example
def test_compute_metrics_perfect_accuracy():
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    confs  = [1.0] * 6
    m = compute_metrics(y_true, y_pred, confs)
    assert m["accuracy"] == 1.0
    assert m["mae"] == 0.0

def test_build_evaluation_summary_injects_swing_thresholds():
    cfg = {"evaluation": {"swing_thresholds": {"acc_retry": 0.70}}}
    summary = build_evaluation_summary({}, cfg=cfg)
    assert summary.targets["swing_acc_retry"] == 0.70

def test_determine_swing_feedback_returns_retry_below_threshold():
    # acc < swing_acc_retry → RETRY  # acc < swing_acc_retry → RETRY
    ...
```

---

### 3.6 `inference/` — 추론 파이프라인 / Inference Pipeline

**파일 / Files:** `tests/unit/test_predictor.py`

단위 테스트 대상이 적음 (모델 로드 의존). 아래만 단위 테스트:
Few unit test targets (model load dependent). Only the following are unit tested:

| 항목 / Item | 테스트 내용 / Test Content |
|------|------------|
| confidence threshold 분기 / Confidence threshold branching | `>= 0.8` → AUTO, `0.5–0.8` → WARN, `< 0.3` → MANUAL |
| 반환 dict 구조 / Return dict structure | `{"label", "confidence", "flag"}` 키 존재 / keys present |

나머지는 `tests/integration/test_predictor_integration.py` 에서 처리.
The remainder is handled in `tests/integration/test_predictor_integration.py`.

---

## 4. 파일 구조 및 네이밍 / File Structure & Naming

```
src/
└── tests/
    ├── unit/                          ← TDD 단위 테스트 (pytest) / TDD unit tests (pytest)
    │   ├── conftest.py                ← 공용 Fixture / Shared fixtures
    │   ├── test_utils_config.py
    │   ├── test_utils_model.py
    │   ├── test_preprocessing.py
    │   ├── test_augmentation.py
    │   ├── test_models.py
    │   ├── test_losses.py
    │   ├── test_metrics.py
    │   └── test_confusion.py
    ├── integration/                   ← 모듈 간 연결 검증 / Cross-module connection validation
    │   ├── conftest.py
    │   ├── test_data_pipeline.py      ← preprocess → dataset → dataloader
    │   └── test_predictor_integration.py
    ├── test_training_phase0.py        ← 기존 Smoke 테스트 (유지) / Existing smoke tests (kept)
    ├── test_training_phase2.py        ← 기존 Smoke 테스트 (유지) / Existing smoke tests (kept)
    ├── test_evaluation.py             ← 기존 Smoke 테스트 (유지) / Existing smoke tests (kept)
    └── test_optuna.py                 ← 기존 Smoke 테스트 (유지) / Existing smoke tests (kept)
```

**네이밍 규칙 / Naming Conventions:**
- 파일 / File: `test_{모듈명 / module_name}.py`
- 함수 / Function: `test_{함수명 / function_name}_{시나리오 / scenario}()`
- 예 / Example: `test_compute_metrics_perfect_accuracy()`, `test_load_config_missing_key_raises()`

---

## 5. Fixture 전략 / Fixture Strategy

모든 공용 Fixture는 `tests/unit/conftest.py`에 정의한다.
All shared fixtures are defined in `tests/unit/conftest.py`.

```python
# src/tests/unit/conftest.py

import pytest
import torch

@pytest.fixture
def minimal_cfg():
    """실제 파일 없이 동작하는 최소 config dict / Minimal config dict without real files"""
    return {
        "system":   {"device": "cpu"},
        "data":     {"channels": ["Y"], "num_levels": 6, "image_size": 128,
                     "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
        "model":    {"backbone": "efficientnet_b0", "frozen_backbone": False},
        "phase0":   {"projection_dim": 128, "hidden_dim": 256, "temperature": 0.1,
                     "epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 1e-5,
                     "augmentation": {}},
        "phase2":   {"dropout": 0.3, "hidden_dim": 256, "epochs": 1, "batch_size": 2,
                     "learning_rate": 1e-4, "weight_decay": 1e-4, "oversample": False,
                     "early_stopping": {"enabled": False, "patience": 5, "min_delta": 1e-4},
                     "augmentation": {},
                     "heads": {
                         "efficientnet_b0": {"mid_dim": None,  "hidden_dim": 256, "dropout": 0.2},
                         "resnet50":        {"mid_dim": 512,   "hidden_dim": 256, "dropout": 0.4},
                     }},
        "evaluation": {"targets": {"overall_accuracy": 0.90, "per_color_accuracy": 0.85,
                                   "per_class_f1": 0.80, "mae": 0.50},
                       "swing_thresholds": {"acc_retry": 0.80, "f1_retry": 0.70, "mae_retry": 0.80}},
        "storage":  {"data_root": "data_set", "labeled_dir": "data_set/labeled",
                     "models_dir": "data_set/models", "reports_dir": "data_set/reports",
                     "logs_dir": "outputs/logs"},
        "train":    {"seed": 42},
        "inference": {"confidence_thresholds": {"auto_accept": 0.8,
                                                "warn_threshold": 0.5,
                                                "manual_review": 0.3}},
    }

@pytest.fixture
def dummy_tensor():
    """(3, 128, 128) float32 랜덤 텐서 / Random float32 tensor of shape (3, 128, 128)"""
    return torch.rand(3, 128, 128)

@pytest.fixture
def dummy_batch():
    """(4, 3, 128, 128) float32 배치 텐서 / Float32 batch tensor of shape (4, 3, 128, 128)"""
    return torch.rand(4, 3, 128, 128)

@pytest.fixture
def perfect_predictions():
    """완전 정확한 예측 (accuracy=1.0, mae=0.0) / Perfect predictions (accuracy=1.0, mae=0.0)"""
    labels = list(range(6)) * 4
    return {"y_true": labels, "y_pred": labels, "confidences": [1.0] * len(labels)}
```

**Fixture 원칙 / Fixture Principles:**
- 실제 파일 I/O가 필요한 Fixture는 `tmp_path` (pytest 내장) 사용 / Use `tmp_path` (pytest built-in) for fixtures requiring real file I/O
- 모델 인스턴스 Fixture는 `device="cpu"` 고정 (MPS/CUDA 환경 의존 제거) / Fix model instance fixtures to `device="cpu"` (remove MPS/CUDA environment dependency)
- 랜덤 값은 `torch.manual_seed(42)` 후 생성 / Generate random values after `torch.manual_seed(42)`

---

## 6. 테스트 작성 규칙 / Test Writing Rules

### 6.1 AAA 패턴 (Arrange-Act-Assert)

```python
def test_compute_metrics_zero_mae_on_perfect_predictions():
    # Arrange
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    confs  = [1.0] * 6

    # Act
    result = compute_metrics(y_true, y_pred, confs)

    # Assert
    assert result["mae"] == 0.0
```

### 6.2 테스트 1개 = 검증 1개 / One Test = One Assertion

```python
# ❌ 잘못된 예 — 하나의 테스트에 여러 검증 / Bad example — multiple assertions in one test
def test_compute_metrics():
    result = compute_metrics(...)
    assert result["accuracy"] == 1.0
    assert result["mae"] == 0.0
    assert result["f1"] == 1.0

# ✅ 올바른 예 — 각각 분리 / Good example — each separated
def test_compute_metrics_perfect_accuracy():
    assert compute_metrics(...)["accuracy"] == 1.0

def test_compute_metrics_zero_mae_on_perfect_predictions():
    assert compute_metrics(...)["mae"] == 0.0
```

### 6.3 엣지케이스 필수 포함 / Edge Cases are Mandatory

모든 함수에 대해 아래 케이스를 고려 / Consider the following cases for every function:
- 빈 입력 / Empty input (`[]`, `{}`, `None`)
- 최소 입력 (원소 1개) / Minimum input (single element)
- 경계값 (0.0, 1.0, NUM_LEVELS-1) / Boundary values (0.0, 1.0, NUM_LEVELS-1)
- 잘못된 타입 → 예외 발생 확인 / Wrong type → verify exception is raised

### 6.4 외부 의존성 격리 / External Dependency Isolation

- 실제 파일 읽기 → `tmp_path` Fixture + 임시 파일 생성 / Real file reads → use `tmp_path` fixture + create temporary files
- 모델 forward → `cpu` device, 작은 dummy tensor / Model forward → `cpu` device, small dummy tensor
- `load_config()` 단위 테스트 → `tmp_path`에 최소 JSON 파일 생성 / `load_config()` unit test → create minimal JSON file in `tmp_path`

### 6.5 금지 사항 / Prohibited

| 금지 / Prohibited | 이유 / Reason |
|------|------|
| 단위 테스트에서 실제 `data_set/` 경로 접근 / Accessing real `data_set/` path in unit tests | CI 환경에서 실패 / Fails in CI environments |
| 단위 테스트에서 `GrayspotModel` 실제 학습 / Running real `GrayspotModel` training in unit tests | 속도 문제 / Speed issues |
| `assert result == 0.12345678` (과도한 정밀도 / excessive precision) | 부동소수점 불안정 → `pytest.approx` 사용 / Float instability → use `pytest.approx` |
| 테스트 간 전역 상태 공유 / Sharing global state between tests | Fixture로 격리할 것 / Isolate with fixtures |

---

## 7. 실행 방법 / How to Run

```bash
# 단위 테스트만 (빠름, < 30초) / Unit tests only (fast, < 30s)
pytest src/tests/unit/ -v

# 통합 테스트 / Integration tests
pytest src/tests/integration/ -v

# 전체 테스트 (Smoke 제외) / All tests (excluding Smoke)
pytest src/tests/unit/ src/tests/integration/ -v

# 특정 모듈만 / Specific module only
pytest src/tests/unit/test_metrics.py -v

# 커버리지 포함 / With coverage
pytest src/tests/unit/ --cov=src --cov-report=term-missing

# 기존 Smoke 테스트 (데이터 필요) / Existing smoke tests (requires data)
python src/tests/test_training_phase0.py --channel Y
python src/tests/test_training_phase2.py --channel Y
```

**pytest.ini / pyproject.toml 설정 (권장):**
```ini
[tool:pytest]
testpaths = src/tests/unit src/tests/integration
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
```

---

## 8. 제외 범위 / Out of Scope

아래 항목은 TDD 단위 테스트 대상이 아니다:
The following items are not subject to TDD unit testing:

| 항목 / Item | 이유 / Reason | 대안 / Alternative |
|------|------|------|
| Phase 0/2 전체 학습 루프 / Full Phase 0/2 training loop | 시간 및 데이터 의존 / Time and data dependent | 기존 Smoke 테스트 유지 / Keep existing smoke tests |
| HTML 리포트 시각적 검증 / Visual verification of HTML reports | 렌더링 환경 의존 / Rendering environment dependent | 수동 확인 / Manual verification |
| Optuna 전체 실행 / Full Optuna run | 시간 의존 / Time dependent | `n_trials=1` Smoke 테스트 / Smoke test |
| MPS/CUDA 디바이스 동작 / MPS/CUDA device behavior | 하드웨어 의존 / Hardware dependent | 로컬 수동 확인 / Local manual verification |
| `GrayspotPredictor` 채널별 캐싱 / Per-channel caching | 모델 파일 의존 / Model file dependent | Integration 테스트 / Integration tests |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|------|------|
| [SSOT_Core.md](SSOT_Core.md) | 코딩 컨벤션 §5 — snake_case, SRP, 명시적 명명 / Coding conventions §5 — snake_case, SRP, explicit naming |
| [Contract.md](Contract.md) | 각 모듈 경계 계약 — 테스트 입출력 spec 참조 / Module boundary contracts — reference for test I/O specs |
| [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) | 예외 코드 — 예외 테스트 작성 시 참조 / Exception codes — reference when writing exception tests |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | 평가 지표 정의 — metrics 테스트 기준값 / Evaluation metric definitions — reference values for metrics tests |
