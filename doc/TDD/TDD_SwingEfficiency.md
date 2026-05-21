---
type: tdd
domain: swing_efficiency
status: failing
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_Evaluation_Reporting.md"
  - "../Contract/Contract_evaluation_reporting.md"
---

# [TDD] Swing Efficiency — 스윙 효율 지표 / Swing Efficiency Metrics

> **목적 / Purpose**: `swing_efficiency.py` 의 동작을 BDD/TDD로 정의한다.
> **테스트 파일 / Test Files**: `src/tests/unit/test_swing_efficiency.py`
> **상태 / Status**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오 / BDD Scenarios

### Feature: Swing Efficiency 계산 / Swing Efficiency Computation

**Scenario 1: 정상적인 Cycle 1 효율 계산 / Normal Cycle 1 efficiency computation**
```
Given  baseline_acc=0.72, cycle_acc=0.81, n_labels_changed=45, cycle=1 이 주어졌을 때
# Given baseline_acc=0.72, cycle_acc=0.81, n_labels_changed=45, cycle=1
When   compute_swing_efficiency() 를 호출하면
# When compute_swing_efficiency() is called
Then   delta_acc ≈ 0.09 이고
# Then delta_acc ≈ 0.09
And    efficiency_ratio ≈ 0.002 (0.09/45) 이고
# And efficiency_ratio ≈ 0.002 (0.09/45)
And    swing_decision 은 "pass", "retry_phase2", "retry_phase0" 중 하나이다
# And swing_decision is one of "pass", "retry_phase2", "retry_phase0"
```

**Scenario 2: 라벨 수정 없이 성능 향상 없음 / No improvement without label changes**
```
Given  baseline_acc=0.72, cycle_acc=0.72, n_labels_changed=0 이 주어졌을 때
# Given baseline_acc=0.72, cycle_acc=0.72, n_labels_changed=0
When   compute_swing_efficiency() 를 호출하면
# When compute_swing_efficiency() is called
Then   delta_acc == 0.0 이고
# Then delta_acc == 0.0
And    efficiency_ratio == 0.0 이다  (0 / 0 → 0.0 처리, 예외 아님 / not an exception)
# And efficiency_ratio == 0.0 (0 / 0 → 0.0, not an exception)
```

**Scenario 3: 성능 하락 Cycle / Performance regression cycle**
```
Given  baseline_acc=0.81, cycle_acc=0.75, n_labels_changed=20 이 주어졌을 때
# Given baseline_acc=0.81, cycle_acc=0.75, n_labels_changed=20
When   compute_swing_efficiency() 를 호출하면
# When compute_swing_efficiency() is called
Then   delta_acc < 0.0 이고
# Then delta_acc < 0.0
And    swing_decision 은 "retry_phase0" 또는 "retry_phase2" 이다
# And swing_decision is "retry_phase0" or "retry_phase2"
```

---

### Feature: 조기 종료 판단 / Early Stop Decision

**Scenario 4: Cycle 2 효율 50% 미만 → 조기 종료 / Cycle 2 efficiency below 50% → early stop**
```
Given  Cycle 1 efficiency_ratio=0.002, Cycle 2 efficiency_ratio=0.0009 이 주어졌을 때
# Given Cycle 1 efficiency_ratio=0.002, Cycle 2 efficiency_ratio=0.0009
When   should_early_stop(current, previous) 를 호출하면
# When should_early_stop(current, previous) is called
Then   True 를 반환한다
# Then True is returned
```

**Scenario 5: 충분히 높은 효율 → 계속 진행 / High efficiency → continue**
```
Given  Cycle 1 efficiency_ratio=0.002, Cycle 2 efficiency_ratio=0.0015 이 주어졌을 때
# Given Cycle 1 efficiency_ratio=0.002, Cycle 2 efficiency_ratio=0.0015
When   should_early_stop(current, previous) 를 호출하면
# When should_early_stop(current, previous) is called
Then   False 를 반환한다
# Then False is returned
```

**Scenario 6: efficiency_ratio=0 → 항상 조기 종료 / efficiency_ratio=0 → always early stop**
```
Given  previous.efficiency_ratio=0.0 일 때
# Given previous.efficiency_ratio=0.0
When   should_early_stop() 를 호출하면
# When should_early_stop() is called
Then   True 를 반환한다  (분모 0 처리)
# Then True is returned (zero denominator handling)
```

---

## 2. TDD 스펙 / TDD Specifications

### 2.1 compute_swing_efficiency()

| 테스트 ID / Test ID | 입력 / Input | 기댓값 / Expected |
| --- | --- | --- |
| T-SE-01 | baseline=0.72, cycle=0.81, n=45 | `delta_acc≈0.09`, `efficiency_ratio≈0.002` |
| T-SE-02 | baseline=0.72, cycle=0.72, n=0 | `delta_acc==0.0`, `efficiency_ratio==0.0` |
| T-SE-03 | baseline=0.81, cycle=0.75, n=20 | `delta_acc<0`, decision in {"retry_phase0","retry_phase2"} |
| T-SE-04 | baseline=0.50, cycle=0.95, n=10 | `delta_acc==0.45`, `efficiency_ratio==0.045` |
| T-SE-05 | cycle=1 | `report.cycle == 1` |
| T-SE-06 | 반환 타입 | `isinstance(report, SwingEfficiencyReport)` |

```python
def test_compute_swing_efficiency_normal():
    cfg = {"evaluation": {"swing_thresholds": {"overall_accuracy": 0.90}}}
    report = compute_swing_efficiency(
        baseline_acc=0.72,
        cycle_acc=0.81,
        n_labels_changed=45,
        cycle=1,
        cfg=cfg,
    )
    assert abs(report.delta_acc - 0.09) < 1e-6
    assert abs(report.efficiency_ratio - (0.09 / 45)) < 1e-9
    assert report.swing_decision in {"pass", "retry_phase2", "retry_phase0"}

def test_compute_swing_efficiency_zero_labels():
    cfg = {"evaluation": {"swing_thresholds": {"overall_accuracy": 0.90}}}
    report = compute_swing_efficiency(0.72, 0.72, n_labels_changed=0, cycle=1, cfg=cfg)
    assert report.delta_acc == 0.0
    assert report.efficiency_ratio == 0.0  # 0/0 → 0.0, no exception

def test_compute_swing_efficiency_regression():
    cfg = {"evaluation": {"swing_thresholds": {"overall_accuracy": 0.90}}}
    report = compute_swing_efficiency(0.81, 0.75, n_labels_changed=20, cycle=2, cfg=cfg)
    assert report.delta_acc < 0
    assert report.swing_decision in {"retry_phase0", "retry_phase2"}
```

### 2.2 should_early_stop()

| 테스트 ID / Test ID | 입력 / Input | 기댓값 / Expected |
| --- | --- | --- |
| T-SE-10 | current.eff=0.0009, prev.eff=0.002 | `True` (0.0009 < 0.002*0.5=0.001) |
| T-SE-11 | current.eff=0.0015, prev.eff=0.002 | `False` (0.0015 >= 0.001) |
| T-SE-12 | current.eff=0.001, prev.eff=0.002 | `False` (경계: 0.001 == 0.001, 경계값 포함 / boundary: inclusive) |
| T-SE-13 | prev.eff=0.0 | `True` (previous가 0이면 항상 종료 / always stop if previous is 0) |

```python
def test_early_stop_true():
    prev = make_report(efficiency_ratio=0.002)
    curr = make_report(efficiency_ratio=0.0009)
    assert should_early_stop(curr, prev) is True

def test_early_stop_false():
    prev = make_report(efficiency_ratio=0.002)
    curr = make_report(efficiency_ratio=0.0015)
    assert should_early_stop(curr, prev) is False

def test_early_stop_zero_previous():
    prev = make_report(efficiency_ratio=0.0)
    curr = make_report(efficiency_ratio=0.001)
    assert should_early_stop(curr, prev) is True
```

### 2.3 SwingEfficiencyReport dataclass

| 테스트 ID / Test ID | 검증 포인트 / Verification |
| --- | --- |
| T-SE-20 | `report.cycle` (int) |
| T-SE-21 | `report.baseline_acc`, `report.cycle_acc` (float) |
| T-SE-22 | `report.delta_acc == cycle_acc - baseline_acc` |
| T-SE-23 | `report.swing_decision in {"pass", "retry_phase2", "retry_phase0"}` |

---

## 3. Conftest 픽스처 / Conftest Fixtures

```python
# src/tests/unit/conftest.py 에 추가 / add to conftest.py

from evaluation.swing_efficiency import SwingEfficiencyReport

def make_report(efficiency_ratio: float, swing_decision: str = "pass") -> SwingEfficiencyReport:
    return SwingEfficiencyReport(
        cycle=1,
        baseline_acc=0.72,
        cycle_acc=0.81,
        delta_acc=0.09,
        n_labels_changed=45,
        efficiency_ratio=efficiency_ratio,
        swing_decision=swing_decision,
    )
```

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | Swing Efficiency 정의 / Swing Efficiency definition |
| [Contract_evaluation_reporting.md](../Contract/Contract_evaluation_reporting.md) | API 계약 / API contract |
