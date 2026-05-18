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

# [TDD] Swing Efficiency — 스윙 효율 지표

> **목적**: `swing_efficiency.py` 의 동작을 BDD/TDD로 정의한다.
> **테스트 파일**: `src/tests/unit/test_swing_efficiency.py`
> **상태**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오

### Feature: Swing Efficiency 계산

**Scenario 1: 정상적인 Cycle 1 효율 계산**
```
Given  baseline_acc=0.72, cycle_acc=0.81, n_labels_changed=45, cycle=1 이 주어졌을 때
When   compute_swing_efficiency() 를 호출하면
Then   delta_acc ≈ 0.09 이고
And    efficiency_ratio ≈ 0.002 (0.09/45) 이고
And    swing_decision 은 "pass", "retry_phase2", "retry_phase0" 중 하나이다
```

**Scenario 2: 라벨 수정 없이 성능 향상 없음**
```
Given  baseline_acc=0.72, cycle_acc=0.72, n_labels_changed=0 이 주어졌을 때
When   compute_swing_efficiency() 를 호출하면
Then   delta_acc == 0.0 이고
And    efficiency_ratio == 0.0 이다  (0 / 0 → 0.0 처리, 예외 아님)
```

**Scenario 3: 성능 하락 Cycle**
```
Given  baseline_acc=0.81, cycle_acc=0.75, n_labels_changed=20 이 주어졌을 때
When   compute_swing_efficiency() 를 호출하면
Then   delta_acc < 0.0 이고
And    swing_decision 은 "retry_phase0" 또는 "retry_phase2" 이다
```

---

### Feature: 조기 종료 판단

**Scenario 4: Cycle 2 효율이 Cycle 1의 50% 미만 → 조기 종료**
```
Given  Cycle 1 efficiency_ratio=0.002, Cycle 2 efficiency_ratio=0.0009 이 주어졌을 때
When   should_early_stop(current, previous) 를 호출하면
Then   True 를 반환한다
```

**Scenario 5: Cycle 2 효율이 충분히 높음 → 계속 진행**
```
Given  Cycle 1 efficiency_ratio=0.002, Cycle 2 efficiency_ratio=0.0015 이 주어졌을 때
When   should_early_stop(current, previous) 를 호출하면
Then   False 를 반환한다
```

**Scenario 6: Cycle 1 efficiency_ratio=0 인 경우 → 항상 조기 종료**
```
Given  previous.efficiency_ratio=0.0 일 때
When   should_early_stop() 를 호출하면
Then   True 를 반환한다  (분모 0 처리)
```

---

## 2. TDD 스펙

### 2.1 compute_swing_efficiency()

| 테스트 ID | 입력 | 기댓값 |
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

| 테스트 ID | 입력 | 기댓값 |
| --- | --- | --- |
| T-SE-10 | current.eff=0.0009, prev.eff=0.002 | `True` (0.0009 < 0.002*0.5=0.001) |
| T-SE-11 | current.eff=0.0015, prev.eff=0.002 | `False` (0.0015 >= 0.001) |
| T-SE-12 | current.eff=0.001, prev.eff=0.002 | `False` (경계: 0.001 == 0.001, 경계값 포함) |
| T-SE-13 | prev.eff=0.0 | `True` (previous가 0이면 항상 종료) |

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

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-SE-20 | `report.cycle` (int) |
| T-SE-21 | `report.baseline_acc`, `report.cycle_acc` (float) |
| T-SE-22 | `report.delta_acc == cycle_acc - baseline_acc` |
| T-SE-23 | `report.swing_decision in {"pass", "retry_phase2", "retry_phase0"}` |

---

## 3. Conftest 픽스처

```python
# src/tests/unit/conftest.py 에 추가

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

| 문서 | 관계 |
| --- | --- |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | Swing Efficiency 정의 |
| [Contract_evaluation_reporting.md](../Contract/Contract_evaluation_reporting.md) | API 계약 |
