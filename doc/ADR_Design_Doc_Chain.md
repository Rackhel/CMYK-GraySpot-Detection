# ADR — 설계 문서 연쇄 체계 채택
# ADR — Adoption of Design Document Chain

---

## 1. 배경 / Background

CMYK Grayspot Detection System 개발 초기에는 코드 구현이 설계 문서보다 먼저 진행되었다. 이후 SSOT 문서들이 사후적으로 작성되었으나, 아래 문제들이 확인되었다.

In the early phase of CMYK Grayspot Detection System development, code implementation preceded design documentation. SSOT documents were subsequently written retroactively, but the following problems were identified.

| 문제 / Problem | 증상 / Symptom |
|---|---|
| 테스트의 의미 근거 부재 / No semantic basis for tests | TDD 테스트가 "어떤 동작을 검증하는가"를 액터 관점에서 기술하지 못함 / TDD tests could not describe "what behavior is being verified" from an actor perspective |
| SSOT ↔ 테스트 간 추적성 단절 / Traceability gap between SSOT and tests | SSOT가 "무엇을 정의하는가"와 테스트가 "무엇을 검증하는가"의 연결고리가 없음 / No link between what SSOT defines and what tests verify |
| 구현 중심 테스트 / Implementation-centric tests | 테스트가 액터 시나리오 없이 내부 API 형상 검증에 집중 / Tests focused on internal API shape without actor scenarios |
| 설계 의도 추론 불가 / Design intent not recoverable | 코드만으로 "왜 이 동작인가"를 재구성할 수 없음 / Code alone could not reconstruct "why this behavior" |

이에 따라 **PRD → SSOT/Contract → BDD → TDD → Code** 의 설계 문서 연쇄 체계를 공식화하였다.

This prompted the formalization of the **PRD → SSOT/Contract → BDD → TDD → Code** design document chain.

---

## 2. 결정 사항 / Decisions

### 2.1 설계 문서 연쇄 계층 공식화
### 2.1 Formalization of the Design Document Chain

**결정 / Decision**: 다음 계층 순서를 프로젝트의 공식 설계 명세 체계로 채택한다.

**Decision**: Adopt the following layer ordering as the project's official design specification framework.

```
PRD
 ↓  요구사항 정의 / Requirements definition
SSOT  (SSOT_*.md)
 ↓  의미 정의 + SOLID 원칙 준수 / Semantic definitions + SOLID compliance
Contract  (Contract.md)
 ↓  모듈 간 공개 API 계약 / Public API contracts between modules
BDD  (BDD.md)
 ↓  액터 관점 동작 명세 / Behavior specification from actor perspective
TDD  (TDD.md)
 ↓  단위·통합 테스트 전략 / Unit & integration test strategy
Code  (src/)
```

### 2.2 각 계층의 책임 분리
### 2.2 Separation of Responsibilities per Layer

| 계층 / Layer | 문서 / Document | 질문 / Answers | 작성 단위 / Written in |
|---|---|---|---|
| PRD | `doc/PRD.md` | 무엇을 만드는가 / What to build | 제품 요구사항 / Product requirements |
| SSOT | `doc/SSOT_*.md` | 용어·파라미터·원칙의 유일한 정의 / Sole definition of terms, parameters, principles | 의미 단위 (Semantic units) |
| Contract | `doc/Contract.md` | 모듈 간 공개 인터페이스 계약 / Public interface contracts between modules | 함수 시그니처·입출력 타입 |
| BDD | `doc/BDD.md` | 액터가 무엇을 경험하는가 / What actors experience | Gherkin (Given–When–Then) |
| TDD | `doc/TDD.md` | 코드 단위가 무엇을 보장하는가 / What code units guarantee | 테스트 케이스 목록 |
| Code | `src/` | 어떻게 구현하는가 / How to implement | Python |

---

## 3. 선택 이유 / Rationale

### 3.1 SSOT/Contract를 BDD 앞에 두는 이유 / Why SSOT/Contract Before BDD

BDD 시나리오는 "채널별 정확도 ≥ 85%", "NUM_LEVELS = 6" 같은 의미 단위를 참조한다. 이 값들이 SSOT에 먼저 정의되지 않으면 시나리오가 각 문서마다 다른 값을 기술할 위험이 있다.

BDD scenarios reference semantic units such as "per-channel accuracy ≥ 85%" and "NUM_LEVELS = 6". Without SSOT defining these first, scenarios risk inconsistent values across documents.

Contract는 BDD 시나리오가 어떤 모듈 경계에서 시작하고 끝나는지를 고정한다. 계약이 없으면 시나리오의 "When Evaluator가 추론을 실행한다"는 어떤 함수 호출인지 특정할 수 없다.

Contract fixes which module boundary a BDD scenario begins and ends at. Without it, "When Evaluator runs inference" in a scenario cannot be tied to a specific function call.

### 3.2 BDD를 TDD 앞에 두는 이유 / Why BDD Before TDD

| 항목 / Item | BDD 없이 TDD만 작성 시 / TDD without BDD | BDD 후 TDD 작성 시 / TDD after BDD |
|---|---|---|
| 테스트 목적 / Test purpose | "이 함수가 (4, 6) shape을 반환하는가" | "Operator가 Phase 2를 실행하면 채널별 분류 결과를 얻는가"의 구현 검증 |
| 커버리지 기준 / Coverage basis | 구현 코드 라인 | BDD 시나리오 추적성 매트릭스 |
| 리그레션 발견 / Regression detection | API 변경 시 실패 | 동작 변경 시 실패 |
| 설계 의도 복원 / Intent recovery | 불가 / Not possible | BDD 시나리오 참조로 복원 가능 |

### 3.3 SSOT/Contract를 사후 작성하지 않는 이유 / Why Not Write SSOT/Contract After the Fact

이 프로젝트의 초기 경험에서 사후 문서화는 다음 문제를 야기했다.

The project's early experience with retroactive documentation produced the following issues.

- SSOT-NM01: 평가 전처리에서 ImageNet 정규화가 적용되지 않아 학습과 불일치 발생 → 코드 분석 후 사후 발견
- SSOT-CS01: 평가 시 색상 공간 불일치 위험이 문서화되지 않아 잠재 버그로 남아있었음
- evaluator.py가 950줄 모놀리식 구조로 성장 → SRP/ISP 위반이 SSOT 사전 원칙 없이는 조기 발견 불가

- SSOT-NM01: ImageNet normalization missing in eval preprocessing caused training/eval mismatch — discovered retroactively after code analysis.
- SSOT-CS01: Color space mismatch risk was undocumented and remained a latent bug.
- evaluator.py grew to 950 lines monolithic — SRP/ISP violations cannot be caught early without prior SSOT principles.

---

## 4. 고려한 대안 / Considered Alternatives

### 4.1 Code → Doc (사후 문서화) / Retroactive Documentation

**채택하지 않은 이유 / Why Not Chosen**: 이미 시도했으며 §1의 문제들을 야기했다. 코드가 설계 원칙 없이 성장하면 SSOT 위반이 누적된 이후에야 발견된다.

**Why Not Chosen**: Already attempted — produced the problems described in §1. When code grows without design principles, SSOT violations only surface after they accumulate.

### 4.2 PRD → TDD (BDD 생략) / Skip BDD

**채택하지 않은 이유 / Why Not Chosen**: TDD는 구현 단위(함수, 클래스)를 검증하지만, "QC Inspector가 잘못 분류된 이미지를 확인하는 시나리오"와 같은 액터 관점의 동작을 표현할 수 없다. BDD 없이는 테스트 추적성 매트릭스를 PRD 요구사항까지 연결할 수 없다.

**Why Not Chosen**: TDD validates implementation units (functions, classes) but cannot express actor-perspective behavior such as "the QC Inspector reviews misclassified images." Without BDD, the test traceability matrix cannot be linked up to PRD requirements.

### 4.3 PRD → BDD → TDD (SSOT/Contract 생략) / Skip SSOT/Contract

**채택하지 않은 이유 / Why Not Chosen**: BDD 시나리오가 임계값·파라미터를 자체 정의하게 되어 "multiple source of truth"가 발생한다. 또한 BDD가 어떤 모듈 API를 호출하는지 계약 없이는 특정할 수 없어 시나리오가 구현에 종속된다.

**Why Not Chosen**: BDD scenarios would define thresholds and parameters themselves, creating multiple sources of truth. Additionally, without Contract, scenarios cannot specify which module API is invoked, making them implementation-dependent.

### 4.4 PRD → SSOT → Contract → TDD → BDD (BDD를 사후에) / BDD After TDD

**채택하지 않은 이유 / Why Not Chosen**: BDD가 TDD 이후에 작성되면 구현 중심 테스트를 사후에 시나리오로 포장하는 데 그친다. BDD의 가치는 "테스트를 무엇으로 채울 것인가"를 TDD 이전에 결정하는 데 있다.

**Why Not Chosen**: Writing BDD after TDD reduces it to retrofitting actor-language wrappers around already implementation-centric tests. BDD's value lies in deciding "what behavior to cover" before TDD specifies how to test it.

---

## 5. 리팩토링 적용 사례 / Refactoring Application Example

이 설계 연쇄 체계가 실제로 적용된 사례로, evaluator.py SRP/ISP 리팩토링을 들 수 있다.

The evaluator.py SRP/ISP refactoring demonstrates this chain applied in practice.

```
SSOT_Core.md §5.1 (SRP 위반 식별)
 ↓
Contract.md §7 (Evaluator 인터페이스 계약 선 동결)
 ↓
evaluator.py 4 Mixin + Orchestrator 분리
  evaluator_inference.py  (InferenceMixin)
  evaluator_metrics.py    (MetricsMixin)
  evaluator_export.py     (ExportMixin)
  evaluator_charts.py     (ChartsMixin)
  evaluator.py            (Orchestrator)
 ↓
BDD.md Feature 5: 평가 결과 저장 시나리오
 ↓
TDD.md TestEvaluator 테스트 전략
```

Contract에서 `from evaluation.evaluator import Evaluator` 를 선 동결했기 때문에, 내부 4 Mixin 분리 이후에도 외부 API 변경 없이 리팩토링을 완료할 수 있었다.

Because Contract froze `from evaluation.evaluator import Evaluator` first, the internal split into 4 Mixins was completed without changing the external API.

---

## 6. 결과 및 트레이드오프 / Consequences and Trade-offs

### 이점 / Benefits

- 코드 변경 전 설계 의도가 3개 계층(SSOT, BDD, TDD)에 기록되어 추적 가능
- BDD 시나리오 → TDD 추적성 매트릭스로 테스트 커버리지 누락을 사전에 발견 가능
- Contract 선 동결로 리팩토링 시 외부 API 안정성 보장
- SSOT 위반(SSOT-NM01, SSOT-CS01 등)이 코드 작성 전 원칙 단계에서 방지됨

- Design intent is recorded across 3 layers (SSOT, BDD, TDD) before code changes, enabling traceability.
- BDD→TDD traceability matrix allows early detection of test coverage gaps.
- Contract freeze ensures external API stability during refactoring.
- SSOT violations (SSOT-NM01, SSOT-CS01, etc.) are prevented at the principles layer before code is written.

### 비용 및 주의사항 / Costs and Cautions

- 코드 변경 시 SSOT → Contract → BDD → TDD → Code 순으로 문서도 함께 갱신해야 함 (문서 부채 발생 위험)
- 소규모 기능 변경에도 체계를 유지하는 규율이 필요
- 설계 연쇄가 길수록 초기 진입 비용이 높아짐 — 신규 기여자는 전체 체계를 이해해야 함

- Code changes require updating documents in SSOT → Contract → BDD → TDD → Code order (risk of documentation debt).
- Discipline is required to maintain the chain even for small feature changes.
- The longer the design chain, the higher the onboarding cost — new contributors must understand the full framework.

---

## 7. 관련 코드 및 문서 / Related Code and Documents

| 참조 / Reference | 내용 / Content |
|---|---|
| [`doc/SSOT_Core.md`](SSOT_Core.md) | SSOT 핵심 원칙, SOLID 설계 원칙, 문서 목록 / Core SSOT principles, SOLID design principles, document index |
| [`doc/Contract.md`](Contract.md) | 모듈 간 공개 API 계약 (Evaluator, Trainer, GrayspotModel 등) / Public API contracts between modules (Evaluator, Trainer, GrayspotModel, etc.) |
| [`doc/BDD.md`](BDD.md) | Gherkin 시나리오 — 7 Features, 24 Scenarios / Gherkin scenarios — 7 Features, 24 Scenarios |
| [`doc/TDD.md`](TDD.md) | 단위·통합·스모크 테스트 전략 및 추적성 매트릭스 / Unit, integration, and smoke test strategy with traceability matrix |
| [`doc/ADR_Model_Select.md`](ADR_Model_Select.md) | 이 체계를 적용한 backbone/head 설계 결정 사례 / Example of this chain applied to backbone/head design decisions |
| [`src/evaluation/evaluator.py`](../src/evaluation/evaluator.py) | 설계 연쇄 적용 결과 — 4 Mixin + Orchestrator / Result of applying the design chain — 4 Mixins + Orchestrator |

---
