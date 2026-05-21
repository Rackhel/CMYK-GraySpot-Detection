---
type: bdd
domain: gui
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_GUI.md"
  - "Contract_gui.md"
---

# [BDD] PyQt6 GUI / PyQt6 Graphical User Interface

> **역할 / Role**: PyQt6 6탭 GUI 애플리케이션의 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior of the PyQt6 6-tab GUI application.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **운영자 / Operator** | GUI를 통해 학습·평가·추론을 실행하는 엔지니어 / Engineer running training, evaluation, and inference through the GUI |
| **데이터 과학자 / Data Scientist** | Embedding 탭에서 라벨을 검토하는 연구자 / Researcher reviewing labels in the Embedding tab |
| **시스템 / System** | QThread Worker를 통해 백그라운드 작업을 실행하는 GUI / GUI running background tasks through QThread Workers |

---

## Feature: GUI 워커 격리 / GUI Worker Isolation

---

### Scenario G.1 — 워커 신호 인터페이스 존재 / Worker Signal Interface Exists

```gherkin
Feature: GUI 워커 격리 / GUI worker isolation

  Scenario: 모든 Worker가 required 신호를 가진다
  Scenario: All Workers have required signals

    Given PyQt6 환경이 사용 가능하다

    When  TrainingWorker, EvaluationWorker, TuningWorker, EmbeddingWorker가 초기화된다

    Then  각 워커가 started, finished, error, progress 신호를 가진다
    And   신호가 Qt Signal 타입이다
```

---

### Scenario G.2 — 워커 UI 직접 접근 금지 / Worker UI Direct Access Prohibited

```gherkin
  Scenario: Worker가 QWidget을 직접 수정하지 않는다
  Scenario: Worker does not directly modify QWidget

    Given TrainingWorker가 실행 중이다

    When  학습이 진행된다

    Then  Worker 내부에서 QLabel, QProgressBar 등 UI 요소에 직접 접근하지 않는다
    And   모든 UI 업데이트는 progress 신호를 통해서만 전달된다
```

---

### Scenario G.3 — 탭 인터페이스 준수 / Tab Interface Compliance

```gherkin
  Scenario: 모든 Tab이 refresh()와 on_worker_finished() 메서드를 구현한다
  Scenario: All Tabs implement refresh() and on_worker_finished() methods

    Given PyQt6 환경이 사용 가능하다

    When  DataTab, TrainingTab, EvaluationTab, SettingsTab, TuningTab, EmbeddingTab이 초기화된다

    Then  각 탭이 refresh() 메서드를 가진다
    And   각 탭이 on_worker_finished() 메서드를 가진다
```

---

### Scenario G.4 — 학습 탭 진행 표시 / Training Tab Progress Display

```gherkin
  Scenario: 학습 탭에서 진행률이 실시간으로 표시된다
  Scenario: Training tab displays progress in real-time

    Given 사용자가 Training 탭에서 채널과 Phase를 선택한다
    And   학습 시작 버튼을 클릭한다

    When  TrainingWorker가 실행된다

    Then  진행률 바가 epoch 진행에 따라 업데이트된다
    And   로그 텍스트 영역에 학습 로그가 출력된다
    And   학습 중 UI가 응답 불능 상태가 되지 않는다
```

---

### Scenario G.5 — 임베딩 탭 라벨 저장 / Embedding Tab Label Save

```gherkin
  Scenario: Embedding 탭에서 라벨 수정 후 저장하면 버전 증가된 CSV가 생성된다
  Scenario: Saving after label modification in Embedding tab creates version-incremented CSV

    Given Embedding 탭이 표시되고 labels_v0.csv가 로드된 상태이다
    And   사용자가 일부 샘플의 라벨을 수정한다

    When  저장 버튼을 클릭한다

    Then  labels_v1.csv가 생성된다
    And   원본 labels_v0.csv가 보존된다
    And   저장 완료 알림이 표시된다
```

---

### Scenario G.6 — 앱 종료 시 워커 정리 / Worker Cleanup on App Exit

```gherkin
  Scenario: 앱 종료 시 실행 중인 Worker가 안전하게 종료된다
  Scenario: Running Workers are safely terminated when app exits

    Given Worker가 백그라운드에서 실행 중이다

    When  사용자가 앱 창을 닫는다

    Then  Worker의 QThread가 join된다
    And   강제 종료(kill)가 발생하지 않는다
    And   미완성 체크포인트가 저장되지 않는다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| G.1 — 워커 신호 / worker signals | `test_gui_workers.py` | `test_worker_has_required_signals` | Unit |
| G.2 — UI 직접 접근 금지 / no direct UI | `test_gui_workers.py` | `test_worker_does_not_touch_ui_directly` | Unit |
| G.3 — 탭 인터페이스 / tab interface | `test_gui_tabs.py` | `test_tab_has_refresh_method` | Unit |
| G.4 — 진행률 표시 / progress display | `test_gui_tabs.py` | `test_training_tab_progress_update` | Unit |
| G.5 — 라벨 저장 / label save | `test_gui_tabs.py` | `test_embedding_tab_save_labels_increments_version` | Unit |
| G.6 — 워커 정리 / worker cleanup | `test_gui_workers.py` | `test_worker_thread_joins_on_quit` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_gui.md](../Contract/Contract_gui.md) | GUI 모듈 경계 계약 / GUI module boundary contract |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | PyQt6 프레임워크 정의 / PyQt6 framework definition |
| [TDD_GUI.md](../TDD/TDD_GUI.md) | GUI TDD 명세 / GUI TDD specification |
