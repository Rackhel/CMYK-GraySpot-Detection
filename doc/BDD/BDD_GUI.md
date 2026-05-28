---
type: bdd
domain: gui
status: Active
last_updated: 2026-05-28
owner: CMYK WooSong Team
related_docs:
  - "SSOT_GUI.md"
  - "Contract_gui.md"
---

# [BDD] PyQt6 GUI / PyQt6 Graphical User Interface

> **역할 / Role**: PyQt6 7탭 GUI 애플리케이션의 관찰 가능한 행동을 정의한다.

---

## 행위자 / Actors

| 행위자 | 역할 |
|---|---|
| **운영자** | GUI를 통해 학습·평가·추론을 실행하는 엔지니어 |
| **데이터 과학자** | Embedding 탭에서 라벨을 검토하는 연구자 |
| **시스템** | QThread Worker를 통해 백그라운드 작업을 실행하는 GUI |

---

## Feature: GUI 워커 격리 / GUI Worker Isolation

### Scenario G.1 — 워커 신호 인터페이스 존재

```gherkin
  Scenario: 모든 Worker가 required 신호를 가진다

    Given PyQt6 환경이 사용 가능하다

    When  TrainingWorker, EvaluationWorker, TuningWorker, EmbeddingWorker,
          InferenceWorker, BatchInferenceWorker, GradCAMWorker가 초기화된다

    Then  각 워커가 progress_updated, log_emitted, finished, error_occurred 신호를 가진다
    And   신호가 Qt pyqtSignal 타입이다
```

### Scenario G.2 — 워커 UI 직접 접근 금지

```gherkin
  Scenario: Worker가 QWidget을 직접 수정하지 않는다

    Given TrainingWorker가 실행 중이다

    When  학습이 진행된다

    Then  Worker 내부에서 QLabel, QProgressBar 등 UI 요소에 직접 접근하지 않는다
    And   모든 UI 업데이트는 progress_updated / log_emitted 신호를 통해서만 전달된다
```

### Scenario G.3 — 탭 인터페이스 준수

```gherkin
  Scenario: 모든 Tab이 refresh()와 on_worker_finished() 메서드를 구현한다

    Given PyQt6 환경이 사용 가능하다

    When  DataTab, TrainingTab, EvaluationTab, OptunaTab,
          EmbeddingTab, InferenceTab, SettingsTab이 초기화된다

    Then  각 탭이 refresh() 메서드를 가진다
    And   각 탭이 on_worker_finished() 메서드를 가진다
```

### Scenario G.4 — 학습 탭 진행 표시

```gherkin
  Scenario: 학습 탭에서 진행률이 실시간으로 표시된다

    Given 사용자가 Training 탭에서 채널과 Phase를 선택한다
    And   학습 시작 버튼을 클릭한다

    When  TrainingWorker가 실행된다

    Then  진행률 바가 epoch 진행에 따라 업데이트된다
    And   로그 텍스트 영역에 학습 로그가 출력된다
    And   학습 완료 후 학습 곡선 차트(TrainingChart)가 표시된다
    And   학습 중 UI가 응답 불능 상태가 되지 않는다
```

### Scenario G.5 — 임베딩 탭 라벨 저장

```gherkin
  Scenario: Embedding 탭에서 라벨 수정 후 저장하면 버전 증가된 CSV가 생성된다

    Given Embedding 탭이 표시되고 사용자가 샘플의 레벨을 수정한다

    When  저장 버튼을 클릭한다

    Then  labels_v{N+1}.csv가 생성된다
    And   원본 파일이 보존된다
    And   저장 완료 메시지가 LogPanel에 표시된다
```

### Scenario G.6 — 앱 종료 시 워커 정리

```gherkin
  Scenario: 앱 종료 시 실행 중인 Worker가 안전하게 종료된다

    Given Worker가 백그라운드에서 실행 중이다

    When  사용자가 앱 창을 닫는다

    Then  Worker의 QThread가 join된다
    And   강제 종료(kill)가 발생하지 않는다
```

### Scenario G.7 — 단일 이미지 추론

```gherkin
  Scenario: 추론 탭에서 이미지와 체크포인트를 선택하면 예측 레벨이 표시된다

    Given 사용자가 Inference 탭을 연다
    And   체크포인트 파일을 선택한다
    And   이미지 파일을 선택한다

    When  추론 실행 버튼을 클릭한다

    Then  예측 레벨 배지가 표시된다
    And   신뢰도(%) 값이 표시된다
    And   Top-3 레벨 목록이 표시된다
```

### Scenario G.8 — 배치 폴더 추론

```gherkin
  Scenario: 추론 탭에서 폴더를 선택하면 모든 이미지가 일괄 추론된다

    Given 사용자가 Inference 탭을 연다
    And   이미지 폴더를 선택한다

    When  배치 추론 버튼을 클릭한다

    Then  각 이미지가 처리될 때마다 결과 테이블에 행이 추가된다
    And   완료 후 LevelAccuracyTable이 레벨별 정확도로 업데이트된다
    And   CSV 내보내기 버튼이 활성화된다
```

### Scenario G.9 — 전처리 파라미터 편집 및 저장

```gherkin
  Scenario: Data 탭에서 전처리 파라미터를 수정하고 저장하면 config.json에 반영된다

    Given 사용자가 Data 탭을 연다
    And   image_size를 128에서 256으로 변경한다

    When  파라미터 저장 버튼을 클릭한다

    Then  src/config/config.json의 data.image_size가 256으로 저장된다
    And   LogPanel에 저장 완료 메시지가 표시된다
    And   테이블이 재스캔된다
```

### Scenario G.10 — GradCAM 시각화

```gherkin
  Scenario: 단일 이미지 추론 완료 후 GradCAM 버튼을 누르면 히트맵이 표시된다

    Given 사용자가 Inference 탭에서 단일 채널로 이미지 추론을 완료했다

    When  GradCAM 실행 버튼을 클릭한다

    Then  GradCAMWorker가 백그라운드에서 실행된다
    And   완료 후 히트맵 오버레이 이미지가 GradCAM 패널에 표시된다
    And   예측 레벨과 신뢰도가 로그에 출력된다
    And   앙상블(all) 모드에서는 GradCAM 버튼이 비활성화된다
```

### Scenario G.11 — 전체 채널 평가

```gherkin
  Scenario: Evaluation 탭에서 "전체(All)" 선택 시 4채널이 순차 평가된다

    Given 사용자가 Evaluation 탭에서 채널을 "전체(All)"로 선택한다

    When  평가 실행 버튼을 클릭한다

    Then  Y, M, C, K 채널이 순차적으로 평가된다
    And   각 채널 완료 시 채널별 비교 테이블이 업데이트된다
    And   모든 채널 완료 후 전체 평균 Acc/F1/MAE가 LogPanel에 출력된다
```

### Scenario G.12 — Optuna Before/After 비교

```gherkin
  Scenario: Optuna HPO 탭에서 스냅샷 저장 후 HPO 실행 시 Before/After 비교가 표시된다

    Given 사용자가 Optuna 탭에서 현재 설정 스냅샷 버튼을 클릭한다
    And   탐색 공간을 수정하고 HPO를 실행한다

    When  HPO가 완료된다

    Then  Before 패널에 스냅샷 시점의 파라미터가 표시된다
    And   After 패널에 HPO 최적 파라미터가 표시된다
    And   Plotly 막대 비교 차트가 렌더링된다
```

### Scenario G.13 — Embedding 다채널 비교

```gherkin
  Scenario: Embedding 탭에서 "전체 비교(All)" 선택 시 4채널이 같은 차트에 표시된다

    Given 사용자가 Embedding 탭에서 채널을 "전체 비교(All)"로 선택한다

    When  임베딩 추출 버튼을 클릭한다

    Then  Y, M, C, K 채널이 순차적으로 추출된다
    And   같은 t-SNE 산점도에 채널별 색상으로 오버레이된다
    And   레벨 순도 탭의 intra-cluster distance 차트가 업데이트된다
```

### Scenario G.14 — Worker Settings 저장

```gherkin
  Scenario: Settings 탭에서 Worker 설정을 변경하고 저장하면 config.json에 반영된다

    Given 사용자가 Settings 탭에서 device를 "cpu"로 변경한다

    When  설정 저장 버튼을 클릭한다

    Then  src/config/config.json의 system.device가 "cpu"로 저장된다
    And   LogPanel에 저장 완료 메시지가 표시된다
```

### Scenario G.15 — 데이터 소스 선택기

```gherkin
  Scenario: Inference 탭에서 데이터 소스를 "Raw"로 변경하면 파일 다이얼로그 시작 디렉토리가 바뀐다

    Given 사용자가 Inference 탭을 연다
    And   데이터 소스를 "Raw"로 변경한다

    When  이미지 선택 버튼을 클릭한다

    Then  파일 다이얼로그가 data_set/raw 에서 열린다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 | TDD 파일 | 테스트 함수 | 계층 |
|---|---|---|---|
| G.1 — 워커 신호 | `test_gui_workers.py` | `test_all_workers_have_signals` | Unit |
| G.2 — UI 직접 접근 금지 | `test_gui_workers.py` | `test_worker_does_not_touch_ui_directly` | Unit |
| G.3 — 탭 인터페이스 | `test_gui_tabs.py` | `test_all_tabs_implement_interface` | Unit |
| G.4 — 진행률 + 학습 곡선 | `test_gui_tabs.py` | `test_training_tab_progress_update` | Unit |
| G.5 — 라벨 저장 | `test_gui_tabs.py` | `test_embedding_tab_save_labels_increments_version` | Unit |
| G.6 — 워커 정리 | `test_gui_workers.py` | `test_worker_thread_joins_on_quit` | Unit |
| G.7 — 단일 추론 | `test_gui_tabs.py` | `test_inference_tab_single_image_guard` | Unit |
| G.8 — 배치 추론 + 레벨 정확도 | `test_gui_tabs.py` | `test_inference_tab_batch_level_accuracy` | Unit |
| G.9 — 전처리 파라미터 저장 | `test_gui_tabs.py` | `test_data_tab_save_preprocess_params` | Unit |
| G.10 — GradCAM | `test_gui_tabs.py` | `test_inference_tab_gradcam_button_state` | Unit |
| G.11 — 전체 채널 평가 | `test_gui_tabs.py` | `test_evaluation_tab_all_channels_sequence` | Unit |
| G.12 — Before/After 비교 | `test_gui_tabs.py` | `test_optuna_tab_snapshot_before_after` | Unit |
| G.13 — 다채널 임베딩 | `test_gui_tabs.py` | `test_embedding_tab_all_channels` | Unit |
| G.14 — Worker 설정 저장 | `test_gui_tabs.py` | `test_settings_tab_worker_settings_save` | Unit |
| G.15 — 데이터 소스 선택기 | `test_gui_tabs.py` | `test_inference_tab_data_source_selector` | Unit |

---

## 관련 문서 / Related Documents

| 문서 | 관계 |
|---|---|
| [Contract_gui.md](../Contract/Contract_gui.md) | GUI 모듈 경계 계약 |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | PyQt6 프레임워크 정의 |
| [TDD_GUI.md](../TDD/TDD_GUI.md) | GUI TDD 명세 |
