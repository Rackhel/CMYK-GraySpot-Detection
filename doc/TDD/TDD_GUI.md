---
type: tdd
domain: gui
status: failing
last_updated: 2026-05-28
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_GUI.md"
  - "../Contract/Contract_gui.md"
---

# [TDD] GUI — PyQt6 Worker 및 탭 계약 / PyQt6 Worker and Tab Contracts

> **목적 / Purpose**: PyQt6 Worker 스레드 시그널, 탭 인터페이스, GUI↔src 의존성을 BDD/TDD로 정의한다. / Define PyQt6 Worker thread signals, tab interfaces, and GUI↔src dependencies via BDD/TDD.
> **테스트 파일 / Test Files**: `src/tests/unit/test_gui_workers.py`, `src/tests/unit/test_gui_tabs.py`
> **상태 / Status**: 🔴 Failing — 구현 전 / Before implementation

---

## 1. BDD 시나리오 / BDD Scenarios

### Feature: Worker 시그널 / Worker Signals

**Scenario 1: 모든 Worker가 필수 시그널을 가진다 / All Workers have required signals**
```
Given  PyQt6 환경이 사용 가능하다
When   TrainingWorker, EvaluationWorker, TuningWorker, EmbeddingWorker,
       InferenceWorker, BatchInferenceWorker, GradCAMWorker 가 초기화된다
Then   각 Worker가 progress_updated, log_emitted, finished, error_occurred 시그널을 가진다
And    시그널이 Qt pyqtSignal 타입이다
```

**Scenario 2: Worker에서 UI 직접 접근 금지 / Worker prohibited from directly accessing UI**
```
Given  TrainingWorker 인스턴스가 있을 때
When   run() 메서드를 검사하면
Then   QWidget, QLabel, QProgressBar 등 UI 위젯을 직접 참조하는 코드가 없다
```

**Scenario 3: 오류 발생 시 error_occurred 시그널 / error_occurred signal on error**
```
Given  잘못된 cfg (필수 키 누락) 로 TrainingWorker 가 생성되었을 때
When   worker.start() 로 실행하면
Then   error_occurred 시그널이 str 타입 메시지와 함께 발생한다
And    finished 시그널은 발생하지 않는다
```

**Scenario 4: 앱 종료 시 Worker가 안전하게 종료된다 / Worker safely terminates on app close**
```
Given  Worker가 백그라운드에서 실행 중이다
When   사용자가 앱 창을 닫는다
Then   Worker의 QThread가 join된다
And    강제 종료(kill)가 발생하지 않는다
```

---

### Feature: BaseTab 인터페이스 / BaseTab Interface

**Scenario 5: 모든 탭이 BaseTab 인터페이스를 구현 / All tabs implement BaseTab interface**
```
Given  7개 탭 클래스가 있을 때
When   각 클래스를 검사하면
Then   refresh() 메서드가 존재하고
And    on_worker_finished(dict) 메서드가 존재한다
```

**Scenario 6: Embedding 탭 라벨 수정 후 버전 증가된 CSV 저장**
```
Given  EmbeddingTab이 표시되고 사용자가 샘플의 레벨을 수정했을 때
When   save_label_correction(path, new_level) 를 호출하면
Then   labels_vN+1.csv 가 생성되고
And    해당 경로의 level이 new_level로 저장된다
```

---

### Feature: 진입점 및 앱 구성 / Entry Point and App Setup

**Scenario 7: QApplication 단일 인스턴스 + 7탭 / QApplication single instance + 7 tabs**
```
Given  main.py 가 실행되었을 때
When   QApplication 이 생성되면
Then   QTabWidget 이 7개 탭을 포함한다
```

---

## 2. TDD 스펙 / TDD Specifications

### 2.1 Worker 시그널 등록 확인 / Worker Signal Registration Check

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-01 | `TrainingWorker` 가 `progress_updated` pyqtSignal 보유 |
| T-GUI-02 | `TrainingWorker` 가 `log_emitted` pyqtSignal 보유 |
| T-GUI-03 | `TrainingWorker` 가 `finished` pyqtSignal 보유 |
| T-GUI-04 | `TrainingWorker` 가 `error_occurred` pyqtSignal 보유 |
| T-GUI-05 | `EvaluationWorker` 동일 4개 시그널 보유 |
| T-GUI-06 | `TuningWorker` 동일 4개 시그널 보유 |
| T-GUI-07 | `EmbeddingWorker` 동일 4개 시그널 보유 |
| T-GUI-08 | `InferenceWorker` 동일 4개 시그널 보유 |
| T-GUI-09 | `BatchInferenceWorker` 동일 4개 시그널 보유 |
| T-GUI-10 | `GradCAMWorker` 동일 4개 시그널 보유 |

```python
def test_all_workers_have_signals():
    from gui.workers.training_worker import TrainingWorker
    from gui.workers.evaluation_worker import EvaluationWorker
    from gui.workers.tuning_worker import TuningWorker
    from gui.workers.embedding_worker import EmbeddingWorker
    from gui.workers.inference_worker import InferenceWorker
    from gui.workers.batch_inference_worker import BatchInferenceWorker
    from gui.workers.gradcam_worker import GradCAMWorker

    workers = [
        TrainingWorker, EvaluationWorker, TuningWorker,
        EmbeddingWorker, InferenceWorker, BatchInferenceWorker, GradCAMWorker,
    ]
    for WorkerClass in workers:
        for signal_name in ["progress_updated", "log_emitted", "finished", "error_occurred"]:
            assert hasattr(WorkerClass, signal_name), \
                f"{WorkerClass.__name__} missing signal: {signal_name}"
```

---

### 2.2 Worker UI 격리 검증 / Worker UI Isolation Verification

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-11 | `TrainingWorker.run()` 소스에 `QWidget`, `QLabel`, `QProgressBar` 직접 참조 없음 |
| T-GUI-12 | Worker가 UI 위젯을 생성자 인수로 받지 않음 |

```python
import inspect

def test_worker_does_not_touch_ui_directly():
    from gui.workers.training_worker import TrainingWorker
    source = inspect.getsource(TrainingWorker.run)
    forbidden = ["QWidget", "QLabel", "QProgressBar", "QLineEdit", "setText", "setValue"]
    for token in forbidden:
        assert token not in source, \
            f"TrainingWorker.run() directly references UI widget: {token}"

def test_worker_thread_joins_on_quit(qtbot):
    from gui.workers.training_worker import TrainingWorker
    import threading
    cfg = {}
    worker = TrainingWorker(cfg, phase=2, channel="Y")
    # is_running() → False after quit() is called
    worker.quit()
    worker.wait(2000)
    assert not worker.isRunning()
```

---

### 2.3 탭 인터페이스 확인 / Tab Interface Check

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-20 | 7개 탭 클래스 모두 `refresh()` 메서드 보유 |
| T-GUI-21 | 7개 탭 클래스 모두 `on_worker_finished(dict)` 메서드 보유 |
| T-GUI-22 | `MainWindow` 의 탭 위젯이 7개 탭 포함 |

```python
def test_all_tabs_implement_interface(qtbot):
    from gui.tabs.tab_data import DataTab
    from gui.tabs.tab_training import TrainingTab
    from gui.tabs.tab_evaluation import EvaluationTab
    from gui.tabs.tab_optuna import OptunaTab
    from gui.tabs.tab_embedding import EmbeddingTab
    from gui.tabs.tab_inference import InferenceTab
    from gui.tabs.tab_settings import SettingsTab

    tabs = [DataTab, TrainingTab, EvaluationTab, OptunaTab,
            EmbeddingTab, InferenceTab, SettingsTab]
    for TabClass in tabs:
        assert hasattr(TabClass, "refresh"), \
            f"{TabClass.__name__} missing refresh()"
        assert hasattr(TabClass, "on_worker_finished"), \
            f"{TabClass.__name__} missing on_worker_finished()"

def test_main_window_has_seven_tabs(qtbot):
    from gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.tab_widget.count() == 7
```

---

### 2.4 Training 탭 진행률 + 학습 곡선 / Training Tab Progress + Chart

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-23 | `TrainingTab` 이 `TrainingChart` 위젯을 포함한다 |
| T-GUI-24 | `TrainingChart.parse_log_line()` 이 epoch 로그를 파싱해 append_epoch를 호출한다 |
| T-GUI-25 | `on_worker_finished()` 호출 후 차트가 result에서 데이터를 로드한다 |

```python
def test_training_tab_progress_update(qtbot):
    from gui.tabs.tab_training import TrainingTab
    from gui.components.training_chart import TrainingChart
    tab = TrainingTab({})
    qtbot.addWidget(tab)
    assert hasattr(tab, "chart") or any(
        isinstance(w, TrainingChart) for w in tab.findChildren(TrainingChart)
    )

def test_training_chart_parses_log_line():
    from gui.components.training_chart import TrainingChart
    chart = TrainingChart()
    chart.parse_log_line("[epoch 1] loss=0.42 val_acc=0.75")
    assert len(chart._epochs) == 1
    assert chart._val_acc[0] == pytest.approx(0.75)
```

---

### 2.5 Embedding 탭 라벨 저장 버전 증가 / Embedding Tab Label Save Version Increment

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-30 | `save_label_correction(path, 3)` 호출 후 `labels_v*.csv` 파일 생성 |
| T-GUI-31 | 저장된 CSV에서 해당 path의 level == 3 |
| T-GUI-32 | 연속 저장 시 버전 번호가 순차 증가 (v1 → v2) |

```python
def test_embedding_tab_save_labels_increments_version(qtbot, tmp_path, monkeypatch):
    from gui.tabs.tab_embedding import EmbeddingTab
    tab = EmbeddingTab(cfg={}, labels_dir=str(tmp_path))
    qtbot.addWidget(tab)
    tab.save_label_correction("img1.png", new_level=3)
    csv_files = sorted(tmp_path.glob("labels_v*.csv"))
    assert len(csv_files) >= 1
    import pandas as pd
    df = pd.read_csv(csv_files[-1])
    row = df[df["path"] == "img1.png"]
    assert not row.empty
    assert int(row.iloc[0]["level"]) == 3
```

---

### 2.6 InferenceTab 가드 및 기능 / InferenceTab Guards and Features

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-40 | 이미지 미선택 시 `start_single_inference()` 크래시 없이 로그만 출력 |
| T-GUI-41 | 폴더 미선택 시 `start_batch_inference()` 크래시 없이 로그만 출력 |
| T-GUI-42 | `InferenceTab.refresh()` 체크포인트 경로 갱신 |
| T-GUI-43 | 단일 추론 완료 후 단일 채널이면 GradCAM 버튼 활성화 |
| T-GUI-44 | 앙상블(all) 채널 추론 완료 후 GradCAM 버튼 비활성 유지 |
| T-GUI-45 | 배치 추론 완료 후 `LevelAccuracyTable` 업데이트 |
| T-GUI-46 | 데이터 소스 변경 시 파일 다이얼로그 시작 경로가 변경된다 |

```python
def test_inference_tab_single_image_guard(qtbot):
    from gui.tabs.tab_inference import InferenceTab
    tab = InferenceTab({})
    qtbot.addWidget(tab)
    tab.start_single_inference()    # 이미지 미선택 — 예외 없이 로그만

def test_inference_tab_gradcam_button_state(qtbot):
    from gui.tabs.tab_inference import InferenceTab
    tab = InferenceTab({})
    qtbot.addWidget(tab)
    # GradCAM 버튼은 초기에 비활성
    assert not tab._run_gradcam_btn.isEnabled()
    # 단일 채널 추론 완료 시뮬레이션
    tab._on_single_finished({"pred_level": 2, "confidence": 0.85,
                              "probs": [], "top3": [], "channel": "Y",
                              "image_path": "test.png", "checkpoint": "ckpt.pth"})
    assert tab._run_gradcam_btn.isEnabled()
    # 앙상블(all) 완료 시 GradCAM 버튼 비활성
    tab._on_single_finished({"pred_level": 2, "confidence": 0.85,
                              "probs": [], "top3": [], "channel": "all",
                              "image_path": "test.png"})
    assert not tab._run_gradcam_btn.isEnabled()

def test_inference_tab_batch_level_accuracy(qtbot):
    from gui.tabs.tab_inference import InferenceTab
    from gui.components.level_accuracy_table import LevelAccuracyTable
    tab = InferenceTab({})
    qtbot.addWidget(tab)
    results = [
        {"filename": "a.png", "path": "data_set/labeled/Y/level_0/a.png",
         "pred_level": 0, "confidence": 0.9, "top3": [], "error": None},
        {"filename": "b.png", "path": "data_set/labeled/Y/level_1/b.png",
         "pred_level": 1, "confidence": 0.7, "top3": [], "error": None},
    ]
    tab._on_batch_finished({"results": results, "total": 2, "succeeded": 2, "failed": 0})
    # LevelAccuracyTable이 존재하고 업데이트됨
    assert tab._level_acc_table is not None

def test_inference_tab_data_source_selector(qtbot):
    from gui.tabs.tab_inference import InferenceTab
    tab = InferenceTab({})
    qtbot.addWidget(tab)
    # 데이터 소스 콤보박스 존재 확인
    assert hasattr(tab, "_src_combo")
    # "raw" 선택 시 시작 경로 변경 확인
    idx = tab._src_combo.findText("Raw") if tab._src_combo.findText("Raw") >= 0 else 0
    tab._src_combo.setCurrentIndex(idx)
    # _DATA_SOURCES 키에 "raw" 매핑 존재
    from gui.tabs.tab_inference import _DATA_SOURCES
    assert "raw" in _DATA_SOURCES
```

---

### 2.7 Data 탭 전처리 파라미터 저장 / DataTab Preprocessing Params Save

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-50 | `_save_preprocess()` 호출 후 `src/config/config.json` 에 `data.image_size` 반영 |
| T-GUI-51 | mean/std 값이 3-요소 리스트로 저장된다 |
| T-GUI-52 | split_ratios 합이 1.0이 아닐 때 저장하지 않고 오류 로그 출력 |

```python
def test_data_tab_save_preprocess_params(qtbot, tmp_path, monkeypatch):
    import json, pathlib
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"data": {"image_size": 128, "mean": [0.5]*3,
                                                  "std": [0.5]*3, "split_ratios": {}}}))
    monkeypatch.setattr("gui.tabs.tab_data._CONFIG_PATH", str(config_path))
    from gui.tabs.tab_data import DataTab
    tab = DataTab({})
    qtbot.addWidget(tab)
    tab._image_size_spin.setValue(256)
    tab._save_preprocess()
    saved = json.loads(config_path.read_text())
    assert saved["data"]["image_size"] == 256
```

---

### 2.8 Evaluation 탭 전체 채널 순차 평가 / EvaluationTab All-Channel Sequential

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-53 | `EvaluationTab` 이 `acc_card`, `f1_card`, `mae_card`, `n_card` 를 포함한다 |
| T-GUI-54 | 채널 "전체(All)" 선택 후 `start_evaluation()` 시 `_pending` 큐에 4개 채널 적재 |
| T-GUI-55 | 4채널 완료 후 `_show_overall_avg()` 가 LogPanel에 전체 평균을 출력한다 |

```python
def test_evaluation_tab_all_channels_sequence(qtbot):
    from gui.tabs.tab_evaluation import EvaluationTab
    tab = EvaluationTab({})
    qtbot.addWidget(tab)
    # 메트릭 카드 존재 확인
    for attr in ["acc_card", "f1_card", "mae_card", "n_card"]:
        assert hasattr(tab, attr), f"EvaluationTab missing {attr}"
    # "All" 채널 콤보 선택
    all_idx = tab._ch_combo.findText("전체(All)")
    if all_idx >= 0:
        tab._ch_combo.setCurrentIndex(all_idx)
        tab._pending = []  # reset
        tab.start_evaluation()
        assert len(tab._pending) == 4
```

---

### 2.9 Optuna 탭 Before/After 비교 / OptunaTab Before/After Comparison

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-56 | `_take_snapshot()` 호출 후 `_before_snapshot` 딕셔너리가 채워진다 |
| T-GUI-57 | `on_worker_finished()` 이후 `_after_lbl` 텍스트가 업데이트된다 |
| T-GUI-58 | 5개 메트릭 카드(best, f1, mae, vacc, tacc)가 존재한다 |

```python
def test_optuna_tab_snapshot_before_after(qtbot):
    from gui.tabs.tab_optuna import OptunaTab
    tab = OptunaTab({})
    qtbot.addWidget(tab)
    # 카드 존재 확인
    for attr in ["best_card", "f1_card", "mae_card", "vacc_card", "tacc_card"]:
        assert hasattr(tab, attr), f"OptunaTab missing {attr}"
    # 스냅샷 저장
    tab._take_snapshot()
    assert isinstance(tab._before_snapshot, dict)
    assert len(tab._before_snapshot) > 0
    # HPO 완료 시뮬레이션
    tab.on_worker_finished({
        "best_params": {"p2_lr": 0.001, "p2_wd": 0.0001},
        "best_value": 0.92,
        "val_acc": 0.90,
        "test_acc": 0.88,
        "macro_f1": 0.87,
        "mae": 0.12,
    })
    assert tab._after_lbl.text() != ""
```

---

### 2.10 Embedding 탭 다채널 / EmbeddingTab Multi-Channel

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-59 | `EmbeddingTab` 에 4개 내부 탭(t-SNE, 레벨 순도, 유사 이미지, 라벨 교정)이 있다 |
| T-GUI-60 | "전체 비교(All)" 선택 후 `start_embedding()` 시 `_pending_channels` 에 4개 채널 적재 |

```python
def test_embedding_tab_all_channels(qtbot):
    from gui.tabs.tab_embedding import EmbeddingTab
    tab = EmbeddingTab({})
    qtbot.addWidget(tab)
    # 내부 QTabWidget이 4개 탭을 포함하는지 확인
    from PyQt6.QtWidgets import QTabWidget
    inner_tabs = tab.findChildren(QTabWidget)
    assert any(t.count() >= 4 for t in inner_tabs)
    # 전체 비교 선택
    all_idx = tab._ch_combo.findText("전체 비교(All)")
    if all_idx >= 0:
        tab._ch_combo.setCurrentIndex(all_idx)
        tab._pending_channels = []
        tab.start_embedding()
        assert len(tab._pending_channels) == 4
```

---

### 2.11 Settings 탭 Worker 설정 저장 / SettingsTab Worker Settings Save

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-61 | `SettingsTab` 이 device, num_workers, infer_batch_size, train_timeout 위젯을 포함한다 |
| T-GUI-62 | `save_settings()` 후 `src/config/config.json` 에 `system.device` 가 반영된다 |

```python
def test_settings_tab_worker_settings_save(qtbot, tmp_path, monkeypatch):
    import json
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"system": {"device": "auto"}, "train": {}}))
    monkeypatch.setattr("gui.tabs.tab_settings._SRC_CONFIG_PATH", str(config_path))
    from gui.tabs.tab_settings import SettingsTab
    tab = SettingsTab({})
    qtbot.addWidget(tab)
    # device 위젯 존재 확인
    assert hasattr(tab, "_device_combo"), "SettingsTab missing _device_combo"
    # "cpu" 선택 후 저장
    cpu_idx = tab._device_combo.findText("cpu")
    if cpu_idx >= 0:
        tab._device_combo.setCurrentIndex(cpu_idx)
    tab.save_settings()
    saved = json.loads(config_path.read_text())
    assert saved["system"]["device"] == "cpu"
```

---

## 3. 추적 매트릭스 / Traceability Matrix

| BDD 시나리오 | TDD 파일 | 테스트 함수 | 테스트 ID |
| --- | --- | --- | --- |
| G.1 — 워커 신호 | `test_gui_workers.py` | `test_all_workers_have_signals` | T-GUI-01~10 |
| G.2 — UI 직접 접근 금지 | `test_gui_workers.py` | `test_worker_does_not_touch_ui_directly` | T-GUI-11 |
| G.3 — 탭 인터페이스 | `test_gui_tabs.py` | `test_all_tabs_implement_interface` | T-GUI-20~21 |
| G.4 — 진행률 + 학습 곡선 | `test_gui_tabs.py` | `test_training_tab_progress_update` | T-GUI-23~25 |
| G.5 — 라벨 저장 | `test_gui_tabs.py` | `test_embedding_tab_save_labels_increments_version` | T-GUI-30~32 |
| G.6 — 워커 정리 | `test_gui_workers.py` | `test_worker_thread_joins_on_quit` | T-GUI-12 |
| G.7 — 단일 추론 가드 | `test_gui_tabs.py` | `test_inference_tab_single_image_guard` | T-GUI-40 |
| G.8 — 배치 추론 + 레벨 정확도 | `test_gui_tabs.py` | `test_inference_tab_batch_level_accuracy` | T-GUI-45 |
| G.9 — 전처리 파라미터 저장 | `test_gui_tabs.py` | `test_data_tab_save_preprocess_params` | T-GUI-50~52 |
| G.10 — GradCAM 버튼 상태 | `test_gui_tabs.py` | `test_inference_tab_gradcam_button_state` | T-GUI-43~44 |
| G.11 — 전체 채널 평가 | `test_gui_tabs.py` | `test_evaluation_tab_all_channels_sequence` | T-GUI-53~55 |
| G.12 — Before/After 비교 | `test_gui_tabs.py` | `test_optuna_tab_snapshot_before_after` | T-GUI-56~58 |
| G.13 — 다채널 임베딩 | `test_gui_tabs.py` | `test_embedding_tab_all_channels` | T-GUI-59~60 |
| G.14 — Worker 설정 저장 | `test_gui_tabs.py` | `test_settings_tab_worker_settings_save` | T-GUI-61~62 |
| G.15 — 데이터 소스 선택기 | `test_gui_tabs.py` | `test_inference_tab_data_source_selector` | T-GUI-46 |

---

## 4. pytest 설정 요구사항 / pytest Configuration Requirements

```ini
# pytest.ini 에 추가 (GUI 테스트용)
[pytest]
qt_api = pyqt6
```

```
# requirements.txt 에 추가
PyQt6>=6.4.0
pytest-qt>=4.2.0
```

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | GUI 구조 정의 |
| [Contract_gui.md](../Contract/Contract_gui.md) | Worker/탭 API 계약 |
| [BDD_GUI.md](../BDD/BDD_GUI.md) | 관찰 가능한 행동 시나리오 |
