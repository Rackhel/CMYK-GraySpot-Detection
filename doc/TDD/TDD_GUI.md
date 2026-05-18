---
type: tdd
domain: gui
status: failing
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_GUI.md"
  - "../Contract/Contract_gui.md"
---

# [TDD] GUI — PyQt6 Worker 및 탭 계약

> **목적**: PyQt6 Worker 스레드 시그널, 탭 인터페이스, GUI↔src 의존성을 BDD/TDD로 정의한다.
> **테스트 파일**: `src/tests/unit/test_gui_workers.py`, `src/tests/unit/test_gui_tabs.py`
> **상태**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오

### Feature: TrainingWorker 시그널

**Scenario 1: 학습 완료 시 finished 시그널 발생**
```
Given  유효한 cfg와 phase=2, channel="Y" 로 TrainingWorker 가 생성되었을 때
When   worker.start() 로 실행하면
Then   학습 완료 후 finished 시그널이 발생하고
And    finished payload의 "val_acc" 키가 float 값을 가진다
```

**Scenario 2: Worker에서 UI 직접 접근 금지**
```
Given  TrainingWorker 인스턴스가 있을 때
When   run() 메서드를 검사하면
Then   QWidget, QLabel, QProgressBar 등 UI 위젯을 직접 참조하는 코드가 없다
```

**Scenario 3: 오류 발생 시 error_occurred 시그널**
```
Given  잘못된 cfg (필수 키 누락) 로 TrainingWorker 가 생성되었을 때
When   worker.start() 로 실행하면
Then   error_occurred 시그널이 str 타입 메시지와 함께 발생한다
And    finished 시그널은 발생하지 않는다
```

---

### Feature: BaseTab 인터페이스

**Scenario 4: 모든 탭이 BaseTab 인터페이스를 구현**
```
Given  6개 탭 클래스가 있을 때
When   각 클래스를 검사하면
Then   refresh() 메서드가 존재하고
And    on_worker_finished(dict) 메서드가 존재한다
```

**Scenario 5: Tab 6 라벨 수정 후 CSV 저장**
```
Given  EmbeddingTab이 표시되고 사용자가 샘플의 레벨을 수정했을 때
When   save_label_correction(path, new_level) 를 호출하면
Then   labels_vN.csv 가 생성되고
And    해당 경로의 level이 new_level로 저장된다
```

---

### Feature: 진입점 및 앱 구성

**Scenario 6: QApplication 단일 인스턴스**
```
Given  main.py 가 실행되었을 때
When   QApplication 이 생성되면
Then   QTabWidget 이 6개 탭을 포함한다
```

---

## 2. TDD 스펙

### 2.1 Worker 시그널 등록 확인

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-01 | `TrainingWorker` 가 `progress_updated` pyqtSignal 보유 |
| T-GUI-02 | `TrainingWorker` 가 `log_emitted` pyqtSignal 보유 |
| T-GUI-03 | `TrainingWorker` 가 `finished` pyqtSignal 보유 |
| T-GUI-04 | `TrainingWorker` 가 `error_occurred` pyqtSignal 보유 |
| T-GUI-05 | `EvaluationWorker` 동일 4개 시그널 보유 |
| T-GUI-06 | `TuningWorker` 동일 4개 시그널 보유 |
| T-GUI-07 | `EmbeddingWorker` 동일 4개 시그널 보유 |

```python
from PyQt6.QtCore import pyqtSignal

def test_training_worker_has_required_signals():
    from gui.workers.training_worker import TrainingWorker
    assert hasattr(TrainingWorker, "progress_updated")
    assert hasattr(TrainingWorker, "log_emitted")
    assert hasattr(TrainingWorker, "finished")
    assert hasattr(TrainingWorker, "error_occurred")

def test_all_workers_have_signals():
    from gui.workers.training_worker import TrainingWorker
    from gui.workers.evaluation_worker import EvaluationWorker
    from gui.workers.tuning_worker import TuningWorker
    from gui.workers.embedding_worker import EmbeddingWorker
    for WorkerClass in [TrainingWorker, EvaluationWorker, TuningWorker, EmbeddingWorker]:
        for signal_name in ["progress_updated", "log_emitted", "finished", "error_occurred"]:
            assert hasattr(WorkerClass, signal_name), f"{WorkerClass.__name__} missing {signal_name}"
```

### 2.2 Worker UI 격리 검증

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-10 | `TrainingWorker.run()` 소스에 `QWidget`, `QLabel`, `QProgressBar` 직접 참조 없음 |
| T-GUI-11 | Worker가 UI 위젯을 생성자 인수로 받지 않음 |

```python
import inspect
def test_training_worker_no_direct_ui_access():
    from gui.workers.training_worker import TrainingWorker
    source = inspect.getsource(TrainingWorker.run)
    forbidden = ["QWidget", "QLabel", "QProgressBar", "QLineEdit", "setText", "setValue"]
    for token in forbidden:
        assert token not in source, f"TrainingWorker.run() directly references UI: {token}"
```

### 2.3 탭 인터페이스 확인

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-20 | 6개 탭 클래스 모두 `refresh()` 메서드 보유 |
| T-GUI-21 | 6개 탭 클래스 모두 `on_worker_finished(dict)` 메서드 보유 |
| T-GUI-22 | `MainWindow` 의 탭 위젯이 6개 탭 포함 |

```python
def test_all_tabs_implement_interface(qtbot):
    from gui.tabs.tab_data import DataTab
    from gui.tabs.tab_training import TrainingTab
    from gui.tabs.tab_evaluation import EvaluationTab
    from gui.tabs.tab_settings import SettingsTab
    from gui.tabs.tab_optuna import OptunaTab
    from gui.tabs.tab_embedding import EmbeddingTab

    for TabClass in [DataTab, TrainingTab, EvaluationTab, SettingsTab, OptunaTab, EmbeddingTab]:
        assert hasattr(TabClass, "refresh"), f"{TabClass.__name__} missing refresh()"
        assert hasattr(TabClass, "on_worker_finished"), f"{TabClass.__name__} missing on_worker_finished()"

def test_main_window_has_six_tabs(qtbot):
    from gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.tab_widget.count() == 6
```

### 2.4 Tab 6: 라벨 수정 저장

| 테스트 ID | 검증 포인트 |
| --- | --- |
| T-GUI-30 | `save_label_correction(path, 3)` 호출 후 CSV 파일 존재 |
| T-GUI-31 | 저장된 CSV에서 해당 path의 level == 3 |

```python
def test_embedding_tab_save_label(qtbot, tmp_path, monkeypatch):
    from gui.tabs.tab_embedding import EmbeddingTab
    tab = EmbeddingTab(cfg={}, labels_dir=tmp_path)
    qtbot.addWidget(tab)
    tab.save_label_correction("img1.png", new_level=3)
    csv_files = list(tmp_path.glob("labels_v*.csv"))
    assert len(csv_files) > 0
```

---

## 3. pytest 설정 요구사항

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
