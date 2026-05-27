---
type: contract
domain: gui
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] GUI — PyQt6 애플리케이션 구현 계약 / PyQt6 Application Implementation Contract

> **목적 / Purpose**: GUI Worker 스레드, 탭 위젯, src 모듈 연동 API 계약을 정의한다. / Defines the API contracts for GUI Worker threads, tab widgets, and src module integration.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-18

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_GUI.md](../SSOT/SSOT_GUI.md)를 최종 판결자로 따른다.
> / This document does not redefine semantic definitions in SSOT documents. For semantic interpretation, follow SSOT_GUI.md as the final authority.

---

## 1. 진입점 계약 / Entry Point Contract

```python
# gui/main.py
from gui.main_window import MainWindow
import sys
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

| 항목 / Item | 값 / Value |
| --- | --- |
| 진입점 / Entry point | `python -m gui.main` 또는 / or `python gui/main.py` |
| QApplication | 단일 인스턴스 / Single instance |
| 탭 컨테이너 / Tab container | `QTabWidget` (6탭 / 6 tabs) |

---

## 2. Worker 스레드 API 계약 / Worker Thread API Contract

### 2.1 공통 시그널 / Common Signals (모든 Worker 공유 / Shared by All Workers)

```python
class BaseWorker(QThread):
    progress_updated = pyqtSignal(int)    # 0–100
    log_emitted      = pyqtSignal(str)    # 로그 메시지 / log message
    finished         = pyqtSignal(dict)   # 결과 dict / result dict
    error_occurred   = pyqtSignal(str)    # 오류 메시지 / error message
```

### 2.2 TrainingWorker

```python
class TrainingWorker(BaseWorker):
    def __init__(self, cfg: dict, phase: int, channel: str): ...
    def run(self) -> None:
        # phase=0: Phase0Trainer(cfg, channel).run()
        # phase=2: Phase2Trainer(cfg, channel).run()
        # 진행 상황 → progress_updated 시그널 / progress → progress_updated signal
        # 완료 → finished({"val_acc": float, "checkpoint": str}) / on completion
```

| 파라미터 / Parameter | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `cfg` | `dict` | `load_config()` 반환값 / return value |
| `phase` | `int` | 0 또는 / or 2 |
| `channel` | `str` | `"Y"`, `"M"`, `"C"`, `"K"` |

### 2.3 EvaluationWorker

```python
class EvaluationWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    def run(self) -> None:
        # Evaluator(cfg, channel, checkpoint_path).run()
        # 완료 → finished({"accuracy": float, "report_path": str}) / on completion
```

### 2.4 TuningWorker

```python
class TuningWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, n_trials: int): ...
    def run(self) -> None:
        # run_optuna(cfg, channel, n_trials)
        # 매 trial 완료 → progress_updated, log_emitted / per trial completion
        # 완료 → finished({"best_params": dict, "best_value": float}) / on completion
```

### 2.5 EmbeddingWorker

```python
class EmbeddingWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    def run(self) -> None:
        # GrayspotModel에서 feature 추출 / extract features → t-SNE 변환 / t-SNE transform
        # 완료 → finished({"embeddings_2d": np.ndarray, "labels": List[int], "paths": List[str]})
```

---

## 3. 탭 위젯 API 계약 / Tab Widget API Contract

### 3.1 공통 인터페이스 / Common Interface

```python
class BaseTab(QWidget):
    def refresh(self) -> None: ...          # 탭 활성화 시 데이터 갱신 / refresh data when tab is activated
    def on_worker_finished(self, result: dict) -> None: ...  # Worker 완료 처리 / handle worker completion
```

### 3.2 Tab 2: TrainingTab

```python
class TrainingTab(BaseTab):
    def start_training(self) -> None:
        # cfg, phase, channel 읽어 TrainingWorker 생성 + start()
        # / read cfg, phase, channel and create + start TrainingWorker
    def stop_training(self) -> None:
        # worker.requestInterruption() + worker.wait()
```

### 3.3 Tab 3: EvaluationTab

```python
class EvaluationTab(BaseTab):
    def load_results(self, report_path: str) -> None:
        # outputs/reports/*.json 로드 → Confusion Matrix 렌더링 / load → rendering
    def show_misclassified(self, predictions: List[dict]) -> None:
        # 오분류 샘플 이미지 + 레벨 정보 표시 / display misclassified images + level info
```

### 3.4 Tab 6: EmbeddingTab

```python
class EmbeddingTab(BaseTab):
    def render_scatter(self, embeddings_2d: np.ndarray, labels: List[int]) -> None:
        # Plotly scatter → QWebEngineView 렌더링 / rendering
    def save_label_correction(self, path: str, new_level: int) -> None:
        # labels_vN.csv에 수정 내용 추가 / add corrections to labels_vN.csv
        # N = 현재 최신 버전 + 1 / current latest version + 1
```

---

## 4. GUI ↔ src 의존성 목록 / GUI ↔ src Dependency List

| GUI 모듈 / GUI Module | 사용하는 src API / Used src API |
| --- | --- |
| `TrainingWorker` | `Phase0Trainer`, `Phase2Trainer` |
| `EvaluationWorker` | `Evaluator` |
| `TuningWorker` | `run_optuna` |
| `EmbeddingWorker` | `GrayspotModel`, `load_config` |
| `Tab 1 (Data)` | `CMYKDataset` (샘플 수 쿼리 / sample count query) |
| `Tab 6 (Embedding)` | `LabelRefiner.compute_priority_score()` |
| 모든 탭 / All tabs | `load_config()` |

---

## 5. 금지 패턴 / Prohibited Patterns

```python
# ❌ Worker에서 UI 직접 수정 금지 / Prohibited: direct UI modification in Worker
self.progress_bar.setValue(50)  # Worker 내부에서 호출 금지 / Must not be called inside Worker

# ✅ 시그널로만 전달 / Pass through signals only
self.progress_updated.emit(50)

# ❌ UI 스레드에서 블로킹 작업 금지 / Prohibited: blocking operations in UI thread
result = trainer.run()  # UI 스레드에서 직접 호출 금지 / Must not be called from UI thread

# ✅ Worker에 위임 / Delegate to Worker
self.worker = TrainingWorker(cfg, phase=2, channel="Y")
self.worker.start()
```

---

## 6. 체크리스트 / Checklist

- [ ] 새 Worker 추가 시 §2 계약 및 SSOT_GUI.md §3 테이블 갱신 / When adding a new Worker, update §2 contract and SSOT_GUI.md §3 table
- [ ] 새 탭 추가 시 `BaseTab` 인터페이스 상속 확인 / When adding a new tab, verify BaseTab interface inheritance
- [ ] Worker → UI 직접 접근 코드 리뷰 시 금지 패턴 §5 확인 / During code review, check §5 prohibited patterns for Worker → UI direct access
- [ ] `run_optuna()` 시그니처 변경 시 TuningWorker §2.4 갱신 / When run_optuna() signature changes, update TuningWorker §2.4

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | GUI 구조 정의 (What) / GUI structure definition |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | TrainingWorker가 호출하는 학습 API / Training API called by TrainingWorker |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | EvaluationWorker가 호출하는 평가 API / Evaluation API called by EvaluationWorker |
| [Contract_tuning_boundary.md](Contract_tuning_boundary.md) | TuningWorker가 호출하는 HPO API / HPO API called by TuningWorker |
