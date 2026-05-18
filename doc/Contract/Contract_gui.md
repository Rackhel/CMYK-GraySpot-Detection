---
type: contract
domain: gui
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] GUI — PyQt6 애플리케이션 구현 계약

> **목적**: GUI Worker 스레드, 탭 위젯, src 모듈 연동 API 계약을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-18

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_GUI.md](../SSOT/SSOT_GUI.md)를 최종 판결자로 따른다.

---

## 1. 진입점 계약

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

| 항목 | 값 |
| --- | --- |
| 진입점 | `python -m gui.main` 또는 `python gui/main.py` |
| QApplication | 단일 인스턴스 |
| 탭 컨테이너 | `QTabWidget` (6탭) |

---

## 2. Worker 스레드 API 계약

### 2.1 공통 시그널 (모든 Worker 공유)

```python
class BaseWorker(QThread):
    progress_updated = pyqtSignal(int)    # 0–100
    log_emitted      = pyqtSignal(str)    # 로그 메시지
    finished         = pyqtSignal(dict)   # 결과 dict
    error_occurred   = pyqtSignal(str)    # 오류 메시지
```

### 2.2 TrainingWorker

```python
class TrainingWorker(BaseWorker):
    def __init__(self, cfg: dict, phase: int, channel: str): ...
    def run(self) -> None:
        # phase=0: Phase0Trainer(cfg, channel).run()
        # phase=2: Phase2Trainer(cfg, channel).run()
        # 진행 상황 → progress_updated 시그널
        # 완료 → finished({"val_acc": float, "checkpoint": str})
```

| 파라미터 | 타입 | 설명 |
| --- | --- | --- |
| `cfg` | `dict` | `load_config()` 반환값 |
| `phase` | `int` | 0 또는 2 |
| `channel` | `str` | `"Y"`, `"M"`, `"C"`, `"K"` |

### 2.3 EvaluationWorker

```python
class EvaluationWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    def run(self) -> None:
        # Evaluator(cfg, channel, checkpoint_path).run()
        # 완료 → finished({"accuracy": float, "report_path": str})
```

### 2.4 TuningWorker

```python
class TuningWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, n_trials: int): ...
    def run(self) -> None:
        # run_optuna(cfg, channel, n_trials)
        # 매 trial 완료 → progress_updated, log_emitted
        # 완료 → finished({"best_params": dict, "best_value": float})
```

### 2.5 EmbeddingWorker

```python
class EmbeddingWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    def run(self) -> None:
        # GrayspotModel에서 feature 추출 → t-SNE 변환
        # 완료 → finished({"embeddings_2d": np.ndarray, "labels": List[int], "paths": List[str]})
```

---

## 3. 탭 위젯 API 계약

### 3.1 공통 인터페이스

```python
class BaseTab(QWidget):
    def refresh(self) -> None: ...          # 탭 활성화 시 데이터 갱신
    def on_worker_finished(self, result: dict) -> None: ...  # Worker 완료 처리
```

### 3.2 Tab 2: TrainingTab

```python
class TrainingTab(BaseTab):
    def start_training(self) -> None:
        # cfg, phase, channel 읽어 TrainingWorker 생성 + start()
    def stop_training(self) -> None:
        # worker.requestInterruption() + worker.wait()
```

### 3.3 Tab 3: EvaluationTab

```python
class EvaluationTab(BaseTab):
    def load_results(self, report_path: str) -> None:
        # outputs/reports/*.json 로드 → Confusion Matrix 렌더링
    def show_misclassified(self, predictions: List[dict]) -> None:
        # 오분류 샘플 이미지 + 레벨 정보 표시
```

### 3.4 Tab 6: EmbeddingTab

```python
class EmbeddingTab(BaseTab):
    def render_scatter(self, embeddings_2d: np.ndarray, labels: List[int]) -> None:
        # Plotly scatter → QWebEngineView 렌더링
    def save_label_correction(self, path: str, new_level: int) -> None:
        # labels_vN.csv에 수정 내용 추가
        # N = 현재 최신 버전 + 1
```

---

## 4. GUI ↔ src 의존성 목록

| GUI 모듈 | 사용하는 src API |
| --- | --- |
| `TrainingWorker` | `Phase0Trainer`, `Phase2Trainer` |
| `EvaluationWorker` | `Evaluator` |
| `TuningWorker` | `run_optuna` |
| `EmbeddingWorker` | `GrayspotModel`, `load_config` |
| `Tab 1 (Data)` | `CMYKDataset` (샘플 수 쿼리) |
| `Tab 6 (Embedding)` | `LabelRefiner.compute_priority_score()` |
| 모든 탭 | `load_config()` |

---

## 5. 금지 패턴

```python
# ❌ Worker에서 UI 직접 수정 금지
self.progress_bar.setValue(50)  # Worker 내부에서 호출 금지

# ✅ 시그널로만 전달
self.progress_updated.emit(50)

# ❌ UI 스레드에서 블로킹 작업 금지
result = trainer.run()  # UI 스레드에서 직접 호출 금지

# ✅ Worker에 위임
self.worker = TrainingWorker(cfg, phase=2, channel="Y")
self.worker.start()
```

---

## 6. 체크리스트

- [ ] 새 Worker 추가 시 §2 계약 및 SSOT_GUI.md §3 테이블 갱신
- [ ] 새 탭 추가 시 `BaseTab` 인터페이스 상속 확인
- [ ] Worker → UI 직접 접근 코드 리뷰 시 금지 패턴 §5 확인
- [ ] `run_optuna()` 시그니처 변경 시 TuningWorker §2.4 갱신

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | GUI 구조 정의 (What) |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | TrainingWorker가 호출하는 학습 API |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | EvaluationWorker가 호출하는 평가 API |
| [Contract_tuning_boundary.md](Contract_tuning_boundary.md) | TuningWorker가 호출하는 HPO API |
