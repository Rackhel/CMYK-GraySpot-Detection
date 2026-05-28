---
type: contract
domain: gui
status: Active
last_updated: 2026-05-28
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
# gui/main.py  (요약 / summary)
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    font_name = _detect_font()          # QFontDatabase에서 플랫폼별 폰트 탐색
    app.setFont(QFont(font_name, 10))
    gui_cfg = _load_gui_cfg()           # gui/assets/config.json
    theme = gui_cfg.get("theme", "dark")
    lang  = gui_cfg.get("lang",  "ko")
    set_lang(lang)                      # i18n 전역 설정
    qss = _load_qss(f"{theme}_theme.qss", font=font_name)   # %FONT%/%ASSETS% 치환
    if qss:
        app.setStyleSheet(qss)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

| 항목 / Item | 값 / Value |
| --- | --- |
| 진입점 / Entry point | `python -m gui.main` 또는 / or `python gui/main.py` |
| QApplication | 단일 인스턴스 / Single instance |
| 탭 컨테이너 / Tab container | `QTabWidget` (7탭 / 7 tabs) |
| 폰트 / Font | `_detect_font()` — `QFontDatabase.families()` 로 플랫폼별 실존 폰트 탐색 |
| 테마·언어 초기화 / Theme & lang init | `gui/assets/config.json` 에서 읽어 시작 전 적용 |

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
    def __init__(self, cfg: dict, channel: str, n_trials: int, phase: int = 2): ...
    def run(self) -> None:
        # run_optuna(cfg, channel, n_trials, phase)
        # 매 trial 완료 → progress_updated, log_emitted / per trial completion
        # 완료 → finished({"best_params": dict, "best_value": float}) / on completion
```

| 파라미터 / Parameter | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `phase` | `int` | 0 (SimCLR) 또는 / or 2 (Supervised), default=2 |

### 2.5 EmbeddingWorker

```python
class EmbeddingWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    def run(self) -> None:
        # GrayspotModel에서 feature 추출 / extract features → t-SNE 변환 / t-SNE transform
        # 완료 → finished({"embeddings_2d": np.ndarray, "labels": List[int], "paths": List[str]})
```

### 2.6 InferenceWorker

```python
class InferenceWorker(BaseWorker):
    def __init__(self, cfg: dict, image_path: str, checkpoint_path: str): ...
    def run(self) -> None:
        # cv2 로드 → resize → ToTensor → _IMAGENET_NORMALIZE(SSOT-NM01) → model.forward
        # 완료 → finished({
        #     "pred_level": int,
        #     "confidence": float,
        #     "probs": List[float],
        #     "top3": List[Tuple[int, float]],
        #     "image_path": str,
        # })
```

| 파라미터 / Parameter | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `image_path` | `str` | 추론할 단일 이미지 경로 / Path to single image |
| `checkpoint_path` | `str` | `.pt` 체크포인트 경로 / Path to .pt checkpoint |

### 2.7 BatchInferenceWorker

```python
class BatchInferenceWorker(BaseWorker):
    def __init__(self, cfg: dict, folder_path: str, checkpoint_path: str): ...
    def run(self) -> None:
        # 폴더 내 모든 이미지 순회 / iterate all images in folder
        # 이미지당 "__ROW__<JSON>" 형식으로 log_emitted 발행 (실시간 테이블 업데이트)
        # / emit "__ROW__<JSON>" per image via log_emitted for live table update
        # 완료 → finished({
        #     "results":   List[dict],   # per-image result
        #     "total":     int,
        #     "succeeded": int,
        #     "failed":    int,
        # })
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
    def start_evaluation(self) -> None:
        # 중복 실행 방지 후 EvaluationWorker 시작 / start EvaluationWorker with duplicate-run guard
    def stop_evaluation(self) -> None: ...
    def start_inference(self) -> None:
        # 단일 이미지 추론 (InferenceWorker) / single-image inference via InferenceWorker
    def _browse_image(self) -> None: ...   # 이미지 선택 다이얼로그
```

### 3.4 Tab 4: SettingsTab (추가 계약 / additional contract)

```python
class SettingsTab(BaseTab):
    _theme_combo: QComboBox   # userData="dark"|"light"
    _lang_combo:  QComboBox   # userData="ko"|"en"
    def get_checkpoint_path(self) -> str: ...
    def save_settings(self) -> None:
        # src/config/config.json + gui/assets/config.json 동시 저장
        # / saves both src/config/config.json and gui/assets/config.json
```

### 3.5 Tab 6: EmbeddingTab

```python
class EmbeddingTab(BaseTab):
    def save_label_correction(self, path: str, new_level: int) -> None:
        # labels_vN.csv 버전 증가 방식 저장 / version-incremented CSV save
        # N = max existing version + 1
```

### 3.6 Tab 7: InferenceTab

```python
class InferenceTab(BaseTab):
    def start_single_inference(self) -> None:
        # 중복 실행 방지 → InferenceWorker 시작 / duplicate guard → start InferenceWorker
    def start_batch_inference(self) -> None:
        # 중복 실행 방지 → BatchInferenceWorker 시작 / duplicate guard → start BatchInferenceWorker
    def _export_csv(self) -> None:
        # 배치 결과를 CSV로 내보내기 / export batch results to CSV
    def refresh(self) -> None:
        # gui/assets/config.json 에서 체크포인트 경로 갱신
        # / reload checkpoint path from gui/assets/config.json
```

---

## 4. GUI ↔ src 의존성 목록 / GUI ↔ src Dependency List

| GUI 모듈 / GUI Module | 사용하는 src API / Used src API |
| --- | --- |
| `TrainingWorker` | `Phase0Trainer`, `Phase2Trainer` |
| `EvaluationWorker` | `Evaluator` |
| `TuningWorker` | `run_optuna` |
| `EmbeddingWorker` | `GrayspotModel`, `load_config` |
| `InferenceWorker` | `build_model` (`src/utils/utils_model`), `_IMAGENET_NORMALIZE` (`src/data/normalize`) |
| `BatchInferenceWorker` | `build_model`, `_IMAGENET_NORMALIZE` (동일 / same as InferenceWorker) |
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
