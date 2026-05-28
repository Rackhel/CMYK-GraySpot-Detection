---
type: contract
domain: gui
status: Active
last_updated: 2026-05-28
owner: CMYK WooSong Team
---

# [Contract] GUI — PyQt6 애플리케이션 구현 계약 / PyQt6 Application Implementation Contract

> **목적 / Purpose**: GUI Worker 스레드, 탭 위젯, src 모듈 연동 API 계약을 정의한다.
> **상태 / Status**: ✅ Accepted [Hard]

> 🔒 **SSOT 경계 원칙**: 의미적 해석이 필요한 경우 [SSOT_GUI.md](../SSOT/SSOT_GUI.md)를 최종 판결자로 따른다.

---

## 1. 진입점 계약 / Entry Point Contract

```python
# gui/main.py
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    font_name = _detect_font()
    app.setFont(QFont(font_name, 10))
    gui_cfg = _load_gui_cfg()           # gui/assets/config.json
    theme = gui_cfg.get("theme", "dark")
    lang  = gui_cfg.get("lang",  "ko")
    set_lang(lang)
    qss = _load_qss(f"{theme}_theme.qss", font=font_name)
    if qss:
        app.setStyleSheet(qss)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

| 항목 | 값 |
| --- | --- |
| 진입점 | `python -m gui.main` 또는 `python gui/main.py` |
| QApplication | 단일 인스턴스 |
| 탭 컨테이너 | `QTabWidget` (7탭) |
| 폰트 | `_detect_font()` — `QFontDatabase.families()` 로 플랫폼별 실존 폰트 탐색 |
| 테마·언어 초기화 | `gui/assets/config.json` 에서 읽어 시작 전 적용 |

---

## 2. Worker 스레드 API 계약 / Worker Thread API Contract

### 2.1 공통 시그널 / Common Signals

```python
class BaseWorker(QThread):
    progress_updated = pyqtSignal(int)    # 0–100
    log_emitted      = pyqtSignal(str)
    finished         = pyqtSignal(dict)
    error_occurred   = pyqtSignal(str)

    def is_cancelled(self) -> bool: ...   # cancel() 호출 여부 확인
    def cancel(self) -> None: ...         # 취소 요청
    def emit_progress(self, pct: int, msg: str) -> None: ...
```

### 2.2 TrainingWorker

```python
class TrainingWorker(BaseWorker):
    def __init__(self, cfg: dict, phase: int, channel: str): ...
    # finished payload:
    # phase=0: {"val_acc": float, "checkpoint": str, "phase": 0, "channel": str}
    # phase=2: {"best_val_acc": float, "test_acc": float, "mae": float,
    #           "checkpoint_path": str, "phase": 2, "channel": str}
```

### 2.3 EvaluationWorker

```python
class EvaluationWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    # finished payload:
    # {
    #     "accuracy":    float,
    #     "macro_f1":    float,   ← 신규
    #     "mae":         float,   ← 신규
    #     "n_samples":   int,     ← 신규
    #     "report_path": str,
    #     "channel":     str,
    # }
```

### 2.4 TuningWorker

```python
class TuningWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, n_trials: int, phase: int = 2): ...
    # finished payload:
    # {
    #     "best_params": dict,
    #     "best_value":  float,
    #     "val_acc":     float,   (optional)
    #     "test_acc":    float,   (optional)
    #     "macro_f1":    float,   (optional)
    #     "mae":         float,   (optional)
    # }
```

### 2.5 EmbeddingWorker

```python
class EmbeddingWorker(BaseWorker):
    def __init__(self, cfg: dict, channel: str, checkpoint_path: str): ...
    # finished payload:
    # {
    #     "embeddings_2d": List[List[float]],   # (N, 2) t-SNE 좌표
    #     "labels":        List[int],
    #     "paths":         List[str],
    #     "channel":       str,
    # }
```

### 2.6 InferenceWorker

```python
class InferenceWorker(BaseWorker):
    def __init__(
        self,
        cfg:             dict,
        image_path:      str,
        checkpoint_path: str,
        channel:         str = "Y",   # "Y"|"M"|"C"|"K"|"all"
    ): ...
    # finished payload (단일 채널):
    # {
    #     "pred_level":  int,
    #     "confidence":  float,
    #     "probs":       List[float],
    #     "top3":        List[Tuple[int, float]],
    #     "image_path":  str,
    #     "channel":     str,
    #     "checkpoint":  str,
    # }
    # finished payload (channel="all", 앙상블):
    # {
    #     "pred_level":    int,
    #     "confidence":    float,
    #     "probs":         List[float],
    #     "top3":          List[Tuple[int, float]],
    #     "per_channel":   {"Y": {"pred": int, "conf": float}, ...},
    #     "channels_used": List[str],
    #     "image_path":    str,
    # }
```

### 2.7 BatchInferenceWorker

```python
class BatchInferenceWorker(BaseWorker):
    def __init__(
        self,
        cfg:             dict,
        folder_path:     str,
        checkpoint_path: str,
        channel:         str = "Y",   # "Y"|"M"|"C"|"K"|"all"
    ): ...
    # log_emitted 에서 "__ROW__<JSON>" 형식으로 실시간 테이블 업데이트
    # finished payload:
    # {
    #     "results":   List[{
    #         "filename": str, "path": str, "pred_level": int,
    #         "confidence": float, "top3": list, "error": str|None
    #     }],
    #     "total":     int,
    #     "succeeded": int,
    #     "failed":    int,
    # }
```

### 2.8 GradCAMWorker (신규)

```python
class GradCAMWorker(BaseWorker):
    def __init__(
        self,
        cfg:             dict,
        image_path:      str,
        checkpoint_path: str,
        channel:         str = "Y",
        target_level:    int | None = None,  # None → argmax(pred)
    ): ...
    # 순수 PyTorch 훅 — 외부 라이브러리 불필요
    # 마지막 Conv2d 레이어에 forward/backward hook 등록
    # finished payload:
    # {
    #     "overlay":    np.ndarray,   # (H, W, 3) RGB, uint8 — 히트맵 오버레이
    #     "cam":        np.ndarray,   # (H, W) 0–1 정규화 히트맵
    #     "pred_level": int,
    #     "confidence": float,
    #     "channel":    str,
    #     "image_path": str,
    # }
```

---

## 3. 탭 위젯 API 계약 / Tab Widget API Contract

### 3.1 공통 인터페이스 / Common Interface

```python
class BaseTab(QWidget):
    def refresh(self) -> None: ...
    def on_worker_finished(self, result: dict) -> None: ...
```

### 3.2 Tab 1: DataTab

```python
class DataTab(BaseTab):
    def refresh(self) -> None:
        # 채널×레벨 샘플 수 재스캔, _status_lbl 업데이트
    def _save_preprocess(self) -> None:
        # image_size, mean/std, split_ratios, labeled_dir → src/config/config.json 저장
```

### 3.3 Tab 2: TrainingTab

```python
class TrainingTab(BaseTab):
    def start_training(self) -> None:
        # _apply_cfg() → phase/channel 읽어 TrainingWorker 생성+시작
    def stop_training(self) -> None: ...
    def save_config(self) -> None:
        # backbone, head, phase0, phase2 파라미터 → src/config/config.json 저장
```

### 3.4 Tab 3: EvaluationTab

```python
class EvaluationTab(BaseTab):
    def start_evaluation(self) -> None:
        # 채널="전체(All)" 시 4채널 순차 평가 (_pending 큐 방식)
        # EvaluationWorker 시작, on_worker_finished 에서 카드·테이블·차트 갱신
    def stop_evaluation(self) -> None: ...
    # finished 시 update: acc_card, f1_card, mae_card, n_card, _ch_table, chart
```

### 3.5 Tab 4: OptunaTab

```python
class OptunaTab(BaseTab):
    def start_tuning(self) -> None: ...
    def stop_tuning(self) -> None: ...
    def save_search_space(self) -> None:
        # 탐색 공간 → src/config/config.json 저장
    def _take_snapshot(self) -> None:
        # 현재 cfg 주요 값을 _before_snapshot 에 저장
    def _render_compare_chart(self, after: dict) -> None:
        # Before/After Plotly 막대 비교 차트 렌더링
```

### 3.6 Tab 5: EmbeddingTab

```python
class EmbeddingTab(BaseTab):
    def start_embedding(self) -> None:
        # 전체(All) 선택 시 4채널 순차 추출 (_pending_channels 큐)
    def stop_embedding(self) -> None: ...
    def save_label_correction(self, path: str, new_level: int) -> None:
        # labels_vN.csv 버전 증가 저장 (N = max existing + 1)
    # on_worker_finished: scatter 갱신, purity 차트 갱신
```

### 3.7 Tab 6: InferenceTab

```python
class InferenceTab(BaseTab):
    def start_single_inference(self) -> None:
        # InferenceWorker(cfg, image_path, ckpt, channel=ch) 시작
        # 완료 후 GradCAM 버튼 활성화 (단일 채널일 때만)
    def start_batch_inference(self) -> None:
        # BatchInferenceWorker 시작
        # 완료 후 _level_acc_table.update_from_results() 호출
    def _start_gradcam(self) -> None:
        # GradCAMWorker 시작 → overlay np.ndarray → QPixmap 변환 → 레이블 표시
    def _export_csv(self) -> None: ...
    def refresh(self) -> None:
        # gui/assets/config.json 에서 checkpoint_path 갱신
```

### 3.8 Tab 7: SettingsTab

```python
class SettingsTab(BaseTab):
    def get_checkpoint_path(self) -> str: ...
    def save_settings(self) -> None:
        # src/config/config.json + gui/assets/config.json 동시 저장
        # 저장 대상: storage, phase2, phase0, train, system(device, timeout 등)
    # _build_worker_settings_group(): device, num_workers, infer_batch_size, train_timeout
```

---

## 4. GUI ↔ src 의존성 목록 / GUI ↔ src Dependency List

| GUI 모듈 | 사용하는 src API |
| --- | --- |
| `TrainingWorker` | `run_phase0`, `run_phase2` |
| `EvaluationWorker` | `run_evaluate` (→ accuracy, macro_f1, mae 포함 JSON 반환) |
| `TuningWorker` | `run_optuna` |
| `EmbeddingWorker` | `build_model`, t-SNE 변환 |
| `InferenceWorker` | `build_model`, `_IMAGENET_NORMALIZE` (SSOT-NM01) |
| `BatchInferenceWorker` | `build_model`, `_IMAGENET_NORMALIZE` |
| `GradCAMWorker` | `build_model`, `_IMAGENET_NORMALIZE`, PyTorch hook API |
| `_ckpt_utils` | `build_model` (공유 헬퍼) |
| `Tab 1 (Data)` | `src/config/config.json` 직접 R/W |
| `Tab 5 (Embedding)` | `LabelRefiner` |
| 모든 탭 | `load_config()` |

---

## 5. 금지 패턴 / Prohibited Patterns

```python
# ❌ Worker에서 UI 직접 수정 금지
self.progress_bar.setValue(50)

# ✅ 시그널로만 전달
self.progress_updated.emit(50)

# ❌ UI 스레드에서 블로킹 작업 금지
result = trainer.run()

# ✅ Worker에 위임
self.worker = TrainingWorker(cfg, phase=2, channel="Y")
self.worker.start()

# ❌ _cancel_requested 직접 참조 금지
if self._cancelled: ...

# ✅ is_cancelled() 사용
if self.is_cancelled(): ...
```

---

## 6. 체크리스트 / Checklist

- [ ] 새 Worker 추가 시 §2 계약 및 SSOT_GUI.md §3 테이블 갱신
- [ ] 새 탭 추가 시 `BaseTab` 인터페이스 상속 확인
- [ ] Worker → UI 직접 접근 코드 리뷰 시 §5 금지 패턴 확인
- [ ] EvaluationWorker finished dict에 macro_f1, mae 포함 여부 확인
- [ ] GradCAMWorker overlay가 np.ndarray(H,W,3) RGB uint8인지 확인

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | GUI 구조 정의 (What) |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | TrainingWorker 호출 API |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | EvaluationWorker 호출 API |
| [Contract_tuning_boundary.md](Contract_tuning_boundary.md) | TuningWorker 호출 API |
