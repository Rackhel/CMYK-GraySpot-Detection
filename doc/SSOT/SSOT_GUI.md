---
type: ssot
domain: gui
status: Active
last_updated: 2026-05-27
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Training_Pipeline.md"
  - "SSOT_Evaluation_Reporting.md"
---

# [SSOT] GUI — PyQt6 애플리케이션 구조 / PyQt6 Application Structure

> **목적 / Purpose**: GUI 탭 구성, Worker 스레드 모델, 시그널/슬롯 패턴을 정의한다. / Define GUI tab layout, Worker thread model, and signal/slot patterns.
> **역할 / Role**: What — GUI 계층의 구조와 책임 경계를 선언한다. / Declare the structure and responsibility boundaries of the GUI layer.

---

## 1. 프레임워크 / Framework

| 항목 / Item | 값 / Value |
| --- | --- |
| 프레임워크 / Framework | PyQt6 |
| 탭 수 / Tab Count | 6개 / 6 tabs |
| 스레드 모델 / Thread Model | QThread 기반 Worker (UI-Non-blocking) / QThread-based Worker (UI-Non-blocking) |
| 차트 렌더링 / Chart Rendering | Plotly (QWebEngineView 임베드 / embedded) |
| 진입점 / Entry Point | `gui/main.py` (`python -m gui.main` 또는 `python gui/main.py`) |

---

## 2. 탭 구조 / Tab Structure

| 탭 번호 / Tab # | 이름 / Name | 역할 / Role | 연결 src 모듈 / Connected src Module |
| --- | --- | --- | --- |
| Tab 1 | Data | 데이터 스캔, 채널×레벨 샘플 수, 이미지 미리보기 / Data scan, channel×level sample count, image preview | `src/data/` |
| Tab 2 | Training | Phase 0/2 모드 선택, 학습 진행 표시, 로그 / Phase 0/2 mode selection, training progress display, logs | `src/training/trainer.py` |
| Tab 3 | Evaluation | Confusion Matrix, 오분류 샘플 뷰어, 메트릭 요약 / Confusion Matrix, misclassified sample viewer, metric summary | `src/evaluation/evaluator.py` |
| Tab 4 | Settings | 모델 체크포인트 선택, config.yaml 파라미터 편집 / Model checkpoint selection, config.yaml parameter editing | `config/` |
| Tab 5 | Optuna HPO | HPO 탐색 진행, trial 현황, best params / HPO search progress, trial status, best params | `src/tuning/optuna_tuner.py` |
| Tab 6 | Embedding | t-SNE scatter, 클러스터링 지표, 라벨 수정 UI / t-SNE scatter, clustering metrics, label correction UI | `src/evaluation/`, `src/data/label_refiner.py` |

---

## 3. Worker 스레드 모델 / Worker Thread Model

| Worker | 역할 / Role | 신호 / Signals (4개) | 생성자 인수 / Constructor Args |
| --- | --- | --- | --- |
| `TrainingWorker(QThread)` | Phase 0 / Phase 2 학습 실행 / Phase 0 / Phase 2 training execution | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `phase`, `channel` |
| `EvaluationWorker(QThread)` | 평가 실행 및 리포트 생성 / Evaluation execution and report generation | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `channel` |
| `TuningWorker(QThread)` | Optuna HPO 실행 / Optuna HPO execution | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg` |
| `EmbeddingWorker(QThread)` | t-SNE 임베딩 추출 / t-SNE embedding extraction | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `channel` |

### 3.1 시그널 규격 / Signal Specification

| 시그널 / Signal | 타입 / Type | 의미 / Meaning |
| --- | --- | --- |
| `progress_updated` | `int` (0–100) | 진행률 / Progress percentage |
| `log_emitted` | `str` | 로그 메시지 / Log message |
| `finished` | `dict` | 완료 결과 / Completion result |
| `error_occurred` | `str` | 오류 메시지 / Error message |

---

## 4. GUI ↔ src 경계 원칙 / GUI ↔ src Boundary Principles

| 원칙 / Principle | 내용 / Detail |
| --- | --- |
| 단방향 의존 / Unidirectional dependency | GUI → src (역방향 금지 / reverse direction prohibited) |
| UI 직접 수정 금지 / No direct UI modification | Worker는 시그널로만 UI 업데이트 / Worker updates UI via signals only |
| 로직 위임 / Logic delegation | GUI는 학습/평가 로직을 직접 구현하지 않는다 / GUI does not implement training/evaluation logic directly |
| Config 경유 / Config pass-through | Worker에 파라미터 전달 시 cfg dict 사용 / Use cfg dict when passing parameters to Worker |

---

## 5. 디렉토리 구조 / Directory Structure

```
gui/
├── main.py                  # QApplication 진입점 (SSOT 공식) / QApplication entry point (canonical)
├── main_window.py           # QMainWindow + QTabWidget (6탭 / 6 tabs)
├── workers/
│   ├── __init__.py
│   ├── base_worker.py       # BaseWorker(QThread) — 공통 시그널 / common signals
│   ├── training_worker.py   # TrainingWorker(QThread)
│   ├── evaluation_worker.py # EvaluationWorker(QThread)
│   ├── inference_worker.py  # InferenceWorker(QThread) — 단일 이미지 추론 / single-image inference
│   ├── tuning_worker.py     # TuningWorker(QThread)
│   └── embedding_worker.py  # EmbeddingWorker(QThread)
├── tabs/
│   ├── __init__.py
│   ├── base_tab.py          # BaseTab(QWidget) — 공통 인터페이스 / common interface
│   ├── tab_data.py          # Tab 1: DataTab(QWidget)
│   ├── tab_training.py      # Tab 2: TrainingTab(QWidget)
│   ├── tab_evaluation.py    # Tab 3: EvaluationTab(QWidget)
│   ├── tab_settings.py      # Tab 4: SettingsTab(QWidget)
│   ├── tab_optuna.py        # Tab 5: OptunaTab(QWidget)
│   └── tab_embedding.py     # Tab 6: EmbeddingTab(QWidget)
├── components/
│   ├── __init__.py
│   ├── plotly_widget.py     # QWebEngineView Plotly 래퍼 / Plotly wrapper
│   ├── image_viewer.py      # 이미지 미리보기 위젯 / Image preview widget
│   ├── log_panel.py         # 로그 출력 패널 / Log output panel
│   ├── metric_card.py       # 지표 카드 위젯 / Metric card widget
│   └── progress_panel.py    # 진행률 바 + 로그 패널 / Progress bar + log panel
├── services/
│   ├── __init__.py
│   ├── training_service.py  # TrainingWorker 생명주기 관리 / lifecycle manager
│   ├── evaluation_service.py
│   ├── tuning_service.py
│   └── embedding_service.py
├── styles/
│   └── dark_theme.qss       # 다크 테마 QSS / Dark theme stylesheet
└── tests/
    └── test_gui_contracts.py # GUI Contract 테스트 / GUI contract tests
```

---

## 6. 탭별 상세 정의 / Per-Tab Detail Definitions

### 6.1 Tab 1: Data
- 데이터셋 디렉토리 스캔: 채널(C/M/Y/K) × 레벨(1~6) 샘플 수 표시 / Dataset directory scan: display sample count per channel (C/M/Y/K) × level (1–6)
- 이미지 미리보기: 선택 샘플 표시 / Image preview: display selected sample

### 6.2 Tab 2: Training
- Phase 선택: Phase 0 (SimCLR) / Phase 2 (Supervised) / Baseline 전체 / Phase selection: Phase 0 (SimCLR) / Phase 2 (Supervised) / Full Baseline
- 채널 선택: Y / M / C / K / All / Channel selection: Y / M / C / K / All
- 진행 표시: epoch progress bar, 실시간 loss/acc 그래프 / Progress display: epoch progress bar, real-time loss/acc graph
- 학습 로그 패널 (ScrollArea) / Training log panel (ScrollArea)
- 학습 시작/중지 버튼 / Start/Stop training buttons

### 6.3 Tab 3: Evaluation
- 채널별 Confusion Matrix (Plotly heatmap) / Per-channel Confusion Matrix (Plotly heatmap)
- 오분류 샘플 뷰어: predicted level vs true level 이미지 비교 / Misclassified sample viewer: predicted level vs true level image comparison
- 메트릭 요약 테이블 (Overall Acc, Per-class F1, MAE) / Metric summary table (Overall Acc, Per-class F1, MAE)
- Swing Efficiency 지표 표시 / Swing Efficiency metric display

### 6.4 Tab 4: Settings
- 모델 체크포인트 파일 선택 (QFileDialog) / Model checkpoint file selection (QFileDialog)
- config.yaml 파라미터 편집 (키-값 폼) / config.yaml parameter editing (key-value form)
- 저장 / 초기화 버튼 / Save / Reset buttons

### 6.5 Tab 5: Optuna HPO
- Study 시작 / 중지 버튼 / Study start / stop buttons
- Trial 진행 현황 테이블 (trial no., value, params) / Trial progress status table (trial no., value, params)
- Best config 적용 버튼 / Apply best config button
- 파라미터 중요도 차트 (Plotly bar) / Parameter importance chart (Plotly bar)

### 6.6 Tab 6: Embedding
- t-SNE scatter plot (채널 선택, 색상=레벨 / channel selection, color=level)
- 클러스터링 품질 지표: ARI, Silhouette Score / Clustering quality metrics: ARI, Silhouette Score
- Priority Score 상위 20% 샘플 하이라이트 / Top 20% Priority Score sample highlight
- 샘플 클릭 → 레벨 재지정 → `labels_vN.csv` 저장 / Click sample → reassign level → save `labels_vN.csv`
- 라벨 버전 선택 드롭다운 (labels_v0 ~ labels_vN) / Label version selection dropdown (labels_v0 ~ labels_vN)

---

## 7. 체크리스트 / Checklist

- [ ] 새 탭 추가 시 §2 테이블 업데이트 / Update §2 table when adding a new tab
- [ ] Worker 추가 시 §3 테이블 및 workers/ 디렉토리 반영 / Reflect in §3 table and workers/ directory when adding a Worker
- [ ] GUI에서 새 src 모듈 사용 시 Contract_gui.md §3 의존성 갱신 / Update Contract_gui.md §3 dependencies when using a new src module from GUI
- [ ] Worker에서 UI 직접 수정 여부 코드 리뷰 시 확인 / Verify during code review that Worker does not directly modify UI

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | 최상위 원칙 / Top-level principles |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | TrainingWorker 호출 API / TrainingWorker call API |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | EvaluationWorker 호출 API / EvaluationWorker call API |
| [SSOT_ROI_Pipeline.md](SSOT_ROI_Pipeline.md) | Tab 6 라벨 정제 연동 / Tab 6 label refinement integration |
