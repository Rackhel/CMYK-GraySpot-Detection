---
type: ssot
domain: gui
status: Active
last_updated: 2026-05-28
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
| 탭 수 / Tab Count | 7개 / 7 tabs |
| 스레드 모델 / Thread Model | QThread 기반 Worker (UI-Non-blocking) / QThread-based Worker (UI-Non-blocking) |
| 차트 렌더링 / Chart Rendering | Plotly (QWebEngineView 임베드 / embedded) |
| 진입점 / Entry Point | `gui/main.py` (`python -m gui.main` 또는 `python gui/main.py`) |
| 다국어 / i18n | `gui/i18n.py` — `t(key)`, `set_lang("ko"\|"en")` |
| 테마 / Theme | `dark_theme.qss` / `light_theme.qss` — `%FONT%` / `%ASSETS%` 런타임 치환 |

---

## 2. 탭 구조 / Tab Structure

| 탭 번호 / Tab # | 이름 / Name | 역할 / Role | 연결 src 모듈 / Connected src Module |
| --- | --- | --- | --- |
| Tab 1 | Data | 데이터셋 스캔·샘플 수·이미지 미리보기·전처리 파라미터 편집 / Dataset scan, sample count, image preview, preprocessing params editor | `src/data/` |
| Tab 2 | Training | Backbone·Head·Phase별 하이퍼파라미터 편집, Phase 0/2 학습 실행, 학습 곡선 / Backbone·Head·phase hyperparams, Phase 0/2 training, learning curves | `src/training/trainer.py` |
| Tab 3 | Evaluation | Accuracy·F1·MAE, 채널별/전체 평가, Confusion Matrix, 채널 비교 테이블 / Accuracy·F1·MAE per channel, Confusion Matrix, channel comparison table | `src/evaluation/evaluator.py` |
| Tab 4 | Optuna HPO | HPO 실행, 탐색 공간 전체 편집, Before/After 비교, 다중 메트릭 카드 / HPO run, full search-space editor, before/after comparison, multi-metric cards | `src/tuning/optuna_tuner.py` |
| Tab 5 | Embedding | t-SNE 다채널 비교, 레벨 순도 분석, 유사 이미지 탐색, 라벨 교정 / Multi-channel t-SNE, level purity, similar-image finder, label correction | `src/evaluation/`, `src/data/label_refiner.py` |
| Tab 6 | Inference | 단일 이미지 추론+GradCAM, 배치 폴더 추론, 레벨별 정확도, CSV 내보내기 / Single inference+GradCAM, batch inference, per-level accuracy, CSV export | `src/utils/utils_model.py`, `src/data/normalize.py` |
| Tab 7 | Settings | 외관(테마·언어), 저장 경로, Worker 설정, Phase 0/2 하이퍼파라미터 / Appearance, storage paths, Worker settings, Phase 0/2 hyperparameters | `src/config/config.json` |

---

## 3. Worker 스레드 모델 / Worker Thread Model

| Worker | 역할 / Role | 신호 / Signals (4개) | 생성자 인수 / Constructor Args |
| --- | --- | --- | --- |
| `TrainingWorker(QThread)` | Phase 0 / Phase 2 학습 실행 | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `phase`, `channel` |
| `EvaluationWorker(QThread)` | 평가 실행 — accuracy, F1, MAE 리포트 생성 | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `channel`, `checkpoint_path` |
| `TuningWorker(QThread)` | Optuna HPO 실행 | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `channel`, `n_trials`, `phase` |
| `EmbeddingWorker(QThread)` | t-SNE 임베딩 추출 | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `channel`, `checkpoint_path` |
| `InferenceWorker(QThread)` | 단일 이미지 추론 (단일 채널 또는 4채널 앙상블) | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `image_path`, `checkpoint_path`, `channel` |
| `BatchInferenceWorker(QThread)` | 폴더 일괄 추론 (단일 채널 또는 앙상블) | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `folder_path`, `checkpoint_path`, `channel` |
| `GradCAMWorker(QThread)` | GradCAM 히트맵 계산 (순수 PyTorch 훅) | `progress_updated`, `log_emitted`, `finished`, `error_occurred` | `cfg`, `image_path`, `checkpoint_path`, `channel`, `target_level` |

### 3.1 시그널 규격 / Signal Specification

| 시그널 / Signal | 타입 / Type | 의미 / Meaning |
| --- | --- | --- |
| `progress_updated` | `int` (0–100) | 진행률 / Progress percentage |
| `log_emitted` | `str` | 로그 메시지 / Log message |
| `finished` | `dict` | 완료 결과 / Completion result |
| `error_occurred` | `str` | 오류 메시지 / Error message |

### 3.2 shared 헬퍼 / Shared Helpers

`gui/workers/_ckpt_utils.py` — `auto_find_checkpoint(cfg, channel)`, `auto_find_all_checkpoints(cfg)`, `run_ensemble(cfg, tensor, ckpt_paths, device)` 를 InferenceWorker, BatchInferenceWorker, GradCAMWorker가 공유한다.

---

## 4. GUI ↔ src 경계 원칙 / GUI ↔ src Boundary Principles

| 원칙 / Principle | 내용 / Detail |
| --- | --- |
| 단방향 의존 / Unidirectional dependency | GUI → src (역방향 금지) |
| UI 직접 수정 금지 / No direct UI modification | Worker는 시그널로만 UI 업데이트 |
| 로직 위임 / Logic delegation | GUI는 학습/평가 로직을 직접 구현하지 않는다 |
| Config 경유 / Config pass-through | Worker에 파라미터 전달 시 cfg dict 사용 |
| SSOT-NM01 준수 | 모든 이미지 전처리에서 `_IMAGENET_NORMALIZE` 사용 |

---

## 5. 디렉토리 구조 / Directory Structure

```
gui/
├── main.py
├── main_window.py                # QMainWindow + QTabWidget (7탭)
├── i18n.py                       # t(key), set_lang("ko"|"en")
├── workers/
│   ├── __init__.py
│   ├── base_worker.py            # BaseWorker(QThread) — 공통 시그널
│   ├── _ckpt_utils.py            # auto_find_checkpoint, run_ensemble 공유 헬퍼
│   ├── training_worker.py        # TrainingWorker
│   ├── evaluation_worker.py      # EvaluationWorker — accuracy, macro_f1, mae 반환
│   ├── inference_worker.py       # InferenceWorker — channel, ensemble 지원
│   ├── batch_inference_worker.py # BatchInferenceWorker — channel, ensemble 지원
│   ├── gradcam_worker.py         # GradCAMWorker — 순수 PyTorch 훅 GradCAM
│   ├── tuning_worker.py          # TuningWorker
│   └── embedding_worker.py       # EmbeddingWorker
├── tabs/
│   ├── __init__.py
│   ├── base_tab.py               # BaseTab(QWidget)
│   ├── tab_data.py               # Tab 1: DataTab — 전처리 파라미터 편집 포함
│   ├── tab_training.py           # Tab 2: TrainingTab — head·phase 설정·학습 곡선
│   ├── tab_evaluation.py         # Tab 3: EvaluationTab — F1·MAE·채널 비교
│   ├── tab_optuna.py             # Tab 4: OptunaTab — Before/After 비교
│   ├── tab_embedding.py          # Tab 5: EmbeddingTab — 4개 내부 탭
│   ├── tab_inference.py          # Tab 6: InferenceTab — GradCAM·레벨 정확도
│   └── tab_settings.py           # Tab 7: SettingsTab — Worker 설정 포함
├── components/
│   ├── __init__.py
│   ├── plotly_widget.py          # QWebEngineView Plotly 래퍼
│   ├── image_viewer.py
│   ├── log_panel.py
│   ├── metric_card.py
│   ├── progress_panel.py
│   ├── training_chart.py         # 실시간 학습 곡선 위젯 (Plotly 라인 차트)
│   └── level_accuracy_table.py   # 레벨별 정확도 테이블 위젯
├── services/
│   ├── __init__.py
│   ├── training_service.py
│   ├── evaluation_service.py
│   ├── tuning_service.py
│   └── embedding_service.py
├── assets/
│   ├── config.json               # GUI 영속 설정 (theme, lang, checkpoint_path)
│   └── *.svg / *.qss
├── styles/
│   ├── dark_theme.qss
│   └── light_theme.qss
└── tests/
    └── test_gui_contracts.py
```

---

## 6. 탭별 상세 정의 / Per-Tab Detail Definitions

### 6.1 Tab 1: Data
- 채널(Y/M/C/K) × 레벨(0~N-1) 샘플 수 테이블, 스캔 버튼, 타임스탬프·경로 상태 레이블
- 이미지 미리보기: 셀 클릭 또는 Browse로 선택
- **전처리 파라미터 편집 (신규)**: `image_size`, normalize `mean`/`std`, `split_ratios(train/val/test)`, `labeled_dir` 편집 → `src/config/config.json` 저장

### 6.2 Tab 2: Training
- **Backbone 설정**: efficientnet_b0 / resnet50, frozen 체크박스
- **Head 설정 (신규)**: `num_levels` (클래스 수), `dropout_rate`
- **Phase별 파라미터 패널 (신규)**: Phase 2 파라미터(epochs, lr, bs, wd) + Phase 0 파라미터(epochs, lr, bs, temperature, projection_dim) 분리 표시
- **학습 실행**: Phase·채널 선택 → 시작/중지
- **학습 곡선 차트 (신규)**: Plotly 라인 차트 — 로그 파싱 또는 완료 후 결과 막대 차트
- 결과 카드: Best Val Acc, Test Acc, MAE

### 6.3 Tab 3: Evaluation
- **채널 선택**: Y / M / C / K / 전체(All) — 전체 선택 시 4채널 순차 평가 후 평균
- **메트릭 카드 (확장)**: Accuracy, Macro F1, MAE, Samples
- **채널별 비교 테이블 (신규)**: Channel | Accuracy | Macro F1 | MAE 4행 테이블
- Confusion Matrix (Plotly heatmap)
- 단일 이미지 추론 섹션 제거됨 (Tab 6 Inference로 통합)

### 6.4 Tab 4: Optuna HPO
- HPO 실행 컨트롤: 채널·Phase·Trials
- **메트릭 카드 (확장)**: Best Value, Macro F1, MAE, Val Acc, Test Acc
- **Before/After 비교 패널 (신규)**: 📸 스냅샷 저장 버튼 → Before 수치 기록, HPO 완료 후 After 수치 표시, Plotly 막대 비교 차트
- 탐색 공간 편집기 (기존 유지): Global, Phase 0 SS, Phase 2 EfficientNet SS, Phase 2 ResNet SS

### 6.5 Tab 5: Embedding
4개 내부 탭으로 재구성:
- **t-SNE 산점도**: 단일 채널 또는 전체 비교 모드(4채널 오버레이, 채널별 색)
- **레벨 순도 (신규)**: 레벨별 intra-cluster distance 막대 차트 — 낮을수록 응집
- **유사 이미지 탐색 (신규)**: 포인트 선택 후 K-nearest 썸네일 그리드 표시
- **라벨 교정**: 기존 유지 — `labels_vN.csv` 버전 증가 저장

### 6.6 Tab 6: Inference
- **데이터 소스 선택기 (신규)**: raw / roi / labeled — 파일/폴더 다이얼로그 시작 디렉토리 변경
- **채널 선택**: Y / M / C / K / 전체 앙상블(All)
- **체크포인트**: 수동 선택 또는 🔍 자동 탐지
- 단일 이미지 추론: 레벨 배지, 신뢰도 바, Top-3
- **GradCAM 시각화 (신규)**: 🔥 버튼 → GradCAMWorker → 활성화 히트맵 오버레이 표시
- 배치 폴더 추론: 실시간 결과 테이블, CSV 내보내기
- **레벨별 정확도 테이블 (신규)**: 배치 완료 후 Level | Y | M | C | K | Overall 테이블 업데이트

### 6.7 Tab 7: Settings
- 외관: 테마(다크/라이트), 언어(한국어/English)
- 저장 경로: labeled_dir, models_dir, reports_dir, checkpoint
- **Worker 설정 (신규)**: device(auto/cpu/cuda/mps), DataLoader workers, Inference batch size, Training timeout(min)
- Phase 2 하이퍼파라미터 폼
- Phase 0 하이퍼파라미터 폼
- 공통 학습 설정 (seed, scheduler, gradient_clip)
- 저장 → `src/config/config.json` + `gui/assets/config.json`

---

## 7. 체크리스트 / Checklist

- [ ] 새 탭 추가 시 §2 테이블 업데이트
- [ ] Worker 추가 시 §3 테이블 및 workers/ 디렉토리 반영
- [ ] GUI에서 새 src 모듈 사용 시 Contract_gui.md §4 의존성 갱신
- [ ] Worker에서 UI 직접 수정 여부 코드 리뷰 시 확인
- [ ] 모든 이미지 전처리에서 SSOT-NM01 (`_IMAGENET_NORMALIZE`) 사용 확인

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | 최상위 원칙 |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | TrainingWorker 호출 API |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | EvaluationWorker 호출 API |
| [SSOT_ROI_Pipeline.md](SSOT_ROI_Pipeline.md) | Tab 5 라벨 정제 연동 |
