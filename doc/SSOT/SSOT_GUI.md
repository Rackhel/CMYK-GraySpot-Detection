---
type: ssot
domain: gui
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Training_Pipeline.md"
  - "SSOT_Evaluation_Reporting.md"
---

# [SSOT] GUI — PyQt6 애플리케이션 구조

> **목적**: GUI 탭 구성, Worker 스레드 모델, 시그널/슬롯 패턴을 정의한다.
> **역할**: What — GUI 계층의 구조와 책임 경계를 선언한다.

---

## 1. 프레임워크

| 항목 | 값 |
| --- | --- |
| 프레임워크 | PyQt6 |
| 탭 수 | 6개 |
| 스레드 모델 | QThread 기반 Worker (UI-Non-blocking) |
| 차트 렌더링 | Plotly (QWebEngineView 임베드) |
| 진입점 | `gui/main.py` |

---

## 2. 탭 구조

| 탭 번호 | 이름 | 역할 | 연결 src 모듈 |
| --- | --- | --- | --- |
| Tab 1 | Data | 데이터 스캔, 채널×레벨 샘플 수, 이미지 미리보기 | `src/data/` |
| Tab 2 | Training | Phase 0/2 모드 선택, 학습 진행 표시, 로그 | `src/training/trainer.py` |
| Tab 3 | Evaluation | Confusion Matrix, 오분류 샘플 뷰어, 메트릭 요약 | `src/evaluation/evaluator.py` |
| Tab 4 | Settings | 모델 체크포인트 선택, config.yaml 파라미터 편집 | `config/` |
| Tab 5 | Optuna HPO | HPO 탐색 진행, trial 현황, best params | `src/tuning/optuna_tuner.py` |
| Tab 6 | Embedding | t-SNE scatter, 클러스터링 지표, 라벨 수정 UI | `src/evaluation/`, `src/data/label_refiner.py` |

---

## 3. Worker 스레드 모델

| 클래스 | 역할 | 호출 src API |
| --- | --- | --- |
| `TrainingWorker(QThread)` | Phase 0 / Phase 2 학습 실행 | `Phase0Trainer.run()`, `Phase2Trainer.run()` |
| `EvaluationWorker(QThread)` | 평가 실행 및 리포트 생성 | `Evaluator.run()` |
| `TuningWorker(QThread)` | Optuna HPO 실행 | `run_optuna()` |
| `EmbeddingWorker(QThread)` | t-SNE 임베딩 추출 | `GrayspotModel` feature 추출 |

### 3.1 시그널 규격

| 시그널 | 타입 | 의미 |
| --- | --- | --- |
| `progress_updated` | `int` (0–100) | 진행률 |
| `log_emitted` | `str` | 로그 메시지 |
| `finished` | `dict` | 완료 결과 |
| `error_occurred` | `str` | 오류 메시지 |

---

## 4. GUI ↔ src 경계 원칙

| 원칙 | 내용 |
| --- | --- |
| 단방향 의존 | GUI → src (역방향 금지) |
| UI 직접 수정 금지 | Worker는 시그널로만 UI 업데이트 |
| 로직 위임 | GUI는 학습/평가 로직을 직접 구현하지 않는다 |
| Config 경유 | Worker에 파라미터 전달 시 cfg dict 사용 |

---

## 5. 디렉토리 구조

```
gui/
├── main.py                  # QApplication 진입점
├── main_window.py           # QMainWindow + QTabWidget (6탭)
├── workers/
│   ├── __init__.py
│   ├── training_worker.py   # TrainingWorker(QThread)
│   ├── evaluation_worker.py # EvaluationWorker(QThread)
│   ├── tuning_worker.py     # TuningWorker(QThread)
│   └── embedding_worker.py  # EmbeddingWorker(QThread)
├── tabs/
│   ├── __init__.py
│   ├── tab_data.py          # Tab 1: DataTab(QWidget)
│   ├── tab_training.py      # Tab 2: TrainingTab(QWidget)
│   ├── tab_evaluation.py    # Tab 3: EvaluationTab(QWidget)
│   ├── tab_settings.py      # Tab 4: SettingsTab(QWidget)
│   ├── tab_optuna.py        # Tab 5: OptunaTab(QWidget)
│   └── tab_embedding.py     # Tab 6: EmbeddingTab(QWidget)
└── components/
    ├── __init__.py
    ├── plotly_widget.py     # QWebEngineView Plotly 래퍼
    ├── image_viewer.py      # 이미지 미리보기 위젯
    └── log_panel.py         # 로그 출력 패널
```

---

## 6. 탭별 상세 정의

### 6.1 Tab 1: Data
- 데이터셋 디렉토리 스캔: 채널(C/M/Y/K) × 레벨(1~6) 샘플 수 표시
- 이미지 미리보기: 선택 샘플 표시

### 6.2 Tab 2: Training
- Phase 선택: Phase 0 (SimCLR) / Phase 2 (Supervised) / Baseline 전체
- 채널 선택: Y / M / C / K / All
- 진행 표시: epoch progress bar, 실시간 loss/acc 그래프
- 학습 로그 패널 (ScrollArea)
- 학습 시작/중지 버튼

### 6.3 Tab 3: Evaluation
- 채널별 Confusion Matrix (Plotly heatmap)
- 오분류 샘플 뷰어: predicted level vs true level 이미지 비교
- 메트릭 요약 테이블 (Overall Acc, Per-class F1, MAE)
- Swing Efficiency 지표 표시

### 6.4 Tab 4: Settings
- 모델 체크포인트 파일 선택 (QFileDialog)
- config.yaml 파라미터 편집 (키-값 폼)
- 저장 / 초기화 버튼

### 6.5 Tab 5: Optuna HPO
- Study 시작 / 중지 버튼
- Trial 진행 현황 테이블 (trial no., value, params)
- Best config 적용 버튼
- 파라미터 중요도 차트 (Plotly bar)

### 6.6 Tab 6: Embedding
- t-SNE scatter plot (채널 선택, 색상=레벨)
- 클러스터링 품질 지표: ARI, Silhouette Score
- Priority Score 상위 20% 샘플 하이라이트
- 샘플 클릭 → 레벨 재지정 → `labels_vN.csv` 저장
- 라벨 버전 선택 드롭다운 (labels_v0 ~ labels_vN)

---

## 7. 체크리스트

- [ ] 새 탭 추가 시 §2 테이블 업데이트
- [ ] Worker 추가 시 §3 테이블 및 workers/ 디렉토리 반영
- [ ] GUI에서 새 src 모듈 사용 시 Contract_gui.md §3 의존성 갱신
- [ ] Worker에서 UI 직접 수정 여부 코드 리뷰 시 확인

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | 최상위 원칙 |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | TrainingWorker 호출 API |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | EvaluationWorker 호출 API |
| [SSOT_ROI_Pipeline.md](SSOT_ROI_Pipeline.md) | Tab 6 라벨 정제 연동 |
