"""
scripts/__init__.py

실행 스크립트 패키지 / Execution scripts package.

이 패키지는 직접 실행되는 스크립트 모음이며 라이브러리 API를 노출하지 않는다.
This package contains directly-executed scripts and does not expose a library API.

스크립트 목록 / Script list:
    run_phase0.py            : Phase 0 SimCLR Contrastive Learning 실행
    run_phase2.py            : Phase 2 Supervised Classification 실행
    run_baseline.py          : Naive Baseline (Supervised-only) 실행
    run_optuna.py            : Optuna 하이퍼파라미터 튜닝 실행
    train.py                 : 통합 학습 진입점
    generate_baseline_report.py : Baseline HTML 리포트 생성

실행 방법 / How to run:
    python -m src.scripts.run_phase0
    python -m src.scripts.run_phase2
    python -m src.scripts.run_baseline
"""
