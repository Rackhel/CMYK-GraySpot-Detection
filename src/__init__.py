"""
src/__init__.py

CMYK Grayspot Detection System — 최상위 패키지.
CMYK Grayspot Detection System — top-level package.

하위 패키지 / Sub-packages:
    config     : JSON 설정 관리 / JSON configuration management
    data       : 데이터셋 · 전처리 · 증강 / Dataset, preprocessing, augmentation
    models     : Backbone · Head · GrayspotModel
    training   : Phase 0/2 학습기 · 손실 함수 / Trainers and loss functions
    evaluation : 지표 계산 · 리포트 생성 / Metrics and report generation
    inference  : 추론 파이프라인 / Inference pipeline
    reporting  : HTML 보고서 생성 / HTML report generation
    tuning     : Optuna 하이퍼파라미터 튜닝 / Hyperparameter tuning
    utils      : 로거 · 유틸리티 / Logger and utilities
    scripts    : 실행 스크립트 / Execution scripts
"""

__version__ = "0.1.0"
__project__ = "grayspot"
