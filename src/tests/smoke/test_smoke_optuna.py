"""
tests/smoke/test_smoke_optuna.py

Optuna 하이퍼파라미터 튜닝 파이프라인 Smoke 테스트.
Smoke tests for Optuna hyperparameter tuning pipeline.

실행 / Run:
    pytest src/tests/smoke/test_smoke_optuna.py -v -m smoke
"""

import sys
from pathlib import Path

import optuna
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from tests.smoke.conftest import data_exists
from tuning.optuna_tuner import objective, run_optuna
from tuning.search_space import get_phase2_search_space

CHANNELS = ["Y", "M", "C", "K"]


@pytest.mark.smoke
def test_search_space_has_required_params():
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "epochs": 1,
            "dropout": 0.2,
            "hidden_dim": 128,
        }
    )
    params = get_phase2_search_space(trial)

    for key in (
        "learning_rate",
        "batch_size",
        "weight_decay",
        "epochs",
        "dropout",
        "hidden_dim",
    ):
        assert key in params, f"search_space에 누락된 파라미터: {key}"


@pytest.mark.smoke
def test_search_space_value_types():
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "epochs": 1,
            "dropout": 0.2,
            "hidden_dim": 128,
        }
    )
    params = get_phase2_search_space(trial)

    assert isinstance(params["learning_rate"], float)
    assert isinstance(params["batch_size"], int)
    assert isinstance(params["weight_decay"], float)
    assert isinstance(params["epochs"], int)
    assert isinstance(params["dropout"], float)
    assert isinstance(params["hidden_dim"], int)


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_objective_returns_float(cfg, channel):
    if not data_exists(cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음 — objective skip")

    # FIX: Use a real, dynamic trial structure instead of a breaking FixedTrial mock
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()

    score = objective(trial, channel=channel.lower())

    assert isinstance(score, float), f"objective 반환값이 float가 아님: {type(score)}"
    assert 0.0 <= score <= 1.0, f"score 범위 오류: {score}"


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_optuna_mini_run_saves_outputs(cfg, channel):
    if not data_exists(cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음 — Optuna run skip")

    run_optuna(n_trials=1, channel=channel.lower())

    output_dir = ROOT_DIR / "outputs" / "optuna"
    assert (
        output_dir / f"best_params_{channel.lower()}.json"
    ).exists(), "best_params JSON 파일이 저장되지 않음"
