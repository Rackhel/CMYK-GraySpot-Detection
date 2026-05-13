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
    
    assert (
        output_dir / f"study_{channel.lower()}.db"
    ).exists(), "study DB 파일이 저장되지 않음"

    assert (
        output_dir / f"trials_summary_{channel.lower()}.json"
    ).exists(), "trials_summary JSON 파일이 저장되지 않음"

@pytest.mark.smoke
def test_resnet50_search_space_has_mid_dim(cfg):
    cfg["model"]["backbone"] = "resnet50"

    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "weight_decay": 1e-3,
            "epochs": 10,
            "dropout": 0.4,
            "hidden_dim": 256,
            "mid_dim": 512,
        }
    )

    params = get_phase2_search_space(trial, cfg)

    assert "mid_dim" in params

@pytest.mark.smoke
def test_efficientnet_search_space_has_no_mid_dim(cfg):
    cfg["model"]["backbone"] = "efficientnet_b0"

    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "epochs": 10,
            "dropout": 0.2,
            "hidden_dim": 128,
        }
    )

    params = get_phase2_search_space(trial, cfg)

    assert "mid_dim" not in params

@pytest.mark.smoke
def test_normalize_channel():
    from tuning.optuna_utils import normalize_channel

    assert normalize_channel("Y") == "y"
    assert normalize_channel("M") == "m"
    assert normalize_channel("C") == "c"
    assert normalize_channel("K") == "k"
    assert normalize_channel("all") == "all"

@pytest.mark.smoke
def test_invalid_channel_raises_error():
    from tuning.optuna_utils import normalize_channel

    with pytest.raises(ValueError):
        normalize_channel("Z")

@pytest.mark.smoke
def test_load_best_params_file_not_found():
    from tuning.optuna_utils import load_best_params

    with pytest.raises(FileNotFoundError):
        load_best_params("Y", output_dir="non_existing_dir")

@pytest.mark.smoke
def test_load_best_params_success(tmp_path):
    from tuning.optuna_utils import load_best_params
    import json

    output_dir = tmp_path / "optuna"
    output_dir.mkdir()

    best_params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "epochs": 10,
        "dropout": 0.2,
        "hidden_dim": 128,
    }

    path = output_dir / "best_params_c.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_params, f)

    loaded = load_best_params("C", output_dir=output_dir)

    assert loaded == best_params

@pytest.mark.smoke
def test_load_best_params_has_required_keys(tmp_path):
    from tuning.optuna_utils import load_best_params
    import json

    output_dir = tmp_path / "optuna"
    output_dir.mkdir()

    best_params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "epochs": 10,
        "dropout": 0.2,
        "hidden_dim": 128,
    }

    with open(
        output_dir / "best_params_y.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_params, f)

    loaded = load_best_params("Y", output_dir)

    required_keys = {
        "learning_rate",
        "batch_size",
        "weight_decay",
        "epochs",
        "dropout",
        "hidden_dim",
    }

    assert required_keys.issubset(set(loaded.keys()))

@pytest.mark.smoke
def test_apply_phase2_params_updates_config(cfg):
    from tuning.optuna_utils import apply_phase2_params

    cfg["model"]["backbone"] = "efficientnet_b0"

    params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "epochs": 10,
        "dropout": 0.2,
        "hidden_dim": 128,
    }

    updated = apply_phase2_params(cfg, params)

    assert updated["phase2"]["learning_rate"] == 1e-4
    assert updated["phase2"]["batch_size"] == 16
    assert updated["phase2"]["weight_decay"] == 1e-4
    assert updated["phase2"]["epochs"] == 10
    assert updated["phase2"]["heads"]["efficientnet_b0"]["dropout"] == 0.2
    assert updated["phase2"]["heads"]["efficientnet_b0"]["hidden_dim"] == 128

@pytest.mark.smoke
def test_apply_phase2_params_missing_key_raises_error(cfg):
    from tuning.optuna_utils import apply_phase2_params

    params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
    }

    with pytest.raises(KeyError):
        apply_phase2_params(cfg, params)

@pytest.mark.smoke
def test_apply_phase2_params_updates_resnet50_mid_dim(cfg):
    from tuning.optuna_utils import apply_phase2_params

    cfg["model"]["backbone"] = "resnet50"

    params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "epochs": 10,
        "dropout": 0.4,
        "hidden_dim": 256,
        "mid_dim": 512,
    }

    updated = apply_phase2_params(cfg, params)

    assert updated["phase2"]["heads"]["resnet50"]["dropout"] == 0.4
    assert updated["phase2"]["heads"]["resnet50"]["hidden_dim"] == 256
    assert updated["phase2"]["heads"]["resnet50"]["mid_dim"] == 512

@pytest.mark.smoke
def test_apply_phase2_params_preserves_backbone(cfg):
    from tuning.optuna_utils import apply_phase2_params

    cfg["model"]["backbone"] = "resnet50"

    params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "epochs": 10,
        "dropout": 0.4,
        "hidden_dim": 256,
        "mid_dim": 512,
    }

    updated = apply_phase2_params(cfg, params)

    assert updated["model"]["backbone"] == "resnet50"