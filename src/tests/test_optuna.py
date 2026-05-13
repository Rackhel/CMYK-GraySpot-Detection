"""
Grayspot — Optuna 튜닝 검증 / Optuna Tuning Validation
tests/test_optuna.py

Optuna 하이퍼파라미터 튜닝 파이프라인이 정상적으로 동작하는지 검증한다.
Validates whether the Optuna hyperparameter tuning pipeline works correctly.

검증 항목 / Validation steps:
    1. config 로드 확인 / Verify config load
    2. search_space 반환값 확인 / Verify search space output
    3. objective 단일 실행 확인 / Verify single objective execution
    4. Optuna 미니 실행 (1~2 trial) / Run mini Optuna optimization (1~2 trials)
    5. 결과 파일 저장 확인 / Verify output file save

실행 / Run:
    python -m src.tests.test_optuna
    python -m src.tests.test_optuna --channel C
    python -m src.tests.test_optuna --channel all
    python -m src.tests.test_optuna --channel M --trials 2
"""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import optuna

# CMYK_MAIN 루트를 sys.path에 추가 / Add CMYK_MAIN root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


# -------------------------------------------------------------------
# Compatibility shim for team imports
# 팀 코드(config_manager.py, grayspot_model.py 등)가
# `from utils import ...` 를 사용하므로,
# 현재 실행 환경에서 top-level `utils` 모듈을 직접 등록한다.
# -------------------------------------------------------------------
logger_mod = importlib.import_module("src.utils.logger")

utils_shim = types.ModuleType("utils")
utils_shim.LoggerMixin = logger_mod.LoggerMixin
utils_shim.get_logger = logger_mod.get_logger
utils_shim.setup_logging = logger_mod.setup_logging
utils_shim.log_training_config = logger_mod.log_training_config
utils_shim.log_epoch_summary = logger_mod.log_epoch_summary

sys.modules["utils"] = utils_shim


from src.scripts.run_baseline import load_config
from src.tuning.optuna_tuner import objective, run_optuna
from src.tuning.search_space import get_phase2_search_space

CHANNELS = ["Y", "M", "C", "K"]


# ── 출력 헬퍼 / Output helpers ────────────────────────────────
def pass_(msg):
    print(f"  [PASS] {msg}")


def fail_(msg):
    print(f"  [FAIL] {msg}")


def info_(msg):
    print(f"  [INFO] {msg}")


def section(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ──────────────────────────────────────────────────────────────
# TEST 1. config 로드 확인 / Config Load
# ──────────────────────────────────────────────────────────────
def test_config() -> bool:
    section("TEST 1. config 로드 확인 / Config Load")
    try:
        config = load_config()
        cfg = config.config

        required = ["data", "model", "phase2", "storage", "train"]
        missing = [k for k in required if k not in cfg]

        if missing:
            fail_(f"누락된 키 / Missing keys: {missing}")
            return False

        pass_("config.yaml 로드 성공 / Loaded successfully")
        pass_(
            f"backbone: {cfg['model']['backbone']} | "
            f"channels: {cfg['data']['channels']} | "
            f"device: {config.get('system.device')}"
        )
        return True

    except Exception as e:
        fail_(f"config 로드 오류 / Load error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 2. search_space 반환값 확인 / Search Space
# ──────────────────────────────────────────────────────────────
def test_search_space() -> bool:
    section("TEST 2. search_space 반환값 확인 / Search Space Verification")
    try:
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 1e-4,
                "batch_size": 16,
                "weight_decay": 1e-4,
                "epochs": 10,
            }
        )

        params = get_phase2_search_space(trial)
        required = ["learning_rate", "batch_size", "weight_decay", "epochs"]
        missing = [k for k in required if k not in params]

        if missing:
            fail_(f"누락된 파라미터 / Missing params: {missing}")
            return False

        pass_(f"search_space 반환 성공 / Returned successfully")
        pass_(f"params: {params}")
        return True

    except Exception as e:
        fail_(f"search_space 오류 / Error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 3. objective 단일 실행 확인 / Objective Single Run
# ──────────────────────────────────────────────────────────────
def test_objective(channel: str) -> bool:
    section("TEST 3. objective 단일 실행 확인 / Objective Single Execution")
    try:
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 1e-4,
                "batch_size": 16,
                "weight_decay": 1e-4,
                "epochs": 3,
            }
        )

        score = objective(trial, channel=channel)

        if not isinstance(score, float):
            fail_(
                f"objective 반환값이 float가 아님 / Return type is not float: {type(score)}"
            )
            return False

        pass_(f"objective 실행 성공 / Executed successfully")
        pass_(f"channel: {channel.upper()} | score: {score:.4f}")
        return True

    except Exception as e:
        fail_(f"objective 실행 오류 / Execution error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 4. Optuna 미니 실행 / Mini Optuna Run
# ──────────────────────────────────────────────────────────────
def test_optuna_run(channel: str, trials: int) -> bool:
    section("TEST 4. Optuna 미니 실행 / Mini Optuna Run")
    try:
        run_optuna(n_trials=trials, channel=channel)
        pass_(
            f"Optuna {trials} trial 실행 성공 / {trials} trials completed successfully"
        )
        return True

    except Exception as e:
        fail_(f"Optuna 실행 오류 / Run error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 5. 결과 파일 저장 확인 / Output File Verification
# ──────────────────────────────────────────────────────────────
def test_outputs(channel: str) -> bool:
    section("TEST 5. 결과 파일 저장 확인 / Output File Verification")

    suffix = channel.lower()
    if suffix == "all":
        suffix = "all"

    output_dir = ROOT_DIR / "outputs" / "optuna"
    db_path = output_dir / f"study_{suffix}.db"
    best_params_path = output_dir / f"best_params_{suffix}.json"
    summary_path = output_dir / f"trials_summary_{suffix}.json"

    passed = True

    if db_path.exists():
        pass_(f"DB 파일 확인 / DB file exists: {db_path.name}")
    else:
        fail_(f"DB 파일 없음 / DB file not found: {db_path.name}")
        passed = False

    if best_params_path.exists():
        pass_(f"best_params 확인 / best_params exists: {best_params_path.name}")
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            pass_(f"best_params 로드 성공 / Loaded successfully: {params}")
        except Exception as e:
            fail_(f"best_params 로드 오류 / Load error: {e}")
            passed = False
    else:
        fail_(f"best_params 파일 없음 / File not found: {best_params_path.name}")
        passed = False

    if summary_path.exists():
        pass_(f"summary 확인 / summary exists: {summary_path.name}")
    else:
        fail_(f"summary 파일 없음 / File not found: {summary_path.name}")
        passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grayspot Optuna 튜닝 검증 / Optuna Tuning Validation"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="테스트할 채널 / Channel to test (Y/M/C/K/all, default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="테스트용 trial 수 / Number of trials for test (default: 1)",
    )
    args = parser.parse_args()

    channel = args.channel.upper()
    valid_channels = {"Y", "M", "C", "K", "ALL"}

    if channel not in valid_channels:
        raise ValueError(
            f"지원하지 않는 채널 / Unsupported channel: {args.channel}. "
            f"선택 가능 / Available: Y, M, C, K, all"
        )

    normalized_channel = channel.lower()

    print("=" * 55)
    print("  Grayspot — Optuna 튜닝 검증 / Tuning Validation")
    print(f"  Channel: {channel}")
    print(f"  Trials : {args.trials}")
    print("=" * 55)

    results = {}
    results["config 로드 / Load"] = test_config()
    results["search_space"] = test_search_space()
    results["objective 실행 / Objective"] = test_objective(normalized_channel)
    results["Optuna 미니 실행 / Mini Run"] = test_optuna_run(
        normalized_channel, args.trials
    )
    results["출력 파일 확인 / Outputs"] = test_outputs(normalized_channel)

    # 최종 결과 / Final results
    print(f"\n{'=' * 55}")
    print("  최종 결과 / Final Results")
    print(f"{'=' * 55}")

    all_passed = True
    for name, result in results.items():
        icon = "[PASS]" if result else "[FAIL]"
        print(f"  {icon}  {name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed. Optuna tuning pipeline is ready.")
    else:
        print("  Some tests failed. Fix the issues above before proceeding.")
    print()


if __name__ == "__main__":
    main()
