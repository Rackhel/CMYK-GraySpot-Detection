"""
utils/optuna_utils.py

Optuna 산출물 저장·로드 유틸리티 / Optuna artifact save/load utilities.

SRP 준수: 이 모듈은 Optuna 산출물 I/O만 담당한다.
SRP compliant: handles Optuna artifact I/O only.

Contract:
    - normalize_channel(channel) → str (lowercase)
    - load_best_params(channel, output_dir) → dict
    - save_best_params(params, channel, output_dir) → Path
    - save_trials_summary(trials, channel, output_dir) → Path
    - apply_phase0_params(cfg, params) → dict
    - apply_phase2_params(cfg, params) → dict
    - resolve_n_jobs(cfg) → int

Fail-Fast (SSOT-FF01):
    - FileNotFoundError if best_params file does not exist
    - ValueError if channel is invalid
    - KeyError if required param keys are missing
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"

VALID_CHANNELS = {"Y", "M", "C", "K", "ALL"}


def normalize_channel(channel: str) -> str:
    """
    채널 문자열을 소문자 파일명 접미사로 정규화한다.
    Normalizes channel string to lowercase filename suffix.

    Args:
        channel: Y / M / C / K / all (대소문자 무관 / case-insensitive)

    Returns:
        소문자 접미사 / Lowercase suffix (e.g. "y", "all")

    Raises:
        ValueError: 지원하지 않는 채널 / Unsupported channel
    """
    ch = channel.upper()
    if ch not in VALID_CHANNELS:
        raise ValueError(
            f"Unsupported channel: '{channel}'. Available: Y, M, C, K, all"
        )
    return ch.lower()


def load_best_params(channel: str, output_dir: str | Path = "outputs/optuna") -> dict:
    """
    저장된 최적 Optuna 파라미터를 로드한다.
    Loads saved best Optuna parameters for a channel.

    Args:
        channel:    Y / M / C / K / all
        output_dir: Optuna 산출물 디렉토리 / Optuna output directory

    Returns:
        최적 파라미터 dict / Best parameter dict

    Raises:
        FileNotFoundError: 파일 미존재 시 (SSOT-FF01)
        ValueError:        유효하지 않은 채널
    """
    suffix = normalize_channel(channel)
    path = Path(output_dir) / f"best_params_{suffix}.json"

    if not path.exists():
        raise FileNotFoundError(f"[SSOT-FF01] Best params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_best_params(
    params: dict, suffix: str, output_dir: str | Path = "outputs/optuna"
) -> Path:
    """
    최적 파라미터를 JSON 파일로 저장한다.
    Saves best parameters to a JSON file.

    Args:
        params:     저장할 파라미터 dict / Parameter dict to save
        suffix:     파일명 접미사 — 채널명 또는 phase 포함 식별자
                    Filename suffix — channel name or phase-prefixed identifier
                    (e.g. "y", "phase2_y", "phase0_m")
        output_dir: 저장 디렉토리 / Output directory

    Returns:
        저장된 파일 경로 / Path to the saved file
    """
    suffix = suffix.lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"best_params_{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return path


def save_trials_summary(
    trials: list, suffix: str, output_dir: str | Path = "outputs/optuna"
) -> Path:
    """
    전체 trial 결과 요약을 JSON 파일로 저장한다.
    Saves all trial results summary to a JSON file.

    Args:
        trials:     optuna.Study.trials 리스트 / optuna.Study.trials list
        suffix:     파일명 접미사 — 채널명 또는 phase 포함 식별자
                    Filename suffix — channel name or phase-prefixed identifier
        output_dir: 저장 디렉토리 / Output directory

    Returns:
        저장된 파일 경로 / Path to the saved file
    """
    suffix = suffix.lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_summary = [
        {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
        }
        for t in trials
    ]

    path = output_dir / f"trials_summary_{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, ensure_ascii=False)

    return path


def apply_phase0_params(cfg: dict, params: dict) -> dict:
    """
    Optuna 최적 파라미터를 phase0 config에 반영한다.
    Applies Optuna best params to the phase0 config section.

    Args:
        cfg:    현재 config dict / Current config dict
        params: Optuna 최적 파라미터 dict / Optuna best params dict

    Returns:
        갱신된 cfg dict / Updated cfg dict

    Raises:
        KeyError: 필수 파라미터 키 누락 / Required param keys missing
    """
    required = {"learning_rate", "weight_decay", "batch_size", "epochs"}
    missing = required - set(params.keys())
    if missing:
        raise KeyError(f"Missing required Phase 0 Optuna params: {sorted(missing)}")

    cfg["phase0"]["learning_rate"] = params["learning_rate"]
    cfg["phase0"]["weight_decay"] = params["weight_decay"]
    cfg["phase0"]["batch_size"] = params["batch_size"]
    cfg["phase0"]["epochs"] = params["epochs"]

    return cfg


def apply_phase2_params(cfg: dict, params: dict) -> dict:
    """
    Optuna 최적 파라미터를 phase2 config에 반영한다.
    Applies Optuna best params to the phase2 config section.

    backbone 불변 조건 / Backbone invariant:
        cfg["model"]["backbone"] 은 이 함수 호출 전후로 변경되지 않는다.
        cfg["model"]["backbone"] must not change before or after this call.

    Args:
        cfg:    현재 config dict / Current config dict
        params: Optuna 최적 파라미터 dict / Optuna best params dict

    Returns:
        갱신된 cfg dict / Updated cfg dict

    Raises:
        KeyError: 필수 파라미터 키 누락 / Required param keys missing
    """
    required = {
        "learning_rate",
        "batch_size",
        "weight_decay",
        "epochs",
        "dropout",
        "hidden_dim",
    }
    missing = required - set(params.keys())
    if missing:
        raise KeyError(f"Missing required Optuna params: {sorted(missing)}")

    cfg["phase2"]["learning_rate"] = params["learning_rate"]
    cfg["phase2"]["batch_size"] = params["batch_size"]
    cfg["phase2"]["weight_decay"] = params["weight_decay"]
    cfg["phase2"]["epochs"] = params["epochs"]

    backbone_name = cfg["model"]["backbone"]
    if "heads" not in cfg["phase2"]:
        cfg["phase2"]["heads"] = {}
    if backbone_name not in cfg["phase2"]["heads"]:
        cfg["phase2"]["heads"][backbone_name] = {}

    cfg["phase2"]["heads"][backbone_name]["dropout"] = params["dropout"]
    cfg["phase2"]["heads"][backbone_name]["hidden_dim"] = params["hidden_dim"]

    if "mid_dim" in params:
        cfg["phase2"]["heads"][backbone_name]["mid_dim"] = params["mid_dim"]

    return cfg


def resolve_n_jobs(cfg: dict) -> int:
    """
    플랫폼에 따라 적합한 n_jobs 값을 config에서 선택한다.
    Selects the appropriate n_jobs from config based on the current platform.

    선택 규칙 / Selection rules:
      - Windows : optuna.n_jobs_windows (기본 4)
                  ThreadPoolExecutor 내에서 fork 없이 안전하게 동작.
                  Safe inside ThreadPoolExecutor — no fork on Windows.
      - macOS   : optuna.n_jobs (기본 1)
                  18 GB RAM 환경에서 동시 trial = 모델 메모리 × N 유발.
                  Concurrent trials × model memory on 18 GB Mac.
      - Linux   : optuna.n_jobs (기본 1)
                  서버 환경이면 값을 올려도 무방.
                  Raise safely on servers with sufficient RAM.

    Args:
        cfg: load_config() 결과 dict / Config dict from load_config()

    Returns:
        유효 스레드 수 (최소 1) / Effective thread count (minimum 1)
    """
    optuna_cfg = cfg.get("optuna", {})
    if _IS_WINDOWS:
        n_jobs = int(optuna_cfg.get("n_jobs_windows", 4))
        label = "Windows (n_jobs_windows)"
        if n_jobs > 1:
            # SQLite + multi-threaded access on Windows can be unstable in this code path.
            # Use a single worker by default to avoid open-file / shared DB issues.
            n_jobs = 1
            label += " → forced 1 for Windows SQLite stability"
    else:
        n_jobs = int(optuna_cfg.get("n_jobs", 1))
        os_name = "macOS" if platform.system() == "Darwin" else "Linux"
        label = f"{os_name} (n_jobs)"

    n_jobs = max(1, n_jobs)
    print(
        f"[Optuna] platform={platform.system()}  effective n_jobs={n_jobs}  source={label}"
    )
    return n_jobs
