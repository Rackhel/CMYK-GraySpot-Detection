"""
utils/utils_config.py

설정 관련 공용 유틸리티 / Config-related shared utilities.

config.json 로드, 경로 해석, 디바이스 감지, 검증을 담당한다.
Handles config.json loading, path resolution, device detection, and validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

# ── 경로 상수 / Path constants ─────────────────────────────────────────────────
_UTILS_DIR = Path(__file__).resolve().parent  # src/utils/
_SRC_DIR = _UTILS_DIR.parent  # src/
_ROOT_DIR = _SRC_DIR.parent  # CMYK_MAIN/
_CONFIG_PATH = _SRC_DIR / "config" / "config.json"


# ──────────────────────────────────────────────────────────────────────────────
# 점 표기법 접근 / Dot-notation accessor
# ──────────────────────────────────────────────────────────────────────────────


def get_nested(cfg: dict, key: str, default: Any = None) -> Any:
    """
    점 표기법으로 중첩 dict 값을 조회한다.
    Retrieves a nested dict value using dot notation.

    Example:
        get_nested(cfg, "phase2.learning_rate")   # → 1e-4
        get_nested(cfg, "logging.level", "INFO")  # → "INFO" if missing
    """
    value = cfg
    for k in key.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(k)
        if value is None:
            return default
    return value


# ──────────────────────────────────────────────────────────────────────────────
# 내부 처리 함수 / Internal processing functions
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_device(raw: str) -> str:
    """
    "auto" | "cuda" | "mps" | "cpu" 문자열을 실제 사용 가능한 디바이스로 변환한다.
    Resolves device config string to an actually available device.
    """
    val = raw.lower()
    gpu = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if val == "auto":
        return "cuda" if gpu else ("mps" if mps else "cpu")
    if val == "cuda":
        if gpu:
            return "cuda"
        if mps:
            return "mps"
        raise RuntimeError("CUDA requested but not available. Use 'auto' or 'cpu'.")
    if val == "mps":
        if mps:
            return "mps"
        raise RuntimeError("MPS requested but not available. Use 'auto' or 'cpu'.")
    if val == "cpu":
        return "cpu"
    raise ValueError(
        f"Invalid system.device value '{raw}'. Allowed: auto, cuda, mps, cpu."
    )


def _resolve_paths(cfg: dict, root_dir: Path) -> None:
    """storage.* 의 상대 경로를 절대 경로로 변환한다 (in-place)."""
    storage = cfg.get("storage", {})
    for key in ("data_root", "labeled_dir", "models_dir", "reports_dir", "logs_dir"):
        if key in storage:
            storage[key] = str(root_dir / storage[key])


def _setup_device(cfg: dict) -> None:
    """system.device 를 실제 디바이스로 해석하고 부가 정보를 추가한다 (in-place)."""
    sys_cfg = cfg.setdefault("system", {})
    selected = _resolve_device(sys_cfg.get("device", "auto"))
    sys_cfg["device"] = selected

    if selected == "cuda":
        sys_cfg["device_name"] = f"cuda:{torch.cuda.current_device()}"
        sys_cfg["device_count"] = torch.cuda.device_count()
    elif selected == "mps":
        sys_cfg["device_name"] = "mps"
        sys_cfg["device_count"] = 1
    else:
        sys_cfg["device_name"] = "cpu"
        sys_cfg["device_count"] = 1


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API / Public API
# ──────────────────────────────────────────────────────────────────────────────


def load_config(
    config_path: Optional[Path] = None,
    root_dir: Optional[Path] = None,
) -> dict:
    """
    config.json을 로드하고 처리된 dict를 반환한다.
    Loads config.json and returns a fully processed dict.

    처리 내용 / Processing:
        1. JSON 파일 로드
        2. storage.* 상대 경로 → 절대 경로 변환
        3. system.device "auto" → 실제 디바이스 이름으로 변환

    Args:
        config_path: config.json 경로 (None이면 기본값 사용)
        root_dir:    프로젝트 루트 (None이면 자동 감지)

    Returns:
        처리된 설정 dict / Processed config dict
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    root = Path(root_dir) if root_dir else _ROOT_DIR

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Please create src/config/config.json from the project template."
        )

    with open(path, encoding="utf-8") as fp:
        cfg = json.load(fp)

    _resolve_paths(cfg, root)
    _setup_device(cfg)
    return cfg


def validate_config(cfg: dict) -> bool:
    """
    필수 설정 항목을 검증한다.
    Validates required configuration fields.

    Returns:
        True if valid, False otherwise
    """
    required = [
        ("data", "channels"),
        ("data", "num_levels"),
        ("model", "backbone"),
        ("phase2", "epochs"),
    ]
    for section, field in required:
        if field not in cfg.get(section, {}):
            print(f"[CONFIG ERROR] Missing required field: '{section}.{field}'")
            return False

    if cfg["data"]["num_levels"] < 2:
        print("[CONFIG ERROR] data.num_levels must be >= 2")
        return False

    for phase in ("phase0", "phase2"):
        if phase in cfg:
            lr = cfg[phase].get("learning_rate", 0)
            if lr <= 0:
                print(f"[CONFIG ERROR] {phase}.learning_rate must be > 0")
                return False

    return True


def create_directories(cfg: dict) -> None:
    """
    설정에 정의된 필수 디렉토리를 생성한다.
    Creates necessary directories defined in config.
    """
    storage = cfg.get("storage", {})
    for key in ("data_root", "labeled_dir", "models_dir", "reports_dir", "logs_dir"):
        if key in storage:
            Path(storage[key]).mkdir(parents=True, exist_ok=True)
