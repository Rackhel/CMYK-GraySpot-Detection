"""
utils/utils_config.py

설정 관련 공용 유틸리티 / Config-related shared utilities.

config.json 로드, 경로 해석, 디바이스 감지, 검증을 담당한다.
Handles config.json loading, path resolution, device detection, and validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch

# ── 경로 상수 / Path constants ─────────────────────────────────────────────────
_UTILS_DIR = Path(__file__).resolve().parent  # src/utils/
_SRC_DIR = _UTILS_DIR.parent  # src/
_ROOT_DIR = _SRC_DIR.parent  # CMYK_MAIN/
_CONFIG_PATH = _SRC_DIR / "config" / "config.json"


# ──────────────────────────────────────────────────────────────────────────────
# 점 표기법 접근 / Dot-notation accessor (base)
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
# 타입별 Typed Accessor — Deep化: 타입 변환 + 범위 검증을 내부로 숨김
# Typed accessors — Deep: hides type coercion and range validation
# ──────────────────────────────────────────────────────────────────────────────


def get_float(
    cfg: dict,
    key: str,
    default: float,
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """
    점 표기법으로 float 값을 조회하고 타입·범위를 검증한다.
    Retrieves a float value via dot notation with type and range validation.

    호출자는 타입 변환·범위 체크 코드를 작성할 필요가 없다.
    Callers do not need to write type conversion or range checking code.

    Example:
        lr = get_float(cfg, "phase2.learning_rate", default=1e-4, min_val=1e-10)
        acc = get_float(cfg, "evaluation.targets.overall_accuracy",
                        default=0.9, min_val=0.0, max_val=1.0)

    Raises:
        TypeError : float 변환 불가 시 / When conversion to float fails
        ValueError: 값이 범위를 벗어난 경우 / When value is out of range
    """
    raw = get_nested(cfg, key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Config '{key}' = {raw!r} cannot be converted to float"
        ) from exc
    if min_val is not None and value < min_val:
        raise ValueError(f"Config '{key}' = {value} is below minimum {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"Config '{key}' = {value} exceeds maximum {max_val}")
    return value


def get_int(
    cfg: dict,
    key: str,
    default: int,
    *,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> int:
    """
    점 표기법으로 int 값을 조회하고 타입·범위를 검증한다.
    Retrieves an int value via dot notation with type and range validation.

    Example:
        epochs = get_int(cfg, "phase2.epochs", default=30, min_val=1)
        levels = get_int(cfg, "data.num_levels", default=6, min_val=2, max_val=10)

    Raises:
        TypeError : int 변환 불가 시 / When conversion to int fails
        ValueError: 값이 범위를 벗어난 경우 / When value is out of range
    """
    raw = get_nested(cfg, key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Config '{key}' = {raw!r} cannot be converted to int") from exc
    if min_val is not None and value < min_val:
        raise ValueError(f"Config '{key}' = {value} is below minimum {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"Config '{key}' = {value} exceeds maximum {max_val}")
    return value


def get_str(
    cfg: dict,
    key: str,
    default: str,
    *,
    allowed: Optional[List[str]] = None,
) -> str:
    """
    점 표기법으로 str 값을 조회하고 허용 값 집합을 검증한다.
    Retrieves a str value via dot notation with allowed-value validation.

    Example:
        backbone = get_str(cfg, "model.backbone", default="efficientnet_b0",
                           allowed=["efficientnet_b0", "resnet50"])

    Raises:
        ValueError: 값이 허용 집합에 없을 경우 / When value is not in allowed set
    """
    raw = get_nested(cfg, key, None)
    value = str(raw) if raw is not None else default
    if allowed is not None and value not in allowed:
        raise ValueError(f"Config '{key}' = '{value}' must be one of {allowed}")
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


# ── 검증 스키마 / Validation schema (Deep化: 규칙을 데이터로 선언) ────────────
# Deep: rules are declared as data, not as imperative code.

# 필수 필드 목록: (section, field) / Required fields: (section, field)
_REQUIRED_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("data", "channels"),
    ("data", "num_levels"),
    ("model", "backbone"),
    ("phase2", "epochs"),
)

# 값 범위 검증 규칙: (section, field, predicate, error_suffix)
# Value range rules: (section, field, predicate, error_suffix)
#   필드가 존재할 때만 검증 / Validated only when the field is present
_VALUE_RULES: Tuple[Tuple[str, str, Callable[[Any], bool], str], ...] = (
    ("data", "num_levels", lambda v: v >= 2, "data.num_levels must be >= 2"),
    ("phase0", "learning_rate", lambda v: v > 0, "phase0.learning_rate must be > 0"),
    ("phase2", "learning_rate", lambda v: v > 0, "phase2.learning_rate must be > 0"),
)


def validate_config(cfg: dict) -> None:
    """
    필수 설정 항목을 검증한다. 실패 시 즉시 ValueError 발생 (Fail-Fast, SSOT-CF01).
    Validates required configuration fields. Raises ValueError on failure (Fail-Fast, SSOT-CF01).

    검증 규칙은 _REQUIRED_FIELDS / _VALUE_RULES 스키마로 선언되어 있다.
    Validation rules are declared in the _REQUIRED_FIELDS / _VALUE_RULES schema.

    Raises:
        ValueError: 필수 섹션·필드 누락 또는 값 범위 위반 시
                    When required section/field is missing or value range is violated
    """
    # 1. 필수 필드 존재 확인 / Check required field existence
    for section, field in _REQUIRED_FIELDS:
        if field not in cfg.get(section, {}):
            raise ValueError(
                f"[CONFIG ERROR / SSOT-CF01] Missing required field: '{section}.{field}'"
            )

    # 2. 값 범위 검증 (필드 존재 시에만) / Range validation (only when field is present)
    for section, field, predicate, error_msg in _VALUE_RULES:
        val = cfg.get(section, {}).get(field)
        if val is not None and not predicate(val):
            raise ValueError(f"[CONFIG ERROR / SSOT-CF01] {error_msg}")

    return None


def create_directories(cfg: dict) -> None:
    """
    설정에 정의된 필수 디렉토리를 생성한다.
    Creates necessary directories defined in config.
    """
    storage = cfg.get("storage", {})
    for key in ("data_root", "labeled_dir", "models_dir", "reports_dir", "logs_dir"):
        if key in storage:
            Path(storage[key]).mkdir(parents=True, exist_ok=True)
