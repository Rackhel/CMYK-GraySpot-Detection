"""
tests/unit/test_utils_config.py

utils/utils_config.py 단위 테스트.
Unit tests for utils/utils_config.py.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from utils.utils_config import (_resolve_device, create_directories,
                                get_nested, load_config, validate_config)

# ── get_nested ──────────────────────────────────────────────────────────────


class TestGetNested:
    def test_top_level_key(self):
        cfg = {"a": 1}
        assert get_nested(cfg, "a") == 1

    def test_nested_key(self):
        cfg = {"a": {"b": {"c": 42}}}
        assert get_nested(cfg, "a.b.c") == 42

    def test_missing_key_returns_default(self):
        cfg = {"a": {"b": 1}}
        assert get_nested(cfg, "a.c", default=0) == 0

    def test_missing_key_default_none(self):
        cfg = {}
        assert get_nested(cfg, "x.y") is None

    def test_intermediate_non_dict_returns_default(self):
        cfg = {"a": 5}
        assert get_nested(cfg, "a.b", default="fallback") == "fallback"

    def test_zero_value_not_treated_as_missing(self):
        cfg = {"a": {"b": 0}}
        assert get_nested(cfg, "a.b", default=99) == 0

    def test_false_value_not_treated_as_missing(self):
        cfg = {"a": {"b": False}}
        assert get_nested(cfg, "a.b", default=True) is False


# ── _resolve_device ─────────────────────────────────────────────────────────


class TestResolveDevice:
    def test_cpu_returns_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_auto_returns_valid_device(self):
        result = _resolve_device("auto")
        assert result in ("cuda", "mps", "cpu")

    def test_auto_uppercase_normalized(self):
        result = _resolve_device("AUTO")
        assert result in ("cuda", "mps", "cpu")

    def test_invalid_device_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid system.device"):
            _resolve_device("tpu")

    def test_cuda_unavailable_raises_or_fallback(self):
        import torch

        if not torch.cuda.is_available():
            # CUDA 없으면 RuntimeError 또는 MPS/CPU fallback
            try:
                result = _resolve_device("cuda")
                assert result in ("mps", "cpu")
            except RuntimeError:
                pass  # 명시적 에러도 허용

    def test_mps_unavailable_raises(self):
        import torch

        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            with pytest.raises(RuntimeError):
                _resolve_device("mps")


# ── load_config ─────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_loads_and_returns_dict(self, minimal_config_file, tmp_path):
        cfg = load_config(config_path=minimal_config_file, root_dir=tmp_path)
        assert isinstance(cfg, dict)

    def test_device_resolved_from_auto(self, minimal_config_file, tmp_path):
        cfg = load_config(config_path=minimal_config_file, root_dir=tmp_path)
        assert cfg["system"]["device"] in ("cuda", "mps", "cpu")

    def test_storage_paths_become_absolute(self, minimal_config_file, tmp_path):
        cfg = load_config(config_path=minimal_config_file, root_dir=tmp_path)
        assert Path(cfg["storage"]["models_dir"]).is_absolute()

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(config_path=tmp_path / "nonexistent.json", root_dir=tmp_path)

    def test_device_name_set_after_load(self, minimal_config_file, tmp_path):
        cfg = load_config(config_path=minimal_config_file, root_dir=tmp_path)
        assert "device_name" in cfg["system"]

    def test_device_count_set_after_load(self, minimal_config_file, tmp_path):
        cfg = load_config(config_path=minimal_config_file, root_dir=tmp_path)
        assert cfg["system"]["device_count"] >= 1


# ── validate_config ─────────────────────────────────────────────────────────


class TestValidateConfig:
    def test_valid_config_returns_true(self, minimal_cfg):
        assert validate_config(minimal_cfg) is True

    def test_missing_data_channels_returns_false(self, minimal_cfg):
        del minimal_cfg["data"]["channels"]
        assert validate_config(minimal_cfg) is False

    def test_missing_model_backbone_returns_false(self, minimal_cfg):
        del minimal_cfg["model"]["backbone"]
        assert validate_config(minimal_cfg) is False

    def test_num_levels_less_than_2_returns_false(self, minimal_cfg):
        minimal_cfg["data"]["num_levels"] = 1
        assert validate_config(minimal_cfg) is False

    def test_zero_learning_rate_returns_false(self, minimal_cfg):
        minimal_cfg["phase2"]["learning_rate"] = 0
        assert validate_config(minimal_cfg) is False

    def test_negative_learning_rate_returns_false(self, minimal_cfg):
        minimal_cfg["phase0"]["learning_rate"] = -1e-3
        assert validate_config(minimal_cfg) is False


# ── create_directories ──────────────────────────────────────────────────────


class TestCreateDirectories:
    def test_creates_all_storage_dirs(self, minimal_cfg, tmp_path):
        minimal_cfg["storage"] = {
            "data_root": str(tmp_path / "data_set"),
            "labeled_dir": str(tmp_path / "data_set/labeled"),
            "models_dir": str(tmp_path / "data_set/models"),
            "reports_dir": str(tmp_path / "data_set/reports"),
            "logs_dir": str(tmp_path / "outputs/logs"),
        }
        create_directories(minimal_cfg)
        assert (tmp_path / "data_set/models").exists()
        assert (tmp_path / "outputs/logs").exists()

    def test_idempotent_when_dirs_already_exist(self, minimal_cfg, tmp_path):
        minimal_cfg["storage"] = {"models_dir": str(tmp_path / "models")}
        create_directories(minimal_cfg)
        create_directories(minimal_cfg)  # 두 번 호출해도 에러 없음
        assert (tmp_path / "models").exists()
