"""
config/config_manager.py

설정 파일 관리 및 로드 / Configuration file management and loading.

설정의 로드, 검증, 경로 해석을 담당합니다.
Handles loading, validation, and path resolution of configuration.
"""

"""
config/config_manager.py

설정 파일 관리 및 로드 / Configuration file management and loading.

설정의 로드, 검증, 경로 해석을 담당합니다.
Handles loading, validation, and path resolution of configuration.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch

try:
    from utils.logger import LoggerMixin
except ImportError:
    from src.utils.logger import LoggerMixin  # type: ignore


class ConfigManager(LoggerMixin):
    """
    YAML 설정 파일을 로드하고 관리하는 클래스.
    Loads and manages YAML configuration files.
    
    기능 / Features:
        - YAML 설정 파일 로드 / Load YAML configuration
        - 상대 경로를 절대 경로로 변환 / Convert relative paths to absolute
        - 디바이스 자동 감지 / Auto-detect device (GPU/CPU)
        - 설정 검증 / Validate configuration
        - 경로 생성 / Create necessary directories
    """
    
    def __init__(self, config_path: Optional[Path] = None, root_dir: Optional[Path] = None):
        """
        ConfigManager 초기화 / Initialize ConfigManager.
        
        Args:
            config_path: 설정 파일 경로 / Path to config file
            root_dir: 프로젝트 루트 디렉토리 / Project root directory
        """
        self.root_dir = root_dir or self._find_root_dir()
        self.config_path = config_path or self.root_dir / "src" / "config" / "config.yaml"
        self.config = {}
        self.load_config()
        self._resolve_paths()
        self._setup_device()
        
    def _find_root_dir(self) -> Path:
        """
        프로젝트 루트 디렉토리 찾기 / Find project root directory.
        CMYK_MAIN 폴더를 루트로 인식합니다.
        """
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "src").exists() and (current / "Dockerfile").exists():
                return current
            current = current.parent
        raise RuntimeError("Could not find project root directory with src/ and Dockerfile")
    
    def load_config(self) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드 / Load YAML configuration file.
        
        Returns:
            설정 dict / Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        
        return self.config
    
    def _resolve_paths(self) -> None:
        """
        상대 경로를 절대 경로로 해석 / Resolve relative paths to absolute paths.
        """
        if "storage" not in self.config:
            return
        
        storage = self.config["storage"]
        path_keys = [
            "data_root", "labeled_dir", "raw_dir", "models_dir",
            "reports_dir", "outputs_dir", "logs_dir"
        ]
        
        for key in path_keys:
            if key in storage:
                rel_path = storage[key]
                abs_path = self.root_dir / rel_path
                storage[key] = str(abs_path)
    
    def _setup_device(self) -> None:
        """
        디바이스 설정 / Setup device configuration.
        자동 감지 옵션 지원합니다 / Supports auto-detection.
        """
        if "system" not in self.config:
            self.config["system"] = {}
        
        device_config = self.config["system"].get("device", "auto").lower()
        gpu_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if device_config == "auto":
            if gpu_available:
                selected = "cuda"
            elif mps_available:
                selected = "mps"
            else:
                selected = "cpu"
        elif device_config == "cuda":
            if gpu_available:
                selected = "cuda"
            elif mps_available:
                self.logger.warning(
                    "CUDA requested but unavailable; falling back to macOS MPS as GPU engine."
                )
                selected = "mps"
            else:
                raise RuntimeError("CUDA requested but not available. Use 'auto' or 'cpu'.")
        elif device_config == "cpu":
            selected = "cpu"
        elif device_config == "mps":
            if not mps_available:
                raise RuntimeError("MPS requested but not available. Use 'auto', 'cuda', or 'cpu'.")
            selected = "mps"
        else:
            raise ValueError(
                "Invalid system.device value. Allowed: auto, cuda, cpu, mps."
            )

        self.config["system"]["device"] = selected
        self.config["system"]["device_platform"] = selected
        self.config["system"]["device_engine"] = "gpu" if selected in {"cuda", "mps"} else "cpu"

        if selected == "cuda":
            self.config["system"]["device_name"] = f"cuda:{torch.cuda.current_device()}"
            self.config["system"]["device_count"] = torch.cuda.device_count()
        elif selected == "mps":
            self.config["system"]["device_name"] = "mps"
            self.config["system"]["device_count"] = 1
        else:
            self.config["system"]["device_name"] = "cpu"
            self.config["system"]["device_count"] = 1
    
    def create_necessary_directories(self) -> None:
        """
        필요한 디렉토리 생성 / Create necessary directories.
        """
        if "storage" not in self.config:
            return
        
        storage = self.config["storage"]
        dir_keys = ["data_root", "labeled_dir", "raw_dir", "models_dir",
                   "reports_dir", "outputs_dir", "logs_dir"]
        
        for key in dir_keys:
            if key in storage:
                dir_path = Path(storage[key])
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값 조회 (닷 표기법 지원) / Get configuration value with dot notation.
        
        Example:
            config.get("storage.models_dir")
            config.get("phase2.learning_rate")
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """
        경로값 조회 (Path 객체 반환) / Get path value as Path object.
        
        Args:
            key: 설정 키 (닷 표기법) / Config key with dot notation
            
        Returns:
            Path 객체 / Path object
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Path not found: {key}")
        return Path(value)
    
    def validate(self) -> bool:
        """
        설정 검증 / Validate configuration.
        
        Returns:
            유효성 여부 / True if valid, False otherwise
        """
        # 핵심 설정 검증 / Validate essential configurations
        required_keys = [
            ("data", "channels"),
            ("data", "num_levels"),
            ("model", "backbone"),
            ("phase2", "epochs"),
        ]
        
        for *parent_keys, leaf_key in required_keys:
            value = self.config
            for key in parent_keys:
                value = value.get(key, {})
            
            if leaf_key not in value:
                self.logger.error(f"Missing required config: {'.'.join(parent_keys + [leaf_key])}")
                return False
        
        # 채널 설정 검증 / Validate channels
        if self.config["data"]["num_levels"] < 2:
            self.logger.error("num_levels must be >= 2")
            return False
        
        # 학습률 범위 검증 / Validate learning rate range
        for phase in ["phase0", "phase2"]:
            if phase in self.config:
                lr = self.config[phase].get("learning_rate", 0)
                if lr <= 0:
                    self.logger.error(f"{phase}.learning_rate must be > 0")
                    return False
        
        return True
    
    def save_config(self, output_dir: Optional[Path] = None) -> None:
        """
        설정을 파일로 저장 (실행 기록용) / Save configuration to file for record keeping.
        
        Args:
            output_dir: 저장 디렉토리 / Output directory
        """
        if output_dir is None:
            output_dir = self.get_path("storage.outputs_dir")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 타임스탐프와 함께 저장 / Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"config_snapshot_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        """설정 정보 출력 / Display configuration info."""
        info = []
        info.append(f"ConfigManager (root: {self.root_dir})")
        device_name = self.config['system'].get('device_name', 'unknown')
        device_engine = self.config['system'].get('device_engine', 'unknown')
        info.append(f"  Device: {device_name} ({device_engine})")
        info.append(f"  Backbone: {self.config['model'].get('backbone', 'unknown')}")
        info.append(f"  Channels: {len(self.config['data'].get('channels', []))} " +
                    f"{self.config['data'].get('channels', [])}")
        return "\n".join(info)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access / 딕셔너리 스타일 접근."""
        return self.config.get(key)


def get_config(config_path: Optional[Path] = None, root_dir: Optional[Path] = None) -> ConfigManager:
    """
    설정 로드 헬퍼 함수 / Helper function to load configuration.
    
    Usage:
        config = get_config()
        learning_rate = config.get("phase2.learning_rate")
        models_dir = config.get_path("storage.models_dir")
    """
    return ConfigManager(config_path=config_path, root_dir=root_dir)
