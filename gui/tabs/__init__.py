"""PyQt6 tab package."""

from gui.tabs.base_tab import BaseTab
from gui.tabs.tab_data import DataTab
from gui.tabs.tab_embedding import EmbeddingTab
from gui.tabs.tab_evaluation import EvaluationTab
from gui.tabs.tab_optuna import OptunaTab
from gui.tabs.tab_settings import SettingsTab
from gui.tabs.tab_training import TrainingTab

__all__ = [
    "BaseTab",
    "DataTab",
    "TrainingTab",
    "EvaluationTab",
    "SettingsTab",
    "OptunaTab",
    "EmbeddingTab",
]
