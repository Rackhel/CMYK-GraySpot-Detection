import optuna


def get_phase2_search_space(trial):
    """
    Phase 2 supervised learning search space
    Phase 2 지도학습 탐색 범위
    """
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [16, 32, 64]
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),
        "epochs": trial.suggest_int(
            "epochs", 10, 30
        ),
    }