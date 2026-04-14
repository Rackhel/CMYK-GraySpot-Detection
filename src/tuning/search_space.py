import optuna


def get_phase2_search_space(trial):
    """
    Phase2 (Supervised Learning) 하이퍼파라미터 탐색 공간 정의
    """

    return {
        # Learning Rate (로그 스케일)
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),

        # Batch Size (선택형)
        "batch_size": trial.suggest_categorical(
            "batch_size", [16, 32, 64]
        ),

        # Weight Decay
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),

        # Epochs
        "epochs": trial.suggest_int(
            "epochs", 10, 30
        ),
    }