import optuna


def get_phase2_search_space(trial: optuna.Trial, cfg: dict = None) -> dict:
    """
    Phase 2 supervised learning search space.
    Phase 2 지도학습 탐색 범위.

    cfg["optuna"]["search_space"] 에서 범위를 읽는다.
    없으면 기본값을 사용한다.
    Reads ranges from cfg["optuna"]["search_space"]; falls back to defaults.

    Args:
        trial: Optuna trial object
        cfg:   config.json dict (선택사항 / optional)

    Returns:
        dict of sampled hyperparameters
    """
    if cfg is None:
        cfg = {}

    ss = cfg.get("optuna", {}).get("search_space", {})

    lr_range = ss.get("learning_rate", [1e-5, 1e-2])
    wd_range = ss.get("weight_decay",  [1e-6, 1e-3])
    bs_opts  = ss.get("batch_size",    [16, 32, 64])
    ep_range = ss.get("epochs",        [10, 30])

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", lr_range[0], lr_range[1], log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", bs_opts
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", wd_range[0], wd_range[1], log=True
        ),
        "epochs": trial.suggest_int(
            "epochs", ep_range[0], ep_range[1]
        ),
    }
