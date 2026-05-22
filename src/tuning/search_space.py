import optuna


def get_phase0_search_space(trial: optuna.Trial, cfg: dict = None) -> dict:
    """
    Phase 0 SimCLR Contrastive Learning search space.

    cfg["optuna"]["phase0"]["search_space"] 에서 범위를 읽는다.
    Reads from cfg["optuna"]["phase0"]["search_space"].

    탐색 대상 / Search targets:
        - learning_rate, weight_decay, batch_size, epochs

    Hard SSOT (탐색 제외) / Not tuned (Hard SSOT):
        - temperature, projection_dim, hidden_dim (모델 구조 고정 / fixed model structure)

    Args:
        trial: Optuna trial object
        cfg:   config.json dict (선택사항 / optional)

    Returns:
        dict of sampled hyperparameters
    """
    if cfg is None:
        cfg = {}

    ss = cfg.get("optuna", {}).get("phase0", {}).get("search_space", {})

    lr_range = ss.get("learning_rate", [1e-4, 1e-2])
    wd_range = ss.get("weight_decay", [1e-6, 1e-4])
    bs_opts = ss.get("batch_size", [16, 32, 64])
    ep_range = ss.get("epochs", [5, 15])

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", lr_range[0], lr_range[1], log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", wd_range[0], wd_range[1], log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", bs_opts),
        "epochs": trial.suggest_int("epochs", ep_range[0], ep_range[1]),
    }


def get_phase2_search_space(trial: optuna.Trial, cfg: dict = None) -> dict:
    """
    Phase 2 supervised learning search space — backbone별 분리 탐색 공간.
    Phase 2 supervised learning search space — backbone-specific separated search space.

    cfg["optuna"]["search_space"][backbone_name] 에서 범위를 읽는다.
    backbone별 항목이 없으면 최상위 search_space → 기본값 순으로 fallback한다.
    Reads from cfg["optuna"]["search_space"][backbone_name];
    falls back to top-level search_space then defaults.

    Args:
        trial: Optuna trial object
        cfg:   config.json dict (선택사항 / optional)

    Returns:
        dict of sampled hyperparameters (backbone에 따라 mid_dim 포함 여부 다름)
        dict of sampled hyperparameters (mid_dim included only for resnet50)
    """
    if cfg is None:
        cfg = {}

    backbone_name = cfg.get("model", {}).get("backbone", "efficientnet_b0")
    # optuna.phase2.search_space 우선, 하위 호환을 위해 optuna.search_space fallback
    # Prefer optuna.phase2.search_space; fall back to optuna.search_space for backward compat
    ss_root = cfg.get("optuna", {}).get("phase2", {}).get(
        "search_space", cfg.get("optuna", {}).get("search_space", {})
    )

    # backbone별 탐색 공간 우선, 없으면 최상위 fallback
    # Backbone-specific space first, then top-level fallback
    ss = ss_root.get(backbone_name, ss_root)

    lr_range = ss.get("learning_rate", [1e-5, 1e-2])
    wd_range = ss.get("weight_decay", [1e-6, 1e-3])
    bs_opts = ss.get("batch_size", [4, 16, 32, 64])
    ep_range = ss.get("epochs", [1, 10, 30])
    do_range = ss.get("dropout", [0.0, 0.5])
    hd_opts = ss.get("hidden_dim", [128, 256])

    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", lr_range[0], lr_range[1], log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", bs_opts),
        "weight_decay": trial.suggest_float(
            "weight_decay", wd_range[0], wd_range[1], log=True
        ),
        "epochs": trial.suggest_int("epochs", ep_range[0], ep_range[1]),
        "dropout": trial.suggest_float("dropout", do_range[0], do_range[1]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", hd_opts),
    }

    # ResNet-50 전용: 중간 압축 차원 탐색 / ResNet-50 only: intermediate compression dim
    if backbone_name == "resnet50":
        md_opts = ss.get("mid_dim", [256, 512, 1024])
        params["mid_dim"] = trial.suggest_categorical("mid_dim", md_opts)

    return params
