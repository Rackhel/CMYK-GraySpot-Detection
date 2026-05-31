"""체크포인트 자동 탐지 + 앙상블 추론 공용 헬퍼.
Shared helpers for auto-finding best checkpoints and running ensemble inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CHANNELS = ["Y", "M", "C", "K"]

# 체크포인트 탐색 우선순위 패턴 (채널명 치환)
_CKPT_PATTERNS = [
    "best_{ch}.pt",
    "phase2_{ch}_effb0_v1.pt",
    "phase2_{ch}_res50_v1.pt",
    "phase2_{ch}_effb0_v1.pth",
]

# 탐색 디렉토리 우선순위
_SEARCH_DIRS_RELATIVE = [
    "data_set/models",
    "outputs/checkpoints",
    "data_set/baseline",
]


def auto_find_checkpoint(cfg: dict, channel: str) -> str:
    """cfg의 models_dir + 기본 경로에서 채널별 최적 체크포인트를 탐색한다.
    Searches models_dir and default paths for the best checkpoint for a channel.

    Returns:
        str — 절대 경로 문자열, 없으면 빈 문자열
    """
    models_dir = cfg.get("storage", {}).get("models_dir", "data_set/models")

    search_dirs = [Path(models_dir)]
    for rel in _SEARCH_DIRS_RELATIVE:
        p = _ROOT / rel
        if p not in search_dirs:
            search_dirs.append(p)

    for d in search_dirs:
        for pattern in _CKPT_PATTERNS:
            candidate = d / pattern.replace("{ch}", channel)
            if candidate.exists():
                return str(candidate)
    return ""


def auto_find_all_checkpoints(cfg: dict) -> dict[str, str]:
    """4개 채널(Y/M/C/K) 각각에 대해 auto_find_checkpoint를 실행한다.
    Returns dict: {channel: path_or_empty}
    """
    return {ch: auto_find_checkpoint(cfg, ch) for ch in _CHANNELS}


def run_ensemble(
    cfg: dict,
    tensor,  # torch.Tensor (1, 3, H, W) — 이미 전처리 완료
    ckpt_paths: dict[str, str],
    device,
) -> dict[str, Any]:
    """4개 채널 모델을 모두 로드해 소프트맥스 확률의 평균을 구한다.
    Loads all channel models and averages their softmax probabilities.

    Args:
        ckpt_paths: {channel: checkpoint_path} — 빈 문자열인 채널은 건너뜀
        tensor:     전처리된 입력 텐서 (1, 3, H, W)
        device:     torch.device

    Returns:
        {
            "pred_level":  int,
            "confidence":  float,
            "probs":       List[float],       # 평균 확률
            "top3":        List[(level, prob)],
            "per_channel": {ch: {"pred": int, "conf": float}},
            "channels_used": List[str],
        }
    """
    import torch
    import torch.nn.functional as F

    from src.utils.utils_model import build_model

    all_probs = []
    per_channel: dict[str, dict] = {}
    channels_used: list[str] = []

    for ch, ckpt in ckpt_paths.items():
        if not ckpt:
            continue
        try:
            model = build_model(cfg, Path(ckpt), device)
            model.eval()
            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = F.softmax(logits, dim=1)[0]
            all_probs.append(probs)
            channels_used.append(ch)
            per_channel[ch] = {
                "pred": int(torch.argmax(probs).item()),
                "conf": float(probs.max().item()),
            }
        except Exception:
            pass  # 로드 실패 채널은 건너뜀

    if not all_probs:
        raise RuntimeError(
            "앙상블: 로드된 채널 모델이 없습니다 / No channel model could be loaded"
        )

    avg_probs = torch.stack(all_probs).mean(dim=0)
    probs_list = avg_probs.cpu().tolist()
    pred_level = int(torch.argmax(avg_probs).item())
    confidence = float(avg_probs[pred_level])
    sorted_idx = sorted(
        range(len(probs_list)), key=lambda i: probs_list[i], reverse=True
    )
    top3 = [(i, probs_list[i]) for i in sorted_idx[:3]]

    return {
        "pred_level": pred_level,
        "confidence": confidence,
        "probs": probs_list,
        "top3": top3,
        "per_channel": per_channel,
        "channels_used": channels_used,
    }
