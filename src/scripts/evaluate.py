"""
Grayspot -- 평가 실행 스크립트 / Evaluation Execution Script
scripts/evaluate.py

학습 완료 후 테스트셋 기준으로 모델 성능을 평가한다.
Evaluates model performance on the test set after training is complete.

사용법 / Usage:
    # 전체 채널 평가 / Evaluate all channels
    python scripts/evaluate.py

    # 특정 채널 평가 / Evaluate specific channel
    python scripts/evaluate.py --channel Y

    # HTML 리포트 생성 포함 / Include HTML report generation
    python scripts/evaluate.py --report

    # Swing Cycle 번호 지정 / Specify Swing Cycle number
    python scripts/evaluate.py --report --cycle 2
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# 루트를 sys.path에 추가 / Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import GrayspotDataset
from data.dataloader import get_phase2_loaders
from models.grayspot_model import GrayspotModel
from evaluation.metrics import (
    evaluate_all_channels,
    print_evaluation_report,
    save_evaluation_results,
)
from utils.logger import get_eval_logger

CHANNELS = ["Y", "M", "C", "K"]


def load_config(path: str = "src/config/config.yaml") -> dict:
    """config.yaml을 로드한다. / Loads config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_channel(cfg: dict, channel: str, logger) -> dict | None:
    """
    채널 1개에 대해 테스트셋 평가를 실행한다.
    Runs test set evaluation for a single channel.

    Args:
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
        channel: "Y" | "M" | "C" | "K"
        logger:  logger 인스턴스 / Logger instance

    Returns:
        {"y_true": [...], "y_pred": [...]} 또는 None (모델 없을 시) / or None if no model
    """
    model_path = Path(cfg["inference"]["model_dir"]) / f"best_{channel}.pt"

    if not model_path.exists():
        logger.warning(f"[{channel}] 모델 없음 / Model not found: {model_path}")
        return None

    # 테스트 데이터셋 로드 / Load test dataset
    test_ds = GrayspotDataset(cfg, channel, split="test", augment=False)

    if len(test_ds) == 0:
        logger.warning(f"[{channel}] 테스트 데이터 없음 / No test data")
        return None

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["phase2"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # 모델 로드 / Load model
    model = GrayspotModel(cfg, phase=2)
    model.load(model_path)
    model.eval()

    logger.info(f"[{channel}] 평가 시작 / Evaluation started -- {len(test_ds)}개 샘플 / samples")

    # 추론 / Run inference
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, labels, _ in test_loader:
            logits = model(x)
            y_pred.extend(logits.argmax(1).tolist())
            y_true.extend(labels.tolist())

    logger.info(f"[{channel}] 추론 완료 / Inference done -- {len(y_true)}개 / samples")

    return {"y_true": y_true, "y_pred": y_pred}


def run_evaluation(cfg: dict, channels: list[str], logger) -> dict:
    """
    지정된 채널들에 대해 전체 평가를 실행한다.
    Runs full evaluation for the specified channels.

    Args:
        cfg:      config.yaml 딕셔너리 / config.yaml dictionary
        channels: 평가할 채널 목록 / List of channels to evaluate
        logger:   logger 인스턴스 / Logger instance

    Returns:
        evaluate_all_channels() 반환값 / Return value of evaluate_all_channels()
    """
    logger.info("=" * 50)
    logger.info("Phase 3 -- 테스트셋 평가 시작 / Test set evaluation started")
    logger.info("=" * 50)

    results = {}
    for ch in channels:
        result = evaluate_channel(cfg, ch, logger)
        if result:
            results[ch] = result

    if not results:
        logger.error("평가 가능한 채널 없음 / No channels available for evaluation")
        logger.error("train.py를 먼저 실행하세요 / Run train.py first")
        return {}

    # 전체 채널 메트릭 계산 + Swing 판단
    # Compute overall metrics + Swing decision
    eval_result = evaluate_all_channels(results, cfg)

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Grayspot 평가 스크립트 / Evaluation Script")
    parser.add_argument("--channel", type=str, default="all",
                        help="평가 채널 / Channel to evaluate (Y/M/C/K/all)")
    parser.add_argument("--report",  action="store_true",
                        help="HTML 리포트 생성 / Generate HTML report")
    parser.add_argument("--cycle",   type=int, default=1,
                        help="Swing Cycle 번호 / Swing Cycle number (default: 1)")
    parser.add_argument("--config",  type=str, default="src/config/config.yaml",
                        help="config.yaml 경로 / Path to config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = get_eval_logger(cfg)

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print("  Grayspot -- Evaluation")
    print(f"  Channels: {target_channels} | Cycle: {args.cycle}")
    print("=" * 55)

    # 평가 실행 / Run evaluation
    eval_result = run_evaluation(cfg, target_channels, logger)

    if not eval_result:
        sys.exit(1)

    # 터미널 출력 / Print to terminal
    print_evaluation_report(eval_result)

    # CSV + JSON 저장 / Save CSV + JSON
    save_evaluation_results(eval_result, cfg)

    # HTML 리포트 생성 (--report 옵션 시) / Generate HTML report (if --report flag)
    if args.report:
        from reporting.html_report import generate_html_report
        report_path = generate_html_report(eval_result, cfg, cycle=args.cycle)
        logger.info(f"HTML 리포트 생성 / HTML report generated: {report_path}")

    # Swing 판단 결과 출력 / Print Swing decision results
    from utils.logger import log_swing_decision
    log_swing_decision(logger, eval_result["swing_decision"])

    # 종료 코드 / Exit code
    # 목표 달성 시 0, 미달 시 1 반환 (CI/CD 연동 가능)
    # Returns 0 if targets met, 1 if not (compatible with CI/CD)
    if eval_result.get("targets_met"):
        print("\n  All targets met. Model is ready for deployment.")
        sys.exit(0)
    else:
        print("\n  Some targets not met. Check swing_decision for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()