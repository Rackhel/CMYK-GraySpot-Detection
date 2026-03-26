"""
Grayspot -- 평가 실행 및 검증 / Evaluation Execution and Validation
tests/test_evaluation.py

test_training.py 통과 후 실행한다.
Run this after test_training.py passes.

실행 순서 / Execution order:
    1. 저장된 모델 존재 확인 / Verify saved models exist
    2. 테스트셋 추론 실행 / Run test set inference
    3. 메트릭 계산 확인 / Verify metric computation
    4. Confusion Matrix 분석 확인 / Verify Confusion Matrix analysis
    5. 결과 파일 저장 확인 / Verify result file save
    6. Swing 판단 결과 확인 / Verify Swing decision results

실행 / Run:
    python tests/test_evaluation.py

    # HTML 리포트 포함 / Include HTML report
    python tests/test_evaluation.py --report
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import GrayspotDataset
from models.grayspot_model import GrayspotModel
from evaluation.metrics import evaluate_all_channels
from evaluation.confusion import run_confusion_analysis
from evaluation.evaluator import Evaluator
from utils.logger import get_eval_logger


def pass_(msg): print(f"  [PASS] {msg}")
def fail_(msg): print(f"  [FAIL] {msg}")
def info_(msg): print(f"  [INFO] {msg}")
def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# TEST 1. 저장된 모델 존재 확인 / Saved Model Existence
# ──────────────────────────────────────────────
def test_models_exist(cfg: dict) -> bool:
    section("TEST 1. 저장된 모델 확인 / Saved Model Verification")
    model_dir = Path(cfg["inference"]["model_dir"])
    channels  = cfg["data"]["channels"]
    passed    = True

    for ch in channels:
        path = model_dir / f"best_{ch}.pt"
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            pass_(f"[{ch}] best_{ch}.pt -- {size_mb:.1f} MB")
        else:
            fail_(f"[{ch}] best_{ch}.pt 없음 / Not found")
            info_("test_training.py 를 먼저 실행하세요 / Run test_training.py first")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 2. 테스트셋 추론 실행 / Test Set Inference
# ──────────────────────────────────────────────
def test_inference_on_testset(cfg: dict) -> tuple[bool, dict]:
    section("TEST 2. 테스트셋 추론 / Test Set Inference")
    channels = cfg["data"]["channels"]
    results  = {}
    passed   = True

    for ch in channels:
        model_path = Path(cfg["inference"]["model_dir"]) / f"best_{ch}.pt"
        if not model_path.exists():
            info_(f"[{ch}] 모델 없음 -- 건너뜀 / No model -- skipping")
            continue

        test_ds = GrayspotDataset(cfg, ch, split="test", augment=False)
        if len(test_ds) == 0:
            info_(f"[{ch}] 테스트 데이터 없음 -- 건너뜀 / No test data -- skipping")
            continue

        try:
            model = GrayspotModel(cfg, phase=2)
            model.load(model_path)
            model.eval()

            loader         = DataLoader(test_ds, batch_size=cfg["phase2"]["batch_size"],
                                        shuffle=False)
            y_true, y_pred = [], []

            with torch.no_grad():
                for x, labels, _ in loader:
                    logits = model(x)
                    y_pred.extend(logits.argmax(1).tolist())
                    y_true.extend(labels.tolist())

            results[ch] = {"y_true": y_true, "y_pred": y_pred}
            pass_(f"[{ch}] 추론 완료 / Inference done -- {len(y_true)}개 샘플 / samples")

        except Exception as e:
            fail_(f"[{ch}] 추론 오류 / Inference error: {e}")
            passed = False

    if not results:
        fail_("추론 결과 없음 / No inference results")
        passed = False

    return passed, results


# ──────────────────────────────────────────────
# TEST 3. 메트릭 계산 확인 / Metric Computation
# ──────────────────────────────────────────────
def test_metrics(cfg: dict, results: dict) -> tuple[bool, dict]:
    section("TEST 3. 메트릭 계산 / Metric Computation")

    if not results:
        fail_("추론 결과 없음 -- TEST 2 먼저 해결하세요 / No results -- fix TEST 2 first")
        return False, {}

    try:
        eval_result = evaluate_all_channels(results, cfg)
        overall     = eval_result["overall_accuracy"]
        tgt         = cfg["evaluation"]["targets"]

        pass_(f"Overall Accuracy: {overall:.4f} (target >= {tgt['overall_accuracy']:.2f})")

        for ch, m in eval_result["per_channel"].items():
            status = "PASS" if eval_result["swing_decision"][ch] == "pass" else "NEEDS REVIEW"
            pass_(f"[{ch}] Acc: {m['accuracy']:.3f} | F1: {m['macro_f1']:.3f} | "
                  f"MAE: {m['mae']:.3f} | {status}")

        return True, eval_result

    except Exception as e:
        fail_(f"메트릭 계산 오류 / Metric computation error: {e}")
        return False, {}


# ──────────────────────────────────────────────
# TEST 4. Confusion Matrix 분석 / Confusion Matrix Analysis
# ──────────────────────────────────────────────
def test_confusion_analysis(cfg: dict, results: dict) -> bool:
    section("TEST 4. Confusion Matrix 분석 / Confusion Matrix Analysis")

    if not results:
        fail_("추론 결과 없음 -- TEST 2 먼저 해결하세요 / No results -- fix TEST 2 first")
        return False

    passed = True
    for ch, res in results.items():
        try:
            analysis = run_confusion_analysis(
                res["y_true"], res["y_pred"], ch, cfg
            )

            if analysis["needs_phase1"]:
                info_(f"[{ch}] 인접 레벨 혼동 감지 / Adjacent confusion detected -- "
                      f"Phase 1 재진입 권장 / Phase 1 re-entry recommended")
            else:
                pass_(f"[{ch}] 인접 레벨 혼동 없음 / No adjacent confusion")

        except Exception as e:
            fail_(f"[{ch}] Confusion 분석 오류 / Analysis error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 5. 결과 파일 저장 확인 / Result File Save
# ──────────────────────────────────────────────
def test_result_files_saved(cfg: dict, eval_result: dict) -> bool:
    section("TEST 5. 결과 파일 저장 확인 / Result File Save Verification")

    if not eval_result:
        fail_("평가 결과 없음 -- TEST 3 먼저 해결하세요 / No eval result -- fix TEST 3 first")
        return False

    from evaluation.metrics import save_evaluation_results
    save_evaluation_results(eval_result, cfg)

    reports_dir = Path(cfg["storage"]["reports_dir"])
    passed      = True

    check_files = [
        cfg["reporting"]["csv_files"]["evaluation_results"],
        "evaluation_results.json",
    ]

    for filename in check_files:
        path = reports_dir / filename
        if path.exists():
            size_kb = path.stat().st_size / 1024
            pass_(f"{filename} -- {size_kb:.1f} KB")
        else:
            fail_(f"{filename} 없음 / Not found")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 6. Swing 판단 결과 확인 / Swing Decision Verification
# ──────────────────────────────────────────────
def test_swing_decision(cfg: dict, eval_result: dict) -> bool:
    section("TEST 6. Swing 판단 결과 / Swing Decision Verification")

    if not eval_result:
        fail_("평가 결과 없음 -- TEST 3 먼저 해결하세요 / No eval result -- fix TEST 3 first")
        return False

    swing    = eval_result.get("swing_decision", {})
    passed   = True
    valid    = {"pass", "phase0", "phase1"}

    for ch, action in swing.items():
        if action not in valid:
            fail_(f"[{ch}] 유효하지 않은 판단값 / Invalid decision: {action}")
            passed = False
        else:
            pass_(f"[{ch}] --> {action.upper()}")

    targets_met = eval_result.get("targets_met", False)
    if targets_met:
        info_("모든 목표 달성 -- 배포 준비 완료 / All targets met -- Ready for deployment")
    else:
        needs_phase0 = [ch for ch, a in swing.items() if a == "phase0"]
        needs_phase1 = [ch for ch, a in swing.items() if a == "phase1"]
        if needs_phase0:
            info_(f"Phase 0 재진입 필요 / Re-entry required: {needs_phase0}")
        if needs_phase1:
            info_(f"Phase 1 재진입 필요 / Re-entry required: {needs_phase1}")

    return passed


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grayspot 평가 검증 / Evaluation Validation")
    parser.add_argument("--report", action="store_true",
                        help="HTML 리포트 생성 / Generate HTML report")
    parser.add_argument("--cycle",  type=int, default=1,
                        help="Swing Cycle 번호 / Swing Cycle number (default: 1)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    print("=" * 55)
    print("  Grayspot -- 평가 검증 / Evaluation Validation")
    print(f"  Swing Cycle: {args.cycle}")
    print("=" * 55)

    cfg        = load_config(args.config)
    results    = {}
    eval_result= {}

    test_results = {}
    test_results["모델 존재 확인 / Model Existence"] = test_models_exist(cfg)

    inf_passed, results = test_inference_on_testset(cfg)
    test_results["테스트셋 추론 / Inference"] = inf_passed

    met_passed, eval_result = test_metrics(cfg, results)
    test_results["메트릭 계산 / Metrics"] = met_passed

    test_results["Confusion 분석 / Confusion Analysis"] = test_confusion_analysis(cfg, results)
    test_results["결과 파일 저장 / File Save"]           = test_result_files_saved(cfg, eval_result)
    test_results["Swing 판단 / Swing Decision"]          = test_swing_decision(cfg, eval_result)

    # HTML 리포트 생성 (선택) / Generate HTML report (optional)
    if args.report and eval_result:
        try:
            from reporting.html_report import generate_html_report
            path = generate_html_report(eval_result, cfg, cycle=args.cycle)
            pass_(f"HTML 리포트 생성 / Report generated: {path.name}")
        except Exception as e:
            fail_(f"HTML 리포트 생성 실패 / Failed: {e}")

    # 최종 결과 / Final results
    print(f"\n{'='*55}")
    print("  최종 결과 / Final Results")
    print(f"{'='*55}")
    all_passed = True
    for name, result in test_results.items():
        icon = "[PASS]" if result else "[FAIL]"
        print(f"  {icon}  {name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed. Proceed to test_inference.py")
    else:
        print("  Some tests failed. Fix the issues above before proceeding.")
    print()


if __name__ == "__main__":
    main()