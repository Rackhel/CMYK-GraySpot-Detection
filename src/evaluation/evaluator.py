"""
Grayspot -- 전체 평가 파이프라인 통합 / Full Evaluation Pipeline Integrator
evaluation/evaluator.py

metrics.py, confusion.py, plots.py, html_report.py를 통합하여
Phase 3 평가 전체를 단일 진입점으로 실행한다.

Integrates metrics.py, confusion.py, plots.py, and html_report.py
to run the full Phase 3 evaluation from a single entry point.

사용법 / Usage:
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(cfg)
    report    = evaluator.run(results, cycle=1)

    # report["targets_met"] 로 배포 가능 여부 확인
    # Check report["targets_met"] to determine deployment readiness
"""

import json
from pathlib import Path

from evaluation.metrics import (
    evaluate_all_channels,
    print_evaluation_report,
    save_evaluation_results,
)
from evaluation.confusion import run_confusion_analysis
from utils.logger import get_eval_logger, log_swing_decision

CHANNELS = ["Y", "M", "C", "K"]


class Evaluator:
    """
    Phase 3 전체 평가 파이프라인을 통합 실행하는 클래스.
    Class that integrates and runs the full Phase 3 evaluation pipeline.

    실행 순서 / Execution order:
        1. 채널별 메트릭 계산 / Per-channel metric computation
        2. Swing 피드백 루프 판단 / Swing feedback loop decision
        3. Confusion Matrix 생성 및 인접 혼동 분석 / Confusion Matrix + adjacent confusion analysis
        4. 그래프 생성 (선택) / Graph generation (optional)
        5. HTML 리포트 생성 (선택) / HTML report generation (optional)
        6. 결과 저장 / Save results

    Example:
        >>> import yaml
        >>> cfg       = yaml.safe_load(open("config/config.yaml"))
        >>> evaluator = Evaluator(cfg)
        >>> results   = {"Y": {"y_true": [...], "y_pred": [...]}, ...}
        >>> report    = evaluator.run(results, cycle=1, plots=True, html=True)
        >>> print(report["targets_met"])
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: config.yaml 딕셔너리 / config.yaml dictionary
        """
        self.cfg    = cfg
        self.logger = get_eval_logger(cfg)

    def run(
        self,
        results: dict[str, dict],
        cycle: int = 1,
        plots: bool = False,
        html: bool = False,
        sample_paths: dict[str, list[str]] = None,
    ) -> dict:
        """
        Phase 3 전체 평가를 실행하고 결과를 반환한다.
        Runs the full Phase 3 evaluation and returns results.

        Args:
            results:      채널별 예측 결과 / Per-channel prediction results
                          {"Y": {"y_true": [...], "y_pred": [...]}, ...}
            cycle:        Swing Cycle 번호 / Swing Cycle number (default: 1)
            plots:        그래프 생성 여부 / Whether to generate graphs (default: False)
            html:         HTML 리포트 생성 여부 / Whether to generate HTML report (default: False)
            sample_paths: 채널별 샘플 경로 (오류 샘플 추출용, 선택)
                          Per-channel sample paths (for error extraction, optional)
                          {"Y": ["path/img1.png", ...], ...}

        Returns:
            {
              "cycle":             1,
              "per_channel":       {"Y": {...}, ...},  # 채널별 메트릭 / Per-channel metrics
              "overall_accuracy":  0.92,               # 전체 정확도 / Overall accuracy
              "swing_decision":    {"Y": "pass", ...}, # Swing 판단 / Swing decisions
              "targets_met":       True,               # 목표 달성 여부 / Targets met
              "confusion_analysis":{"Y": {...}, ...},  # Confusion 분석 / Confusion analysis
              "plot_paths":        [...],              # 생성된 그래프 경로 / Generated plot paths
              "report_path":       Path(...),          # HTML 리포트 경로 / HTML report path
            }
        """
        self.logger.info("=" * 55)
        self.logger.info(f"Phase 3 -- Swing Cycle {cycle} 평가 시작 / Evaluation started")
        self.logger.info("=" * 55)

        if not results:
            self.logger.error("평가 결과 없음 / No evaluation results provided")
            return {}

        # ── 1. 채널별 메트릭 + Swing 판단 / Per-channel metrics + Swing decision ──
        eval_result = evaluate_all_channels(results, self.cfg)
        eval_result["cycle"] = cycle

        # 터미널 출력 / Print to terminal
        print_evaluation_report(eval_result)

        # Swing 판단 로그 기록 / Log Swing decision
        log_swing_decision(self.logger, eval_result["swing_decision"])

        # ── 2. Confusion Matrix 분석 / Confusion Matrix analysis ──
        confusion_analysis = {}
        for ch in CHANNELS:
            if ch not in results:
                continue

            y_true = results[ch]["y_true"]
            y_pred = results[ch]["y_pred"]
            paths  = (sample_paths or {}).get(ch, None)

            confusion_analysis[ch] = run_confusion_analysis(
                y_true, y_pred, ch, self.cfg, sample_paths=paths
            )

            # Phase 1 재진입 필요 여부 로그 / Log Phase 1 re-entry need
            if confusion_analysis[ch]["needs_phase1"]:
                self.logger.warning(
                    f"[{ch}] 인접 레벨 혼동 감지 -- Phase 1 재진입 권장 / "
                    f"Adjacent confusion detected -- Phase 1 re-entry recommended"
                )

        eval_result["confusion_analysis"] = confusion_analysis

        # ── 3. 결과 저장 (CSV + JSON) / Save results (CSV + JSON) ──
        save_evaluation_results(eval_result, self.cfg)
        self._save_full_report(eval_result, cycle)

        # ── 4. 그래프 생성 (선택) / Generate graphs (optional) ──
        plot_paths = []
        if plots:
            self.logger.info("그래프 생성 중 / Generating plots...")
            try:
                from reporting.plots import generate_all_plots
                plot_paths = generate_all_plots(eval_result, self.cfg)
                self.logger.info(f"{len(plot_paths)}개 그래프 생성 완료 / plots generated")
            except ImportError:
                self.logger.warning(
                    "matplotlib 미설치 / Not installed -- 그래프 생성 건너뜀 / Skipping plot generation\n"
                    "설치 / Install: pip install matplotlib"
                )

        eval_result["plot_paths"] = plot_paths

        # ── 5. HTML 리포트 생성 (선택) / Generate HTML report (optional) ──
        report_path = None
        if html:
            self.logger.info("HTML 리포트 생성 중 / Generating HTML report...")
            try:
                from reporting.html_report import generate_html_report
                report_path = generate_html_report(eval_result, self.cfg, cycle=cycle)
                self.logger.info(f"HTML 리포트 생성 완료 / Report generated: {report_path}")
            except Exception as e:
                self.logger.error(f"HTML 리포트 생성 실패 / Failed: {e}")

        eval_result["report_path"] = report_path

        # ── 최종 요약 출력 / Print final summary ──
        self._print_final_summary(eval_result, cycle)

        return eval_result

    def _save_full_report(self, eval_result: dict, cycle: int) -> None:
        """
        전체 평가 결과를 단일 JSON 파일로 저장한다.
        Saves the full evaluation result as a single JSON file.
        """
        reports_dir = Path(self.cfg["storage"]["reports_dir"])
        out_path    = reports_dir / f"full_eval_cycle_{cycle}.json"

        # JSON 직렬화 가능하도록 변환 / Convert for JSON serialization
        serializable = {
            k: v for k, v in eval_result.items()
            if k not in ("plot_paths", "report_path")
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        self.logger.info(f"전체 평가 결과 저장 / Full evaluation saved: {out_path}")

    def _print_final_summary(self, eval_result: dict, cycle: int) -> None:
        """
        평가 완료 후 최종 요약을 출력한다.
        Prints the final summary after evaluation.
        """
        targets_met   = eval_result.get("targets_met", False)
        overall       = eval_result.get("overall_accuracy", 0)
        swing         = eval_result.get("swing_decision", {})
        needs_phase0  = [ch for ch, a in swing.items() if a == "phase0"]
        needs_phase1  = [ch for ch, a in swing.items() if a == "phase1"]

        print("\n" + "=" * 55)
        print(f"  Swing Cycle {cycle} -- 평가 완료 / Evaluation Complete")
        print("=" * 55)
        print(f"  Overall Accuracy : {overall:.4f}")
        print(f"  Targets Met      : {'Yes -- Ready for deployment' if targets_met else 'No -- See swing decisions below'}")

        if needs_phase0:
            print(f"  Phase 0 재진입 / Re-entry required : {', '.join(needs_phase0)}")
        if needs_phase1:
            print(f"  Phase 1 재진입 / Re-entry required : {', '.join(needs_phase1)}")

        if targets_met:
            print("\n  All targets met. Proceed to model conversion and deployment.")
        else:
            max_cycles = self.cfg["evaluation"]["swing_max_cycles"]
            if cycle >= max_cycles:
                print(f"\n  Max Swing cycles ({max_cycles}) reached.")
                print("  Consider reviewing the approach or expanding the dataset.")
            else:
                print(f"\n  Cycle {cycle}/{max_cycles} complete. Re-run training for flagged channels.")

        print()