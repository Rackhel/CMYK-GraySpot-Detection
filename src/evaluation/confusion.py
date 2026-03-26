"""
Grayspot -- Confusion Matrix 생성 및 분석 / Confusion Matrix Generation and Analysis
evaluation/confusion.py

Confusion Matrix를 생성하고 오류 패턴을 분석한다.
Generates Confusion Matrix and analyzes error patterns.

주요 기능 / Key features:
    - 채널별 Confusion Matrix 생성 / Per-channel Confusion Matrix generation
    - 인접 레벨 혼동 분석 / Adjacent level confusion analysis
    - 오류 샘플 추출 / Error sample extraction
    - Phase 3 피드백 루프 연계 / Phase 3 feedback loop integration
"""

import csv
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

CHANNELS    = ["Y", "M", "C", "K"]
LEVEL_NAMES = {0: "Normal", 1: "Very Subtle", 2: "Slight", 3: "Clear", 4: "Severe", 5: "Critical"}


# ──────────────────────────────────────────────
# Confusion Matrix 생성 / Confusion Matrix Generation
# ──────────────────────────────────────────────
def build_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_levels: int = 6,
) -> np.ndarray:
    """
    예측값과 실제값으로 Confusion Matrix를 생성한다.
    Builds a Confusion Matrix from predictions and ground truth labels.

    Args:
        y_true:     실제 레벨 리스트 / Ground truth level list
        y_pred:     예측 레벨 리스트 / Predicted level list
        num_levels: 레벨 수 (기본값 6) / Number of levels (default 6)

    Returns:
        (num_levels, num_levels) numpy 배열 / numpy array
        cm[i][j] = 실제 i를 j로 예측한 횟수 / Count of true i predicted as j

    Example:
        >>> cm = build_confusion_matrix(y_true, y_pred)
        >>> print(cm)
    """
    labels = list(range(num_levels))
    return confusion_matrix(y_true, y_pred, labels=labels)


# ──────────────────────────────────────────────
# Confusion Matrix 출력 / Print Confusion Matrix
# ──────────────────────────────────────────────
def print_confusion_matrix(cm: np.ndarray, channel: str) -> None:
    """
    Confusion Matrix를 터미널에 보기 좋게 출력한다.
    Prints the Confusion Matrix to the terminal in a readable format.

    Args:
        cm:      build_confusion_matrix() 반환값 / Return value of build_confusion_matrix()
        channel: "Y" | "M" | "C" | "K"
    """
    num_levels = cm.shape[0]

    print(f"\n  Confusion Matrix -- Channel [{channel}]")
    print(f"  (행: 실제 레벨 / Row: True Level | 열: 예측 레벨 / Col: Predicted Level)")
    print()

    # 헤더 / Header
    header = f"  {'':>6}" + "".join(f"{'P'+str(j):>7}" for j in range(num_levels))
    print(header)
    print("  " + "-" * (6 + 7 * num_levels))

    for i in range(num_levels):
        row_sum = cm[i].sum()
        row     = f"  {'T'+str(i):>6}"
        for j in range(num_levels):
            val = cm[i][j]
            # 대각선(정답)은 강조 / Highlight diagonal (correct)
            marker = "*" if i == j else " "
            row   += f"{val:>6}{marker}"
        row += f"  | {row_sum:>4}"
        print(row)

    print("  " + "-" * (6 + 7 * num_levels))

    # 열 합계 / Column totals
    col_totals = f"  {'Total':>6}" + "".join(f"{cm[:,j].sum():>7}" for j in range(num_levels))
    print(col_totals)
    print()


# ──────────────────────────────────────────────
# 인접 레벨 혼동 분석 / Adjacent Level Confusion Analysis
# ──────────────────────────────────────────────
def analyze_adjacent_confusion(
    cm: np.ndarray,
    channel: str,
    threshold: float = 0.2,
) -> list[dict]:
    """
    인접 레벨 간 혼동이 심한 쌍을 분석한다.
    Analyzes level pairs with high adjacent confusion.

    Level 2 <-> 3 처럼 경계가 모호한 레벨 쌍을 찾아
    Phase 1 재진입이 필요한지 판단하는 데 활용한다.

    Identifies level pairs with ambiguous boundaries (e.g., Level 2 <-> 3)
    to determine whether Phase 1 re-entry is needed.

    Args:
        cm:        Confusion Matrix / Confusion matrix
        channel:   "Y" | "M" | "C" | "K"
        threshold: 혼동 비율 임계값 / Confusion rate threshold (default: 0.2 = 20%)

    Returns:
        혼동이 심한 쌍 목록 / List of highly confused pairs
        [{"pair": (2, 3), "rate_2to3": 0.35, "rate_3to2": 0.28, "action": "phase1"}, ...]

    Example:
        >>> pairs = analyze_adjacent_confusion(cm, "C")
        >>> for p in pairs:
        ...     print(p)
    """
    num_levels = cm.shape[0]
    confused_pairs = []

    for i in range(num_levels - 1):
        j = i + 1  # 인접 레벨 / Adjacent level

        # i -> j 혼동 비율 / Confusion rate i -> j
        row_sum_i = cm[i].sum()
        rate_i2j  = cm[i][j] / row_sum_i if row_sum_i > 0 else 0.0

        # j -> i 혼동 비율 / Confusion rate j -> i
        row_sum_j = cm[j].sum()
        rate_j2i  = cm[j][i] / row_sum_j if row_sum_j > 0 else 0.0

        if rate_i2j >= threshold or rate_j2i >= threshold:
            confused_pairs.append({
                "channel":   channel,
                "pair":      (i, j),
                "pair_name": f"L{i}({LEVEL_NAMES[i]}) <-> L{j}({LEVEL_NAMES[j]})",
                "rate_i2j":  round(rate_i2j, 4),
                "rate_j2i":  round(rate_j2i, 4),
                "action":    "phase1",  # Level 경계 재검토 필요 / Level boundary review needed
            })

    return confused_pairs


# ──────────────────────────────────────────────
# 오류 샘플 추출 / Error Sample Extraction
# ──────────────────────────────────────────────
def extract_error_samples(
    y_true: list[int],
    y_pred: list[int],
    sample_paths: list[str],
    channel: str,
    max_samples: int = 20,
) -> list[dict]:
    """
    오분류된 샘플을 추출하여 반환한다.
    Extracts and returns misclassified samples.

    Phase 1 재검토 시 우선 검수 대상 목록을 만드는 데 활용한다.
    Used to build a priority review list for Phase 1 re-entry.

    Args:
        y_true:       실제 레벨 리스트 / Ground truth level list
        y_pred:       예측 레벨 리스트 / Predicted level list
        sample_paths: 샘플 파일 경로 리스트 / Sample file path list
        channel:      "Y" | "M" | "C" | "K"
        max_samples:  최대 추출 수 / Max number of samples to extract

    Returns:
        오분류 샘플 목록 / List of misclassified samples
        [{"path": "...", "true": 2, "pred": 3, "diff": 1}, ...]

    Example:
        >>> errors = extract_error_samples(y_true, y_pred, paths, "Y")
        >>> for e in errors[:5]:
        ...     print(e)
    """
    errors = []
    for true, pred, path in zip(y_true, y_pred, sample_paths):
        if true != pred:
            errors.append({
                "channel": channel,
                "path":    path,
                "true":    true,
                "pred":    pred,
                "diff":    abs(true - pred),  # 레벨 차이 / Level difference
            })

    # 레벨 차이가 큰 순서로 정렬 (심각한 오류 우선)
    # Sort by level difference descending (most severe errors first)
    errors.sort(key=lambda x: x["diff"], reverse=True)

    return errors[:max_samples]


# ──────────────────────────────────────────────
# 전체 분석 실행 / Run Full Analysis
# ──────────────────────────────────────────────
def run_confusion_analysis(
    y_true: list[int],
    y_pred: list[int],
    channel: str,
    cfg: dict,
    sample_paths: list[str] = None,
) -> dict:
    """
    Confusion Matrix 생성 및 전체 분석을 실행한다.
    Runs Confusion Matrix generation and full analysis.

    Args:
        y_true:       실제 레벨 리스트 / Ground truth level list
        y_pred:       예측 레벨 리스트 / Predicted level list
        channel:      "Y" | "M" | "C" | "K"
        cfg:          config.yaml 딕셔너리 / config.yaml dictionary
        sample_paths: 샘플 파일 경로 리스트 (없으면 오류 샘플 추출 생략)
                      Sample file path list (error extraction skipped if None)

    Returns:
        {
          "channel":          "Y",
          "confusion":        [[...], ...],     # Confusion Matrix
          "adjacent_confused": [...],           # 인접 레벨 혼동 쌍 / Adjacent confused pairs
          "error_samples":     [...],           # 오분류 샘플 / Misclassified samples
          "needs_phase1":      True | False,    # Phase 1 재진입 필요 여부
        }

    Example:
        >>> result = run_confusion_analysis(y_true, y_pred, "Y", cfg)
        >>> if result["needs_phase1"]:
        ...     print("Phase 1 재진입 필요")
    """
    num_levels = cfg["data"]["num_levels"]
    fb         = cfg["evaluation"]["swing_feedback"]

    # Confusion Matrix 생성 / Build Confusion Matrix
    cm = build_confusion_matrix(y_true, y_pred, num_levels)

    # 터미널 출력 / Print to terminal
    print_confusion_matrix(cm, channel)

    # 인접 레벨 혼동 분석 / Analyze adjacent confusion
    adjacent_confused = analyze_adjacent_confusion(
        cm,
        channel,
        threshold=0.2,
    )

    if adjacent_confused:
        print(f"  [{channel}] 인접 레벨 혼동 감지 / Adjacent confusion detected:")
        for pair in adjacent_confused:
            print(f"    {pair['pair_name']} -- "
                  f"{pair['rate_i2j']:.0%} / {pair['rate_j2i']:.0%}")
    else:
        print(f"  [{channel}] 인접 레벨 혼동 없음 / No adjacent confusion detected")

    # 오류 샘플 추출 / Extract error samples
    error_samples = []
    if sample_paths:
        error_samples = extract_error_samples(y_true, y_pred, sample_paths, channel)
        print(f"  [{channel}] 오분류 샘플 / Misclassified samples: {len(error_samples)}개")

    # Phase 1 재진입 필요 여부 판단 / Determine if Phase 1 re-entry is needed
    needs_phase1 = len(adjacent_confused) > 0

    result = {
        "channel":           channel,
        "confusion":         cm.tolist(),
        "adjacent_confused": adjacent_confused,
        "error_samples":     error_samples,
        "needs_phase1":      needs_phase1,
    }

    # 결과 저장 / Save results
    _save_confusion_result(result, cfg)

    return result


# ──────────────────────────────────────────────
# 결과 저장 / Save Results
# ──────────────────────────────────────────────
def _save_confusion_result(result: dict, cfg: dict) -> None:
    """
    Confusion Matrix 분석 결과를 JSON과 CSV로 저장한다.
    Saves Confusion Matrix analysis results to JSON and CSV.
    """
    reports_dir = Path(cfg["storage"]["reports_dir"])
    channel     = result["channel"]

    # JSON 저장 / Save JSON
    json_path = reports_dir / f"confusion_analysis_{channel}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 인접 혼동 CSV 저장 / Save adjacent confusion CSV
    if result["adjacent_confused"]:
        csv_path = reports_dir / f"adjacent_confusion_{channel}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["channel", "pair_name", "rate_i2j", "rate_j2i", "action"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pair in result["adjacent_confused"]:
                writer.writerow({
                    "channel":   pair["channel"],
                    "pair_name": pair["pair_name"],
                    "rate_i2j":  pair["rate_i2j"],
                    "rate_j2i":  pair["rate_j2i"],
                    "action":    pair["action"],
                })

    # 오류 샘플 CSV 저장 / Save error samples CSV
    if result["error_samples"]:
        csv_path = reports_dir / f"error_samples_{channel}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["channel", "path", "true", "pred", "diff"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["error_samples"])

    print(f"  [{channel}] 분석 결과 저장 / Analysis saved: {reports_dir}")