---
type: tdd
domain: evaluate_script
status: failing
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_Training_Pipeline.md"
  - "../Contract/Contract_roi_pipeline.md"
---

# [TDD] Evaluate Script — 독립 평가 스크립트 / Independent Evaluation Script

> **목적 / Purpose**: `src/scripts/evaluate.py` CLI 동작을 BDD/TDD로 정의한다.
> **테스트 파일 / Test Files**: `src/tests/unit/test_evaluate_script.py`, `src/tests/integration/test_evaluate_integration.py`
> **상태 / Status**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오 / BDD Scenarios

### Feature: CLI 평가 스크립트 실행 / CLI Evaluation Script Execution

**Scenario 1: 유효한 체크포인트로 단일 채널 평가 / Single-channel evaluation with valid checkpoint**
```
Given  outputs/models/best_Y.pt 가 존재하고 --channel Y 인수가 주어졌을 때
When   python -m src.scripts.evaluate --channel Y 를 실행하면
Then   종료 코드 0으로 완료되고
And    outputs/reports/ 에 JSON 리포트 파일이 생성되고
And    outputs/reports/ 에 HTML 리포트 파일이 생성된다
```

**Scenario 2: 체크포인트 미존재 시 종료 / Exit when checkpoint is missing**
```
Given  체크포인트 파일이 존재하지 않을 때
When   python -m src.scripts.evaluate --channel Y 를 실행하면
Then   종료 코드 1로 종료되고
And    오류 메시지가 stderr에 출력된다
```

**Scenario 3: --channel all 옵션으로 전체 채널 평가 / All-channel evaluation**
```
Given  C, M, Y, K 모두의 체크포인트가 존재할 때
When   python -m src.scripts.evaluate --channel all 를 실행하면
Then   채널별 리포트가 각각 생성된다
```

**Scenario 4: --output-dir 커스텀 경로 지정 / Custom output directory**
```
Given  --output-dir /tmp/my_reports 인수가 주어졌을 때
When   evaluate.py 를 실행하면
Then   /tmp/my_reports/ 에 리포트가 저장된다
```

---

## 2. TDD 스펙 / TDD Specifications

### 2.1 CLI 인수 파싱 / CLI Argument Parsing

| 테스트 ID / Test ID | 입력 argv / Input argv | 기댓값 / Expected |
| --- | --- | --- |
| T-EVAL-01 | `["--channel", "Y"]` | `args.channel == "Y"` |
| T-EVAL-02 | `["--channel", "all"]` | `args.channel == "all"` |
| T-EVAL-03 | `[]` (필수 인수 누락) | `SystemExit` (argparse) |
| T-EVAL-04 | `["--channel", "X"]` (유효하지 않은 채널) | `SystemExit` 또는 `ValueError` |
| T-EVAL-05 | `["--channel", "Y", "--output-dir", "/tmp/r"]` | `args.output_dir == "/tmp/r"` |

```python
def test_parse_args_channel():
    from scripts.evaluate import parse_args
    args = parse_args(["--channel", "Y"])
    assert args.channel == "Y"

def test_parse_args_missing_channel():
    from scripts.evaluate import parse_args
    with pytest.raises(SystemExit):
        parse_args([])
```

### 2.2 리포트 생성 검증 / Report Generation Validation

| 테스트 ID / Test ID | 전제 조건 / Precondition | 검증 포인트 / Verification |
| --- | --- | --- |
| T-EVAL-10 | 유효 체크포인트 존재 | JSON 파일 생성 (`Path.exists()`) |
| T-EVAL-11 | 유효 체크포인트 존재 | JSON 내 `accuracy` 키 존재 |
| T-EVAL-12 | 유효 체크포인트 존재 | HTML 파일 생성 |
| T-EVAL-13 | 체크포인트 미존재 | 함수 반환값 또는 `sys.exit(1)` |

```python
def test_evaluate_creates_json_report(mock_checkpoint, tmp_output_dir, minimal_cfg):
    # evaluate.py main() 실행 후 파일 존재 확인 / Check file existence after main() execution
    from scripts.evaluate import main
    main(["--channel", "Y", "--output-dir", str(tmp_output_dir)])
    json_files = list(tmp_output_dir.glob("*.json"))
    assert len(json_files) > 0

def test_evaluate_missing_checkpoint_exits(tmp_path):
    from scripts.evaluate import main
    with pytest.raises(SystemExit) as exc_info:
        main(["--channel", "Y", "--checkpoint", str(tmp_path / "nonexistent.pt")])
    assert exc_info.value.code == 1
```

---

## 3. 통합 테스트 스펙 / Integration Test Specifications

| 테스트 ID / Test ID | 시나리오 / Scenario | 검증 포인트 / Verification |
| --- | --- | --- |
| T-EVAL-INT-01 | 실제 체크포인트 로드 → Evaluator 실행 → 리포트 저장 | JSON accuracy 값이 float [0,1] |
| T-EVAL-INT-02 | --channel all 실행 시 4채널 리포트 모두 생성 | 4개 JSON 파일 존재 |

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | 실행 명령 정의 / Execution command definition |
| [Contract_roi_pipeline.md](../Contract/Contract_roi_pipeline.md) | evaluate.py API 계약 / API contract |
| [Contract_evaluation_reporting.md](../Contract/Contract_evaluation_reporting.md) | Evaluator API |
