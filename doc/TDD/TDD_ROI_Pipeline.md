---
type: tdd
domain: roi_pipeline
status: passing
last_updated: 2026-05-21
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_ROI_Pipeline.md"
  - "../Contract/Contract_roi_pipeline.md"
---

# [TDD] ROI Pipeline — ROI 추출 및 CMYK 분리 / ROI Extraction and CMYK Splitting

> **목적 / Purpose**: `ROIExtractor` 클래스의 동작을 BDD 시나리오와 TDD 스펙으로 정의한다.
> **테스트 파일 / Test Files**: `src/tests/unit/test_roi_extractor.py`, `src/tests/integration/test_roi_pipeline.py`
> **상태 / Status**: ✅ Passing — T-ROI-01~14 (10/10), T-ROI-INT-01~03 (9/9) 통과

---

## 1. BDD 시나리오 / BDD Scenarios

### Feature: CMYK 채널 분리 / CMYK Channel Splitting

**Scenario 1: 정상 BGR 이미지에서 CMYK 채널 분리 / CMYK channel splitting from normal BGR image**
```
Given  (128, 128, 3) BGR uint8 이미지가 주어졌을 때
# Given a (128, 128, 3) BGR uint8 image
When   split_cmyk(image) 를 호출하면
# When split_cmyk(image) is called
Then   결과 dict에 "C", "M", "Y", "K" 키가 존재하고
# Then the result dict has "C", "M", "Y", "K" keys
And    각 채널 array의 shape는 (128, 128)이고
# And each channel array shape is (128, 128)
And    각 채널 값의 범위는 [0.0, 1.0] float32이다
# And each channel value range is [0.0, 1.0] float32
```

**Scenario 2: 순수 흰색 이미지 → CMYK 분리 / Pure white image CMYK splitting**
```
Given  모든 픽셀이 (255, 255, 255) 흰색인 BGR 이미지가 주어졌을 때
# Given a BGR image where all pixels are (255, 255, 255) white
When   split_cmyk(image) 를 호출하면
# When split_cmyk(image) is called
Then   C, M, Y 채널은 모두 0.0이고
# Then C, M, Y channels are all 0.0
And    K 채널도 0.0이다
# And K channel is also 0.0
```

**Scenario 3: 순수 검정 이미지 → CMYK 분리 / Pure black image CMYK splitting**
```
Given  모든 픽셀이 (0, 0, 0) 검정인 BGR 이미지가 주어졌을 때
# Given a BGR image where all pixels are (0, 0, 0) black
When   split_cmyk(image) 를 호출하면
# When split_cmyk(image) is called
Then   C, M, Y 채널은 모두 1.0이고
# Then C, M, Y channels are all 1.0
And    K 채널도 1.0이다
# And K channel is also 1.0
```

---

### Feature: ROI 패치 추출 / ROI Patch Extraction

**Scenario 4: fixed 모드 패치 추출 / Patch extraction in fixed mode**
```
Given  유효한 스캔 이미지 경로와 cfg(roi.mode="fixed", roi.fixed_coords=[0,0,128,128])가 주어졌을 때
# Given a valid scan image path and cfg(roi.mode="fixed", roi.fixed_coords=[0,0,128,128])
When   extract_patches(image_path, channel="Y", level=3) 를 호출하면
# When extract_patches(image_path, channel="Y", level=3) is called
Then   반환 리스트의 각 원소 shape는 (128, 128, 3)이고
# Then each element in the returned list has shape (128, 128, 3)
And    dtype은 uint8이다
# And dtype is uint8
```

**Scenario 5: 존재하지 않는 경로 에러 / Non-existent path error handling**
```
Given  존재하지 않는 파일 경로가 주어졌을 때
# Given a file path that does not exist
When   extract_patches(invalid_path, ...) 를 호출하면
# When extract_patches(invalid_path, ...) is called
Then   FileNotFoundError가 발생한다
# Then FileNotFoundError is raised
```

**Scenario 6: ROI 좌표 범위 초과 / ROI coordinates exceeding image bounds**
```
Given  (64, 64) 크기 이미지와 roi.fixed_coords=[0,0,256,256]이 주어졌을 때
# Given a (64, 64) image and roi.fixed_coords=[0,0,256,256]
When   extract_patches() 를 호출하면
# When extract_patches() is called
Then   이미지 범위 내로 클리핑하여 패치를 반환하거나 빈 리스트([])를 반환한다
# Then patches clipped to image bounds are returned, or an empty list ([]) is returned
```

**Scenario 7: 단채널 → 3채널 변환 / Single-channel to 3-channel conversion**
```
Given  Y채널 (grayscale float [0,1]) 이 주어졌을 때
# Given a Y channel (grayscale float [0,1])
When   3채널로 변환하면
# When converted to 3 channels
Then   shape는 (H, W, 3)이고
# Then shape is (H, W, 3)
And    세 채널의 값이 동일하다 (ch_uint8 을 3회 복제)
# And the three channel values are identical (ch_uint8 duplicated 3 times)
```

---

## 2. TDD 스펙 / TDD Specifications

### 2.1 split_cmyk()

**테스트 파일**: `src/tests/unit/test_roi_extractor.py`

| 테스트 ID / Test ID | 입력 / Input | 기댓값 / Expected | 검증 포인트 / Verification |
| --- | --- | --- | --- |
| T-ROI-01 | 흰색 BGR 이미지 (255,255,255) | C=M=Y=K=0.0 | `np.allclose(result["C"], 0.0)` |
| T-ROI-02 | 검정 BGR 이미지 (0,0,0) | C=M=Y=K=1.0 | `np.allclose(result["K"], 1.0)` |
| T-ROI-03 | 순수 빨강 (B=0,G=0,R=255) → BGR=(0,0,255) | C=0.0, M=1.0, Y=1.0 | 채널별 검증 |
| T-ROI-04 | 랜덤 BGR (128,128,3) | 모든 채널 [0,1] 범위 | `assert (arr >= 0).all() and (arr <= 1).all()` |
| T-ROI-05 | 입력 dtype=uint8 | 반환 dtype=float32 | `assert result["C"].dtype == np.float32` |

```python
# 스펙 예시 / Spec example — 구현 전 실패해야 함 / must fail before implementation
def test_split_cmyk_white_image():
    extractor = ROIExtractor(cfg=minimal_cfg())
    white = np.ones((128, 128, 3), dtype=np.uint8) * 255
    result = extractor.split_cmyk(white)
    assert set(result.keys()) == {"C", "M", "Y", "K"}
    assert np.allclose(result["C"], 0.0)
    assert np.allclose(result["K"], 0.0)

def test_split_cmyk_black_image():
    extractor = ROIExtractor(cfg=minimal_cfg())
    black = np.zeros((128, 128, 3), dtype=np.uint8)
    result = extractor.split_cmyk(black)
    assert np.allclose(result["C"], 1.0)
    assert np.allclose(result["K"], 1.0)
```

### 2.2 extract_patches()

| 테스트 ID / Test ID | 입력 / Input | 기댓값 / Expected | 검증 포인트 / Verification |
| --- | --- | --- | --- |
| T-ROI-10 | 유효 이미지 경로, fixed 모드 | `List[np.ndarray]`, 각 (128,128,3) uint8 | shape + dtype |
| T-ROI-11 | 존재하지 않는 경로 | `FileNotFoundError` | `pytest.raises` |
| T-ROI-12 | channel="Y" | 반환 패치가 3채널이며 3채널 값이 동일 | `assert np.array_equal(p[:,:,0], p[:,:,1])` |
| T-ROI-13 | level=3 | 반환 리스트 비어 있지 않음 | `assert len(patches) > 0` |
| T-ROI-14 | cfg 없이 생성 시도 | `TypeError` 또는 `ValueError` | `pytest.raises` |

```python
def test_extract_patches_returns_correct_shape(tmp_image_path, minimal_cfg):
    extractor = ROIExtractor(cfg=minimal_cfg)
    patches = extractor.extract_patches(tmp_image_path, channel="Y", level=3)
    assert len(patches) > 0
    for p in patches:
        assert p.shape == (128, 128, 3)
        assert p.dtype == np.uint8

def test_extract_patches_invalid_path(minimal_cfg):
    extractor = ROIExtractor(cfg=minimal_cfg)
    with pytest.raises(FileNotFoundError):
        extractor.extract_patches("/nonexistent/path.png", channel="Y", level=1)
```

---

## 3. 통합 테스트 스펙 / Integration Test Specifications

**테스트 파일**: `src/tests/integration/test_roi_pipeline.py` ✅ **9/9 통과**

| 테스트 ID | 클래스::함수 | 시나리오 | 상태 |
| --- | --- | --- | --- |
| T-ROI-INT-01a | `TestROIExtractorToFile::test_patches_saved_as_png` | extract_patches_from_roi() → PNG 저장 확인 | ✅ |
| T-ROI-INT-01b | `TestROIExtractorToFile::test_saved_patch_shape` | 저장 PNG = (128,128,3) uint8 | ✅ |
| T-ROI-INT-01c | `TestROIExtractorToFile::test_saved_filename_follows_naming_convention` | 파일명 `{roi_stem}_{idx:04d}.png` 형식 | ✅ |
| T-ROI-INT-02a | `TestContrastiveDatasetLoad::test_dataset_loads_without_error` | ContrastiveDataset 생성 에러 없음 | ✅ |
| T-ROI-INT-02b | `TestContrastiveDatasetLoad::test_item_returns_two_tensors` | (view1, view2) 텐서 쌍 반환 | ✅ |
| T-ROI-INT-02c | `TestContrastiveDatasetLoad::test_tensor_shape_is_3_128_128` | 각 텐서 shape (3,128,128) | ✅ |
| T-ROI-INT-02d | `TestContrastiveDatasetLoad::test_tensor_dtype_float32` | dtype float32 | ✅ |
| T-ROI-INT-03a | `TestModelForwardPass::test_phase0_forward_output_shape` | Phase 0 출력 (B, 128) | ✅ |
| T-ROI-INT-03b | `TestModelForwardPass::test_phase2_forward_output_shape` | Phase 2 출력 (B, 6) | ✅ |

---

## 4. Conftest 픽스처 / Conftest Fixtures

```python
# src/tests/unit/conftest.py 에 추가할 픽스처 / Fixtures to add to conftest.py

@pytest.fixture
def minimal_cfg():
    return {
        "roi": {"mode": "fixed", "fixed_coords": [0, 0, 128, 128]},
        "data": {"image_size": 128},
    }

@pytest.fixture
def tmp_image_path(tmp_path):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "test_scan.png"
    cv2.imwrite(str(path), img)
    return path
```

---

## 5. prepare_dataset.py 테스트 스펙 / prepare_dataset.py Test Spec

**테스트 파일**: `src/tests/unit/test_prepare_dataset.py` ✅ **17/17 통과**

### 5.1 _parse_filename()

| 테스트 ID | 입력 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-PREP-01 | `"lvl3_Scanned_Documents_(113)_3_1_M"` | `(3, "M")` | ✅ |
| T-PREP-02 | Y/M/C/K 각 채널 파일명 | `(level, channel)` | ✅ |
| T-PREP-03 | `"lvl0_scan_001_Y"` | `(0, "Y")` | ✅ |
| T-PREP-04 | `"lvl5_scan_001_C"` | `(5, "C")` | ✅ |
| T-PREP-05 | 패턴 불일치 문자열 | `None` | ✅ |
| T-PREP-06 | 유효하지 않은 채널 (Z, R) | `None` | ✅ |

### 5.2 _load_roi_labels()

| 테스트 ID | 조건 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-PREP-10 | roi_labels.csv 없음 | `{}` | ✅ |
| T-PREP-11 | 유효한 CSV | `{stem: level}` 매핑 | ✅ |
| T-PREP-12 | CSV 로드 | level 값이 `int` 타입 | ✅ |
| T-PREP-13 | 4채널 CSV | 4개 키 모두 로드 | ✅ |
| T-PREP-14 | 헤더만 있는 CSV | `{}` | ✅ |

### 5.3 채널별 독립 라벨링 오버라이드

| 테스트 ID | 시나리오 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-PREP-20 | roi_labels.csv 레벨 vs 파일명 레벨 | roi_labels.csv 우선 | ✅ |
| T-PREP-21 | CSV에 없는 파일 | 파일명 레벨 fallback | ✅ |
| T-PREP-22 | 같은 스캔 4채널 | 채널마다 다른 레벨 가능 | ✅ |

### 5.4 상수 검증

| 테스트 ID | 항목 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-PREP-30 | `EXTRACT_CAP` | `{0:330,1:330,2:330,3:265,4:165,5:100}` | ✅ |
| T-PREP-31 | 합계 | 1,520 | ✅ |
| T-PREP-32 | `CHANNELS` | `{Y, M, C, K}` | ✅ |

---

## 6. augment_dataset.py 테스트 스펙 / augment_dataset.py Test Spec

**테스트 파일**: `src/tests/unit/test_augment_dataset.py` ✅ **14/14 통과**

### 6.1 PRD_TARGETS 상수

| 테스트 ID | 항목 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-AUG-01 | `PRD_TARGETS` | `{0:330,1:330,2:330,3:265,4:165,5:100}` | ✅ |
| T-AUG-02 | 합계 | 1,520 | ✅ |

### 6.2 _augment_image()

| 테스트 ID | 입력 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-AUG-10 | PIL Image | 반환값 PIL.Image 인스턴스 | ✅ |
| T-AUG-11 | 128×128 이미지 | 크기 동일 유지 | ✅ |
| T-AUG-12 | 30회 호출 | 여러 다른 결과 (무작위 변환) | ✅ |
| T-AUG-13 | RGB 이미지 | mode 동일 유지 | ✅ |
| T-AUG-14 | 원본 이미지 | 원본 미수정 | ✅ |

### 6.3 _read_csv() / _write_csv()

| 테스트 ID | 시나리오 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-AUG-20 | 없는 파일 읽기 | 빈 리스트 | ✅ |
| T-AUG-21 | 쓰기 후 읽기 roundtrip | 동일 데이터 복원 | ✅ |
| T-AUG-22 | 저장된 CSV 헤더 | `filepath,channel,level` | ✅ |
| T-AUG-23 | `_read_csv` 반환 타입 | `list[dict]` | ✅ |

### 6.4 증강 발동 조건

| 테스트 ID | 시나리오 | 기댓값 | 상태 |
| --- | --- | --- | --- |
| T-AUG-30 | 목표 달성 | shortage = 0 | ✅ |
| T-AUG-31 | 목표 미달 (100 < 330) | shortage = 230 | ✅ |
| T-AUG-32 | 목표 초과 (400 > 330) | shortage = 0 | ✅ |

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | ROI 추출 정의 / ROI extraction definition |
| [Contract_roi_pipeline.md](../Contract/Contract_roi_pipeline.md) | API 계약 / API contract |
