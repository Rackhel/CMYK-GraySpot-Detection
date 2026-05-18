---
type: tdd
domain: roi_pipeline
status: failing
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_ROI_Pipeline.md"
  - "../Contract/Contract_roi_pipeline.md"
---

# [TDD] ROI Pipeline — ROI 추출 및 CMYK 분리

> **목적**: `ROIExtractor` 클래스의 동작을 BDD 시나리오와 TDD 스펙으로 정의한다.
> **테스트 파일**: `src/tests/unit/test_roi_extractor.py`, `src/tests/integration/test_roi_pipeline.py`
> **상태**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오

### Feature: CMYK 채널 분리

**Scenario 1: 정상 BGR 이미지에서 CMYK 채널 분리**
```
Given  (128, 128, 3) BGR uint8 이미지가 주어졌을 때
When   split_cmyk(image) 를 호출하면
Then   결과 dict에 "C", "M", "Y", "K" 키가 존재하고
And    각 채널 array의 shape는 (128, 128)이고
And    각 채널 값의 범위는 [0.0, 1.0] float32이다
```

**Scenario 2: 순수 흰색 이미지 (RGB=255,255,255) → CMYK 분리**
```
Given  모든 픽셀이 (255, 255, 255) 흰색인 BGR 이미지가 주어졌을 때
When   split_cmyk(image) 를 호출하면
Then   C, M, Y 채널은 모두 0.0이고
And    K 채널도 0.0이다
```

**Scenario 3: 순수 검정 이미지 (RGB=0,0,0) → CMYK 분리**
```
Given  모든 픽셀이 (0, 0, 0) 검정인 BGR 이미지가 주어졌을 때
When   split_cmyk(image) 를 호출하면
Then   C, M, Y 채널은 모두 1.0이고
And    K 채널도 1.0이다
```

---

### Feature: ROI 패치 추출

**Scenario 4: fixed 모드 — 정상 이미지에서 패치 추출**
```
Given  유효한 스캔 이미지 경로와 cfg(roi.mode="fixed", roi.fixed_coords=[0,0,128,128])가 주어졌을 때
When   extract_patches(image_path, channel="Y", level=3) 를 호출하면
Then   반환 리스트의 각 원소 shape는 (128, 128, 3)이고
And    dtype은 uint8이다
```

**Scenario 5: 존재하지 않는 이미지 경로 — 에러 처리**
```
Given  존재하지 않는 파일 경로가 주어졌을 때
When   extract_patches(invalid_path, ...) 를 호출하면
Then   FileNotFoundError가 발생한다
```

**Scenario 6: ROI 좌표가 이미지 범위를 벗어난 경우**
```
Given  (64, 64) 크기 이미지와 roi.fixed_coords=[0,0,256,256]이 주어졌을 때
When   extract_patches() 를 호출하면
Then   이미지 범위 내로 클리핑하여 패치를 반환하거나 빈 리스트([])를 반환한다
```

**Scenario 7: 단채널 → 3채널 변환 확인**
```
Given  Y채널 (grayscale float [0,1]) 이 주어졌을 때
When   3채널로 변환하면
Then   shape는 (H, W, 3)이고
And    세 채널의 값이 동일하다 (ch_uint8 을 3회 복제)
```

---

## 2. TDD 스펙

### 2.1 split_cmyk()

**테스트 파일**: `src/tests/unit/test_roi_extractor.py`

| 테스트 ID | 입력 | 기댓값 | 검증 포인트 |
| --- | --- | --- | --- |
| T-ROI-01 | 흰색 BGR 이미지 (255,255,255) | C=M=Y=K=0.0 | `np.allclose(result["C"], 0.0)` |
| T-ROI-02 | 검정 BGR 이미지 (0,0,0) | C=M=Y=K=1.0 | `np.allclose(result["K"], 1.0)` |
| T-ROI-03 | 순수 빨강 (B=0,G=0,R=255) → BGR=(0,0,255) | C=0.0, M=1.0, Y=1.0 | 채널별 검증 |
| T-ROI-04 | 랜덤 BGR (128,128,3) | 모든 채널 [0,1] 범위 | `assert (arr >= 0).all() and (arr <= 1).all()` |
| T-ROI-05 | 입력 dtype=uint8 | 반환 dtype=float32 | `assert result["C"].dtype == np.float32` |

```python
# 스펙 예시 — 구현 전 실패해야 함
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

| 테스트 ID | 입력 | 기댓값 | 검증 포인트 |
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

## 3. 통합 테스트 스펙

**테스트 파일**: `src/tests/integration/test_roi_pipeline.py`

| 테스트 ID | 시나리오 | 검증 포인트 |
| --- | --- | --- |
| T-ROI-INT-01 | 스캔 이미지 → CMYK 분리 → Y채널 패치 추출 → 저장 | 파일 시스템에 PNG 저장 확인 |
| T-ROI-INT-02 | 추출된 패치를 CMYKDataset이 로드 가능 | `CMYKDataset` 로드 후 tensor shape `(3,128,128)` |
| T-ROI-INT-03 | 추출된 패치를 GrayspotModel에 forward pass | 출력 shape `(1, 6)` |

---

## 4. Conftest 픽스처

```python
# src/tests/unit/conftest.py 에 추가할 픽스처

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

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | ROI 추출 정의 |
| [Contract_roi_pipeline.md](../Contract/Contract_roi_pipeline.md) | API 계약 |
