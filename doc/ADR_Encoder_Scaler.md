# Encoder · Scaler 미사용 설계 근거
# Why Encoder and Scaler Are Not Used

---

## 1. Label Encoder가 필요 없는 이유
## 1. Why Label Encoder Is Not Needed

이 프로젝트의 라벨은 **이미 정수(0~5)** 입니다.  
The labels in this project are **already integers (0~5)**.

폴더 구조가 곧 라벨 정의입니다.  
The folder structure itself defines the labels.

```
data_set/labeled/Y/0/*.png   →  level = 0
data_set/labeled/Y/1/*.png   →  level = 1
data_set/labeled/Y/2/*.png   →  level = 2
...
data_set/labeled/Y/5/*.png   →  level = 5
```

`CMYKDataset`이 폴더명을 그대로 `int`로 읽고,  
PyTorch `CrossEntropyLoss`는 정수 라벨을 직접 입력받습니다.

`CMYKDataset` reads the folder name directly as `int`,  
and PyTorch `CrossEntropyLoss` accepts integer labels natively.

> `sklearn.LabelEncoder`는 `["cat", "dog"]` 같은 **문자열 라벨을 정수로 변환**할 때 필요한 도구입니다.  
> 라벨이 처음부터 정수인 이 프로젝트에서는 해당되지 않습니다.
>
> `sklearn.LabelEncoder` is a tool for **converting string labels** (e.g., `["cat", "dog"]`) into integers.  
> It is not applicable here since labels are integers from the start.

---

## 2. Scaler가 필요 없는 이유
## 2. Why Feature Scaler Is Not Needed

### 2.1 `/255.0` 자체가 MinMax Scaling
### 2.1 `/255.0` Is Already MinMax Scaling

`preprocessing.py`의 정규화가 이미 스케일링 역할을 합니다.  
The normalization in `preprocessing.py` already serves as scaling.

```python
# data/preprocessing.py
image = image.astype(np.float32) / 255.0   # [0, 255] → [0.0, 1.0]
```

이는 `MinMaxScaler`와 동일한 효과입니다.  
This has the same effect as `MinMaxScaler`.

```
MinMaxScaler 공식 / MinMaxScaler formula:
  X_scaled = (X - X_min) / (X_max - X_min)
           = (X - 0) / (255 - 0)
           = X / 255
```

### 2.2 CNN에서 추가 Scaler가 불필요한 3가지 이유
### 2.2 Three Reasons Additional Scaler Is Unnecessary in CNNs

| 이유 / Reason | 설명 / Description |
|---------------|---------------------|
| **픽셀 범위 고정** / Fixed pixel range | 이미지 픽셀은 항상 [0, 255] → [0, 1]로 범위가 명확 / Image pixels always have a known range [0, 255] → [0, 1] |
| **BatchNorm 내장** / Built-in BatchNorm | EfficientNet · ResNet 내부에 BatchNorm이 있어 레이어별 분포를 자동 정규화 / EfficientNet and ResNet have internal BatchNorm that automatically normalizes layer-wise distributions |
| **가중치 적응** / Weight adaptation | CNN이 학습 과정에서 가중치를 통해 스케일을 스스로 학습함 / The CNN learns to adapt to the input scale through its weights during training |

---

## 3. 추가 고려 사항 — ImageNet 표준화
## 3. Additional Consideration — ImageNet Normalization

현재 코드는 `EfficientNet_B0_Weights.DEFAULT`(ImageNet 사전학습 가중치)를 사용하지만,  
ImageNet 표준화는 적용하지 않고 있습니다.

The current code uses `EfficientNet_B0_Weights.DEFAULT` (ImageNet pretrained weights),  
but does not apply ImageNet normalization.

```python
# 현재 방식 / Current approach
image / 255.0                        # [0, 1] 정규화만 적용 / Only [0, 1] normalization

# ImageNet 사전학습 기준 권장 방식 / Recommended for ImageNet pretrained weights
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
(image - mean) / std                 # ImageNet 통계 기반 표준화 / ImageNet statistics normalization
```

### 현재 방식이 실용적으로 허용되는 이유
### Why the Current Approach Is Practically Acceptable

| 항목 / Item | 내용 / Detail |
|-------------|---------------|
| **도메인 차이** / Domain difference | CMYK 프린터 결함 이미지는 ImageNet과 도메인이 완전히 다름 / CMYK printer defect images are in a completely different domain from ImageNet |
| **Phase 0 재학습** / Phase 0 retraining | SimCLR Contrastive Learning이 backbone을 도메인 데이터로 재학습시킴 / SimCLR Contrastive Learning retrains the backbone on domain-specific data |
| **적용 시점** / When to apply | 성능을 끝까지 쥐어짜야 할 마지막 단계에서 검토 / Consider applying during final optimization for maximum performance |

---

## 4. 요약
## 4. Summary

```
전통적 ML 파이프라인 / Traditional ML pipeline:
  Raw data → LabelEncoder → StandardScaler → Model

이 프로젝트 / This project:
  Raw image → /255.0 (MinMax) → CNN (BatchNorm 내장) → CrossEntropyLoss(정수 라벨)
```

- **LabelEncoder**: 라벨이 이미 정수(폴더명 0~5)이므로 불필요  
  **LabelEncoder**: Not needed — labels are already integers (folder names 0~5)

- **Scaler**: `/255.0`이 MinMax Scaling을 대체하며, CNN 내부 BatchNorm이 추가 정규화를 처리  
  **Scaler**: `/255.0` replaces MinMax Scaling, and CNN's internal BatchNorm handles additional normalization
