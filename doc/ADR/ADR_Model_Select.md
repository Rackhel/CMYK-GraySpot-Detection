# ADR — Backbone 및 ClassifierHead 설계 결정
# ADR — Backbone and ClassifierHead Design Decisions

---

## 1. 배경 / Background

CMYK Grayspot Detection System은 인쇄물의 그레이스팟 결함을 0~5 단계(6-class)로 분류한다.
CMYK Grayspot Detection System classifies printer grayspot defects into 6 levels (0~5).

- 입력 크기 / Input size: `128 × 128` BGR 이미지
- 채널 수 / Channels: 4개 독립 학습 (Y, M, C, K) / 4 independently trained channels
- 학습 패러다임 / Training paradigm: Phase 0 SimCLR → Phase 2 Supervised (Swing Architecture)
- 데이터 규모 / Dataset size: 소규모 인쇄 결함 도메인 / Small-scale print defect domain
- 사전학습 방식 / Pretraining: ImageNet pretrained weights → Phase 0 domain adaptation → Phase 2 fine-tuning

---

## 2. 결정 사항 / Decisions

### 2.1 Backbone 선택 — EfficientNet-B0 vs ResNet-50 병행 지원
### 2.1 Backbone Selection — Dual Support for EfficientNet-B0 and ResNet-50

**결정 / Decision**: 단일 backbone 고정이 아닌, `config.json`의 `model.backbone` 으로 두 backbone을 모두 지원한다.
**Decision**: Rather than fixing a single backbone, support both backbones via `model.backbone` in `config.json`.

### 2.2 ClassifierHead 구조 — Backbone별 특화 설계
### 2.2 ClassifierHead Structure — Backbone-Specific Architecture

**결정 / Decision**: ClassifierHead를 backbone에 따라 다른 구조로 특화한다.

| Backbone | ClassifierHead 구조 / Structure | `mid_dim` |
|---|---|---|
| `efficientnet_b0` | `1280 → hidden_dim → 6` (직접 압축 / Direct) | None |
| `resnet50` | `2048 → mid_dim → hidden_dim → 6` (단계적 압축 / Staged) | 512 |

---

## 3. 선택 이유 / Rationale

### 3.1 EfficientNet-B0 선택 근거 / Why EfficientNet-B0

| 항목 / Item | 내용 / Detail |
|---|---|
| **SE-Attention 내재** / Built-in SE-Attention | MBConv 블록마다 채널별 중요도를 `Sigmoid`로 게이팅 — 소규모 데이터에서 유의미한 채널 선택 효과 / Per-channel gating via `Sigmoid` in each MBConv block — effective channel selection on small data |
| **파라미터 효율** / Parameter efficiency | ~5.3M params (ResNet-50 대비 약 5배 적음) — 소규모 데이터셋 과적합 위험 낮춤 / ~5.3M params (≈5× fewer than ResNet-50) — reduces overfitting risk on small datasets |
| **직접 압축 head 적합** / Fits direct-compression head | SE가 이미 중요 채널 선별 완료 → 1280차원에서 바로 압축해도 정보 손실 최소화 / SE already selects important channels → minimal information loss with direct compression from 1280-dim |
| **ImageNet 호환** / ImageNet compatible | `EfficientNet_B0_Weights.DEFAULT` pretrained weights — ImageNet 통계 정규화와 직접 호환 / Compatible with ImageNet statistics normalization |

### 3.2 ResNet-50 선택 근거 / Why ResNet-50

| 항목 / Item | 내용 / Detail |
|---|---|
| **검증된 아키텍처** / Proven architecture | Bottleneck block 기반 hierachical feature 추출 — 산업 결함 분류에서 광범위하게 검증됨 / Bottleneck-based hierarchical features — extensively validated in industrial defect classification |
| **풍부한 feature space** / Rich feature space | 2048차원 feature — 세밀한 그레이스팟 텍스처 패턴 포착 가능성 / 2048-dim features — potential to capture fine-grained grayspot texture patterns |
| **channel-generic features** / Channel-generic features | SE 없이 모든 채널을 비선별적으로 전달 → head에서 단계적 선별 구조가 필요 / All channels passed without selection → staged compression head needed |

### 3.3 EfficientNet-B0 직접 압축 Head 근거 / Why Direct Compression for EfficientNet-B0

SE-attention이 backbone 내부에서 채널 중요도를 이미 Sigmoid로 게이팅한다. 출력 1280차원은 이미 정제된(refined) feature다. 단일 Linear 압축 `1280 → 256`이 적합하며, 과도한 중간 레이어 추가는 소규모 데이터에서 과적합을 유발한다.

SE-attention already gates channel importance via Sigmoid inside the backbone. The 1280-dim output is already refined. A single linear compression `1280 → 256` is appropriate; adding excessive intermediate layers risks overfitting on small datasets.

### 3.4 ResNet-50 단계적 압축 Head 근거 / Why Staged Compression for ResNet-50

ResNet-50은 SE 없이 2048차원을 비선별적으로 출력한다. Head가 채널 선별 역할까지 담당해야 하며, `2048 → 256` 단번 압축은 정보 손실이 크다. `2048 → 512 → 256` 단계적 압축이 점진적으로 중요 특징을 유지하면서 압축한다.

ResNet-50 outputs 2048-dim features without channel selection. The head must also perform channel selection; direct `2048 → 256` compression causes large information loss. Staged `2048 → 512 → 256` progressively preserves important features while compressing.

---

## 4. 고려한 대안 / Considered Alternatives

### 4.1 단일 backbone 고정 / Single Fixed Backbone

| 대안 / Alternative | 채택하지 않은 이유 / Why Not Chosen |
|---|---|
| EfficientNet-B0만 사용 / EffB0 only | 성능 비교 실험 불가 — ResNet-50의 2048 feature space 활용 가능성 배제 / Prevents performance comparison — excludes ResNet-50's 2048 feature space |
| ResNet-50만 사용 / ResNet-50 only | SE-attention의 채널 선별 이점 포기, 파라미터 5배 더 많아 소규모 데이터에서 불리 / Forgoes SE channel selection benefit; 5× more params disadvantageous for small data |

### 4.2 동일 head 구조 공용 / Shared Head Structure for Both Backbones

**채택하지 않은 이유 / Why Not Chosen**: 동일 `hidden_dim=256` head를 공용하면 ResNet-50의 2048→256 단번 압축에서 정보 손실이 크다. EfficientNet-B0에서는 SE 이후 추가 중간 레이어가 불필요한 과파라미터화를 유발한다.

**Why Not Chosen**: Sharing the same `hidden_dim=256` head causes large information loss in ResNet-50's one-shot `2048→256` compression. For EfficientNet-B0, the additional intermediate layer creates unnecessary over-parameterization after SE.

### 4.3 ViT / Swin Transformer

**채택하지 않은 이유 / Why Not Chosen**: Self-Attention의 `O(N²·d)` 연산 비용이 `128×128` 입력 + 소규모 데이터 환경에서 과도하다. Pretrained ViT의 position embedding이 128×128 해상도와 호환성 문제 발생 가능.

**Why Not Chosen**: Self-Attention `O(N²·d)` compute cost is excessive for `128×128` input + small dataset. Pretrained ViT position embeddings may have compatibility issues with 128×128 resolution.

---

## 5. SE-Attention vs 일반 Attention 비교 / SE-Attention vs General Attention

| 항목 / Item | SE-Attention (EfficientNet-B0) | Self-Attention (Transformer) |
|---|---|---|
| **범위 / Scope** | 채널만 / Channel only | 공간 + 채널 / Spatial + channel |
| **게이팅 / Gating** | `Sigmoid` (0~1 스케일) | `Softmax` (분포 합산 = 1) |
| **연산 / Complexity** | `O(C²/r)` — 경량 / Lightweight | `O(N²·d)` — 공간 크기에 제곱 비례 / Quadratic in spatial size |
| **Q/K/V 분해 / Decomposition** | 없음 / None | Q, K, V 선형 변환 / Linear projection |
| **위치 정보 / Position** | 미사용 (채널 통계만) / Unused (channel stats only) | Position embedding 필요 / Requires position embedding |
| **소규모 데이터 / Small data** | ✅ 적합 / Suitable | ❌ 과적합 위험 / Overfitting risk |

> SE-Attention은 `Global Average Pooling → FC → ReLU → FC → Sigmoid` 구조로 채널별 중요도를 학습한다.
> SE-Attention learns per-channel importance via `GAP → FC → ReLU → FC → Sigmoid`.

---

## 6. 결과 및 트레이드오프 / Consequences and Trade-offs

### 채택 시 이점 / Benefits of This Decision

- Backbone 비교 실험이 `config.json`의 `model.backbone` 변경만으로 가능
- ResNet-50의 2048 feature space를 단계적 head로 효과적으로 활용 가능
- EfficientNet-B0의 SE 특성에 맞는 compact head로 과적합 위험 최소화
- `phase2.heads.{backbone}` config 구조로 Optuna가 backbone별 독립 탐색 공간 사용 가능

### 비용 및 주의사항 / Costs and Cautions

- ResNet-50 ClassifierHead 파라미터 수 증가 → 소규모 데이터에서 Optuna dropout 탐색 필수
- `phase2.heads.{backbone}.hidden_dim` / `mid_dim` 변경 시 **재학습 필수** (Hard SSOT)
- Backbone 변경 시 Phase 0 backbone artifact와 Phase 2 model의 Channel Invariant 반드시 준수 (SSOT-FF01)

---

## 7. 관련 코드 및 문서 / Related Code and Documents

| 참조 / Reference | 내용 / Content |
|---|---|
| [`src/models/classifier.py`](../src/models/classifier.py) | `ClassifierHead` — `mid_dim` 파라미터로 backbone별 구조 분기 / backbone-specific structure branch via `mid_dim` parameter |
| [`src/models/grayspot_model.py`](../src/models/grayspot_model.py) | `phase2.heads.{backbone}` config 읽어 `ClassifierHead` 생성 / reads `phase2.heads.{backbone}` config to construct `ClassifierHead` |
| [`src/tuning/search_space.py`](../src/tuning/search_space.py) | backbone별 Optuna 탐색 공간 분기 / Optuna search space branched per backbone |
| [`src/config/config.json`](../src/config/config.json) | `phase2.heads`, `optuna.search_space.{backbone}` 정의 / defines `phase2.heads` and `optuna.search_space.{backbone}` |
| [SSOT_Model_Architecture.md](SSOT_Model_Architecture.md) | §5 ClassifierHead 상세 구조 / §5 ClassifierHead detailed structure |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | §3.3 backbone별 파라미터, §7 Optuna 탐색 공간 / §3.3 per-backbone parameters, §7 Optuna search space |
| [SSOT_GlobalVariables.md](SSOT_GlobalVariables.md) | Hard SSOT `mid_dim`, `hidden_dim` 분류 / Hard SSOT classification of `mid_dim`, `hidden_dim` |

---
