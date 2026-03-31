# S1 Review / S1 리뷰

---

## 1. Yelhas (R1) — Data & Preprocessing / 데이터 및 전처리

### 1.1 S1 Experience / S1 경험

> To be filled in. / 작성 예정.

---

## 2. Koshoi & Jin-Hyeong Yang (R2, R5) — Model & Training / GUI

### 2.1 Koshoi (Team Leader)

> To be filled in. / 작성 예정.

### 2.2 Jin-Hyeong Yang (R2) — S1 Experience / S1 경험

#### 02_model_test.ipynb
> EfficientNet-B0 forward pass 확인 및 pretrained API 변경 대응. Head를 nn.Identity()로 교체하고 ResNet-50과 feature_dim 차이를 직접 확인했다.
> Confirmed EfficientNet-B0 forward pass output and updated deprecated pretrained API. Replaced head with nn.Identity() and verified feature_dim difference from ResNet-50.

#### 03_training.ipynb

> CrossEntropyLoss에 Softmax 중복 적용 시 gradient 소실 현상과 model.eval() 누락 시 추론 결과 불일치 문제를 직접 실험으로 확인했다.
> Verified gradient vanishing from double Softmax with CrossEntropyLoss, and confirmed inconsistent inference results when model.eval() was omitted.

#### 05_contrastive.ipynb

> Positive Pair 기반 InfoNCE Loss 구현을 통해 배치 크기와 temperature τ가 학습에 미치는 영향을 확인하고, L2 정규화 누락 시 loss NaN 발생을 경험했다.
> Implemented InfoNCE Loss with Positive Pairs and observed the impact of batch size and temperature τ, while experiencing loss NaN when L2 normalization was omitted.

---

## 3. Habin Ham (R3) — Evaluation & Reporting / 평가 및 리포팅

### 3.1 S1 Experience / S1 경험

> 이번에 Confusion Matrix와 Accuracy를 출력하는 Notebook과 t-SNE/UMAP 임베딩 시각화 Notebook을 담당하면서 코딩을 진행했다. 이전까지 관련 경험이 많지 않았고 지식도 부족해서 프로젝트를 따라가는 것이 솔직히 쉽지 않았다. 하지만 하나하나 오류를 수정하고 결과값이 정상적으로 나올 때마다 큰 만족감을 느꼈고, 점점 재미를 느끼며 작업을 이어갈 수 있었다.

> This time, I worked on implementing notebooks for Confusion Matrix and Accuracy output and t-SNE/UMAP embedding visualization. Since I had limited prior experience and knowledge in this area, it was honestly challenging to keep up with the project. However, as I fixed errors step by step and started getting proper results, I felt a strong sense of accomplishment, which made the process increasingly enjoyable.

---

## 4. Jeahwan Lee (R4) — Tuning & Optimization / 튜닝 및 최적화

### 4.1 S1 Experience / S1 경험

> Since there was no assigned part yet, I studied how to use Optuna.
> 아직 담당한 파트가 없어 Optuna 사용법을 공부하였습니다.

---

## 5. Rackhel (R6) — Integration & Infrastructure / 통합 및 인프라

### 5.1 S1 Experience / S1 경험

> I created the GitHub repository for the team and built the project structure.
> 팀을 위해 깃허브 저장소를 만들고 프로젝트 구조를 구축했습니다.
