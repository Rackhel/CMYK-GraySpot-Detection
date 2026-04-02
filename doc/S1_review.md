# S1 Review / S1 리뷰

---

## 1. Yelhas (R1) — Data & Preprocessing / 데이터 및 전처리

### 1.1 S1 Experience / S1 경험

> To be filled in. / 작성 예정.

---

## 2. Koshoi & Jin-Hyeong Yang (R2, R5) — Model & Training / GUI

### 2.1 Koshoi (Team Leader)

> 팀 리더로서 Stage 1은 저에게 상당히 도전적인 단계였습니다. 특히 프로젝트 실행 계획과 팀 역할을 동시에 관리해야 했기 때문에 초기에는 부담이 컸습니다. 초반에는 팀 내 구조와 협업 방식이 명확하지 않아, 각 팀원이 개별적으로 작업을 진행하고 있었고, 서로 어떤 작업을 하고 있는지 파악하기 어려운 상황이었습니다. 이로 인해 전체적인 진행 속도가 느려졌습니다.

> 이 문제를 해결하기 위해 가장 먼저 체계적인 작업 환경을 구축하는 데 집중했습니다. 모든 팀원이 자신의 작업을 Notion에 기록하거나 저 또는 리포트 담당자에게 직접 공유하도록 시스템을 정리하였습니다. 이를 통해 팀 내 가시성과 협업 효율이 크게 향상되었습니다.

> 또한, 특정 역할에 대한 관리가 부족했던 점도 문제였습니다. 특히 데이터셋을 담당하는 R1 역할에 대해 충분한 확인과 관리가 이루어지지 않아, 데이터셋 준비에 약 2주가 소요되었습니다. 이 경험을 통해 작업 모니터링과 역할 관리의 중요성을 깨닫게 되었습니다.

> 하지만 시간이 지나면서 상황은 점점 개선되었습니다. 멘토님께서 제공해주신 실행 계획과 PRD 덕분에 프로젝트 방향이 명확해졌고, 팀 관리 또한 훨씬 수월해졌습니다.

> 전체적으로 Stage 1은 여러 어려움이 있었지만, 팀원들과 함께 많은 것을 배우는 과정이었습니다. 특히 협업 방식, 역할 분배, 프로젝트 관리 측면에서 큰 성장을 이룰 수 있었습니다. Stage 2에서는 작업 관리와 커뮤니케이션을 더욱 철저히 하여, 유사한 문제가 발생하지 않도록 개선할 계획입니다.

> As a team leader, Stage 1 was quite challenging for me due to the sudden increase in responsibility, especially in managing team roles and aligning the project with the execution plan. At the beginning, there was a lack of clear structure and coordination within the team. Each member was working independently, but there was no visibility into who was doing what, which slowed down our progress.

> To address this, my first priority was to establish a structured working environment. I introduced a system where all team members report their tasks through Notion or directly communicate updates to me or the members responsible for reporting. This significantly improved transparency and coordination.

> Another issue was insufficient supervision of specific roles, particularly the data-related tasks (R1). Due to this, dataset preparation took longer than expected (around two weeks). This highlighted a gap in task monitoring and responsibility tracking on my side.

> Despite these difficulties, the situation improved over time. With the mentor providing a clear execution plan and PRD, the project direction became more structured, and managing the team became more efficient.

> Overall, Stage 1 was completed with several organizational challenges, but it was a valuable learning phase. Both I and my teammates gained a better understanding of teamwork, responsibility distribution, and project management. Moving forward to Stage 2, I plan to maintain stricter control over task tracking and ensure better communication to avoid similar delays.

### 2.2 Jin-Hyeong Yang (R2) — S1 Experience / S1 경험

#### 02_model_test.ipynb
 
> EfficientNet-B0를 처음 로드해서 forward pass를 돌려봤을 때 출력 차원이 1280이 나오는 걸 확인했다. `pretrained=True` 방식이 deprecated 되어 있어서 `weights=EfficientNet_B0_Weights.DEFAULT` 방식으로 바꿔야 했는데, 버전마다 API가 달라질 수 있다는 걸 실감했다. Head를 `nn.Identity()`로 교체하는 방식이 단순하면서도 효과적이었다. ResNet-50과 비교했을 때 EfficientNet-B0가 파라미터 수는 훨씬 적지만 `feature_dim`도 달라서 Head를 고정 크기로 만들면 안 된다는 점을 직접 확인했다.

> When loading EfficientNet-B0 for the first time and running a forward pass, I confirmed that the output dimension was 1280. The `pretrained=True` argument was deprecated, so I had to switch to `weights=EfficientNet_B0_Weights.DEFAULT`. This made me realize that APIs can differ between versions. Replacing the head with `nn.Identity()` was simple yet effective. Compared with ResNet-50, I directly confirmed that EfficientNet-B0 has far fewer parameters but a different `feature_dim`, which means the head cannot be built with a fixed input size.
 
#### 03_training.ipynb

> `CrossEntropyLoss` 안에 Softmax가 포함되어 있다는 걸 알면서도 Head에 실수로 Softmax를 붙이면 어떻게 되는지 직접 실험해봤다. 출력값이 이미 0~1 사이로 눌려있는 상태에서 다시 Softmax가 적용되면 gradient가 거의 0에 가까워져서 학습이 제대로 안 됐다. `model.eval()` 호출을 빠뜨렸을 때 같은 이미지를 여러 번 추론해도 결과가 달라지는 현상도 직접 겪었는데 BatchNorm과 Dropout이 train 모드에서는 확률적으로 동작한다는 걸 몸으로 느꼈다. 데이터가 적을 때 `drop_last=True` 설정 때문에 배치가 아예 안 만들어지는 상황도 겪었다. 데이터가 적어 오류가 나는 것을 데이터 증강과 Oversampling을 통해 클래스 불균형을 해소하여 해결하였고, epochs를 10에서 30으로 늘림으로써 더 높은 정확도를 획득하였다.

> Although I knew that `CrossEntropyLoss` already includes Softmax internally, I deliberately attached Softmax to the head to see what would happen. Since the outputs were already compressed into the 0~1 range, applying Softmax again caused the gradient to approach nearly zero, preventing proper training. I also experienced a case where omitting `model.eval()` caused different results each time the same image was inferred — this made me feel firsthand that BatchNorm and Dropout behave stochastically in train mode. I also encountered a situation where the `drop_last=True` setting caused no batches to be created at all when data was scarce. Due to the limited dataset size, errors occurred during training. These were resolved by applying data augmentation combined with Oversampling to balance the class distribution. Additionally, increasing the number of epochs from 10 to 30 resulted in higher accuracy.
 
#### 05_contrastive.ipynb
 
> 라벨 없이도 학습이 된다는 개념 자체가 처음에는 와닿지 않았다. 같은 이미지의 두 augmentation을 Positive Pair로 정의하고 InfoNCE Loss로 가깝게 당기는 방식을 직접 구현해보니까 왜 배치 크기가 클수록 좋은지 이해됐다. Negative Pair가 많아야 모델이 더 discriminative한 표현을 학습하기 때문이다. temperature τ 값을 0.5로 높였을 때와 0.1로 낮췄을 때 loss 수렴 속도가 눈에 띄게 달랐다. L2 정규화를 빠뜨렸을 때 유사도 행렬 값이 폭발적으로 커져서 loss가 nan이 되는 경험도 했다.

> The concept of learning without labels was hard to grasp at first. But after directly implementing the approach of defining two augmentations of the same image as a Positive Pair and pulling them together with InfoNCE Loss, I understood why a larger batch size is better — more Negative Pairs allow the model to learn more discriminative representations. There was also a noticeable difference in loss convergence speed when the temperature τ was raised to 0.5 versus lowered to 0.1. I also experienced the loss becoming NaN when L2 normalization was omitted, because the similarity matrix values exploded.

---

## 3. Habin Ham (R3) — Evaluation & Reporting / 평가 및 리포팅

### 3.1 S1 Experience / S1 경험

> 이번에 Confusion Matrix와 Accuracy를 출력하는 Notebook과 t-SNE/UMAP 임베딩 시각화 Notebook을 담당하면서 코딩을 진행했다. 이전까지 관련 경험이 많지 않았고 지식도 부족해서 프로젝트를 따라가는 것이 솔직히 쉽지 않았다. 하지만 하나하나 오류를 수정하고 결과값이 정상적으로 나올 때마다 큰 만족감을 느꼈고, 점점 재미를 느끼며 작업을 이어갈 수 있었다.

> This time, I worked on implementing notebooks for Confusion Matrix and Accuracy output and t-SNE/UMAP embedding visualization. Since I had limited prior experience and knowledge in this area, it was honestly challenging to keep up with the project. However, as I fixed errors step by step and started getting proper results, I felt a strong sense of accomplishment, which made the process increasingly enjoyable.

---

## 4. Jeahwan Lee (R4) — Tuning & Optimization / 튜닝 및 최적화

### 4.1 S1 Experience / S1 경험

> 아직 담당한 파트가 없어 Optuna 사용법을 공부하였습니다.

> Since there was no assigned part yet, I studied how to use Optuna.

---

## 5. Rackhel (R6) — Integration & Infrastructure / 통합 및 인프라

### 5.1 S1 Experience / S1 경험

> 팀을 위해 깃허브 저장소를 만들고 프로젝트 구조를 구축했습니다.

> I created the GitHub repository for the team and built the project structure.
