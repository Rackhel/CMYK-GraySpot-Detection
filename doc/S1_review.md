# S1_review


## Yelhas (R1)

####


## Koshoi & Jin-Hyeong Yang (R2, R5)

### 1.1 Koshoi

#### 

### 1.2 Jin-Hyeong Yang

#### 02_model_test.ipynb
EfficientNet-B0를 처음 로드해서 forward pass를 돌려봤을 때 출력 차원이 1280이 나오는 걸 확인했다. pretrained=True 방식이 deprecated 되어 있어서 weights=EfficientNet_B0_Weights.DEFAULT 방식으로 바꿔야 했는데, 버전마다 API가 달라질 수 있다는 걸 실감했다. Head를 nn.Identity()로 교체하는 방식이 단순하면서도 효과적이었다. ResNet-50과 비교했을 때 EfficientNet-B0가 파라미터 수는 훨씬 적지만 feature_dim도 달라서 Head를 고정 크기로 만들면 안 된다는 점을 직접 확인했다.

#### 03_training.ipynb
CrossEntropyLoss 안에 Softmax가 포함되어 있다는 걸 알면서도 Head에 실수로 Softmax를 붙이면 어떻게 되는지 직접 실험해봤다. 출력값이 이미 0~1 사이로 눌려있는 상태에서 다시 Softmax가 적용되면 gradient가 거의 0에 가까워져서 학습이 제대로 안 됐다. model.eval() 호출을 빠뜨렸을 때 같은 이미지를 여러 번 추론해도 결과가 달라지는 현상도 직접 겪었는데 BatchNorm과 Dropout이 train 모드에서는 확률적으로 동작한다는 걸 몸으로 느꼈다. 데이터가 적을 때 drop_last=True 설정 때문에 배치가 아예 안 만들어지는 상황도 겪었다.

#### 05_contrastive.ipynb
라벨 없이도 학습이 된다는 개념 자체가 처음에는 와닿지 않았다. 같은 이미지의 두 augmentation을 Positive Pair로 정의하고 InfoNCE Loss로 가깝게 당기는 방식을 직접 구현해보니까 왜 배치 크기가 클수록 좋은지 이해됐다. Negative Pair가 많아야 모델이 더 discriminative한 표현을 학습하기 때문이다. temperature τ 값을 0.5로 높였을 때와 0.1로 낮췄을 때 loss 수렴 속도가 눈에 띄게 달랐다. L2 정규화를 빠뜨렸을 때 유사도 행렬 값이 폭발적으로 커져서 loss가 nan이 되는 경험도 했다.

## Habin Ham (R3)

####


## Jeahwan Lee (R4)

#### I studied how to use Optuna because there is no part he was in charge of yet.
#### 아직 그가 담당한 파트가 없어 Optuna 사용법을 공부하였습니다.


## Rackhel (R6)

#### I made github repository for the team, and built the project structure
#### 팀을 위해 깃허브 저장소를 만들고 프로젝트 구조를 구축했습니다
