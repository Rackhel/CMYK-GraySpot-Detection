import optuna

# -------------------------
#  여기 나중에 팀 코드 넣기
# -------------------------
def train_model(learning_rate, batch_size, optimizer_name, dropout):

  
    score = 0.7

    if 0.0005 <= learning_rate <= 0.005:
        score += 0.1

    if batch_size == 32:
        score += 0.05

    if optimizer_name == "Adam":
        score += 0.03

    if 0.2 <= dropout <= 0.4:
        score += 0.02

    return score


# -------------------------
# Optuna 실행 함수
# -------------------------
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    score = train_model(
        learning_rate,
        batch_size,
        optimizer_name,
        dropout
    )

    return score


# -------------------------
# 실행
# -------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print("Best Params:", study.best_params)
print("Best Score:", study.best_value)