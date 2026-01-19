import json
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_from_log_file(checkpoint_path, save_path="./learning_curves_from_log.png"):
    # 1. trainer_state.json 파일 찾기
    log_file = os.path.join(checkpoint_path, "trainer_state.json")

    if not os.path.exists(log_file):
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data["log_history"]

    # 2. 로그 분리
    train_loss = []
    val_loss = []

    for entry in history:
        if "loss" in entry:  # 학습 손실
            train_loss.append({"epoch": entry["epoch"], "train_loss": entry["loss"]})
        if "eval_loss" in entry:  # 검증 손실
            val_loss.append({"epoch": entry["epoch"], "val_loss": entry["eval_loss"]})

    # 데이터프레임 변환 및 병합
    df_train = pd.DataFrame(train_loss)
    df_val = pd.DataFrame(val_loss)

    # 3. 그래프 시각화
    plt.figure(figsize=(10, 6))

    if not df_train.empty:
        plt.plot(df_train["epoch"], df_train["train_loss"], label="Train Loss", color="blue", alpha=0.6)
    if not df_val.empty:
        plt.plot(df_val["epoch"], df_val["val_loss"], label="Val Loss", color="red", marker="o")

    plt.title("Training and Validation Loss Curve", fontsize=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(save_path)
    print(f"✅ 그래프가 저장되었습니다: {save_path}")
    plt.show()
if __name__ == "__main__":
    # 🔴 본인의 실제 체크포인트 경로로 수정하세요
    # 예: "./checkpoints/checkpoint-1000" 또는 최신 체크포인트 폴더
    MY_CHECKPOINT_PATH = "./checkpoints/Phase_3_-_Full_Training"
    plot_from_log_file(MY_CHECKPOINT_PATH)