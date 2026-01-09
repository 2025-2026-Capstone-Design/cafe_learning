import pandas as pd
import torch
import torch.nn as nn
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ==========================================
# 0. 설정 및 하이퍼파라미터
# ==========================================
MODEL_NAME = "klue/bert-base"
DATA_PATH = "final_data.csv"
SAVE_PATH = "./final_cafe_model"
CHECKPOINT_PATH = "./checkpoints"  # 🆕 체크포인트 저장 경로
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# 🔴 중요: 모델에게 12개 숫자의 의미를 알려주는 매핑 테이블
ASPECT_NAMES = [
    "커피/음료", "베이커리/빵", "케이크", "쿠키/구움과자",
    "빙수/과일", "기타 디저트", "공간/편의시설", "분위기/감성",
    "서비스", "가격/가성비", "선물/포장", "혼잡도/웨이팅"
]

SENTIMENT_NAMES = {
    0: "해당없음",
    1: "긍정",
    2: "부정",
    3: "중립"
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# ==========================================
# 1. 데이터셋 클래스
# ==========================================
class CafeAspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label_str = str(self.labels[item]).zfill(12)
        label_tensor = torch.tensor([int(c) for c in label_str], dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


# ==========================================
# 2. 멀티 출력 모델 정의
# ==========================================
class MultiOutputBert(nn.Module):
    def __init__(self, model_name):
        super(MultiOutputBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 12 * 4)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        logits = self.out(output).view(-1, 12, 4)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0
            for i in range(12):
                loss += loss_fct(logits[:, i, :], labels[:, i])

        return {'loss': loss, 'logits': logits}


# ==========================================
# 3. 커스텀 트레이너
# ==========================================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 4. 예측 함수
# ==========================================
def predict_review(model, tokenizer, review_text, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=2).flatten().tolist()

    print(f"\n📝 리뷰: {review_text}")
    print("-" * 40)

    detected = False
    for idx, label in enumerate(preds):
        if label != 0:
            aspect = ASPECT_NAMES[idx]
            sentiment = SENTIMENT_NAMES[label]
            print(f"👉 [{aspect}] : {sentiment}")
            detected = True

    if not detected:
        print("👉 분석된 감성이 없습니다.")
    print("-" * 40)


# ==========================================
# 메인 실행 코드
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 로드
    print("📂 데이터 로딩 중...")
    try:
        df = pd.read_csv(DATA_PATH, dtype={'label': str})  # 🔧 수정됨
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
        exit()

    print(f"총 데이터 개수: {len(df)}개")

    # 🆕 2. 데이터 분할: Train(4000) / Val(500) / Test(500)
    # 먼저 Train + Val vs Test 분리
    temp_texts, test_texts, temp_labels, test_labels = train_test_split(
        df['Original_Review'].tolist(),
        df['label'].tolist(),
        test_size=500,  # 테스트셋 500개 고정
        random_state=SEED,
        shuffle=True
    )

    # 남은 데이터에서 Train vs Val 분리
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=500,  # 검증셋 500개 고정
        random_state=SEED,
        shuffle=True
    )

    print(f"학습 데이터: {len(train_texts)}개")
    print(f"검증 데이터: {len(val_texts)}개")
    print(f"테스트 데이터: {len(test_texts)}개")

    # 3. 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)  # 🆕

    # 4. 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiOutputBert(MODEL_NAME).to(device)

    # 🆕 5. 체크포인트에서 재개할지 확인
    resume_from_checkpoint = None
    if os.path.exists(CHECKPOINT_PATH):
        checkpoints = [d for d in os.listdir(CHECKPOINT_PATH) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            resume_from_checkpoint = os.path.join(CHECKPOINT_PATH, latest_checkpoint)
            print(f"🔄 체크포인트에서 재개: {resume_from_checkpoint}")

    # 6. 학습 설정
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_PATH,  # 🔧 체크포인트 경로로 변경
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # 🆕 최고 모델 기준
        report_to="none",
        save_total_limit=3,  # 🔧 최근 3개 체크포인트 보관
        resume_from_checkpoint=resume_from_checkpoint  # 🆕 재개 설정
    )

    # 7. 학습 시작
    print("🚀 학습 시작...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 8. 최종 모델 저장
    print(f"💾 최종 모델 저장 중... ({SAVE_PATH})")
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_PATH}/model_state_dict.pt")
    tokenizer.save_pretrained(SAVE_PATH)

    # ----------------------------------------
    # 9. 🆕 테스트 데이터셋으로 최종 평가
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("📊 테스트 데이터셋으로 최종 평가 시작")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=2)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 측면별 정확도
    print("\n📈 측면별 정확도 (테스트셋):")
    print("-" * 50)
    for i, aspect_name in enumerate(ASPECT_NAMES):
        correct = (all_preds[:, i] == all_labels[:, i]).sum().item()
        total = all_labels.shape[0]
        accuracy = correct / total * 100
        print(f"{aspect_name:15s}: {accuracy:.2f}%")

    total_correct = (all_preds == all_labels).sum().item()
    total_elements = all_labels.numel()
    overall_accuracy = total_correct / total_elements * 100
    print("-" * 50)
    print(f"전체 정확도: {overall_accuracy:.2f}%")

    # ----------------------------------------
    # 10. 샘플 예측
    # ----------------------------------------
    print("\n🔍 학습된 모델로 샘플 테스트")

    test_reviews = [
        "커피는 진짜 맛있는데 직원이 좀 불친절해서 기분 나빴음",
        "케이크랑 커피 둘 다 너무 맛있고 분위기도 짱 좋아요!",
        "가격은 비싼데 맛은 그냥 편의점 수준이네요 실망입니다.",
        "웨이팅이 너무 길어서 힘들었지만 소금빵 먹자마자 용서됨"
    ]

    for review in test_reviews:
        predict_review(model, tokenizer, review, device)