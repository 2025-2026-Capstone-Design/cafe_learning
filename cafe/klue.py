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
DATA_PATH = "final_data.csv"  # 🔴 본인의 데이터 파일명으로 변경!
SAVE_PATH = "./final_cafe_model"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# 🔴 중요: 모델에게 12개 숫자의 의미를 알려주는 매핑 테이블
ASPECT_NAMES = [
    "커피/음료",  # 1번째 (인덱스 0)
    "베이커리/빵",  # 2번째
    "케이크",  # 3번째
    "쿠키/구움과자",  # 4번째
    "빙수/과일",  # 5번째
    "기타 디저트",  # 6번째
    "공간/편의시설",  # 7번째
    "분위기/감성",  # 8번째
    "서비스",  # 9번째
    "가격/가성비",  # 10번째
    "선물/포장",  # 11번째
    "혼잡도/웨이팅"  # 12번째 (인덱스 11)
]

SENTIMENT_NAMES = {
    0: "해당없음",
    1: "긍정",
    2: "부정",
    3: "중립"
}


# 랜덤 시드 고정 (재현성 확보)
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
# 1. 데이터셋 클래스 (12자리 숫자 파싱)
# ==========================================
class CafeAspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels  # 12자리 문자열 리스트
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # 0이 잘려도 강제로 12자리로 맞춤 (예: '1100' -> '000000001100')
        label_str = str(self.labels[item]).zfill(12)

        # 문자열을 숫자 리스트로 변환 (예: "102..." -> [1, 0, 2...])
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
# 2. 멀티 출력 모델 정의 (Multi-Output Model)
# ==========================================
class MultiOutputBert(nn.Module):
    def __init__(self, model_name):
        super(MultiOutputBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)

        # 12개의 측면 * 4개의 감정(0,1,2,3) = 48개 출력
        self.out = nn.Linear(self.bert.config.hidden_size, 12 * 4)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        # [Batch, 48] -> [Batch, 12, 4] 형태로 변환
        logits = self.out(output).view(-1, 12, 4)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0
            # 12개의 측면에 대해 각각 Loss를 구해서 더함
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
# 4. 예측(테스트) 함수 정의
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
        # 확률이 가장 높은 클래스(0~3) 선택
        preds = torch.argmax(logits, dim=2).flatten().tolist()

    print(f"\n📝 리뷰: {review_text}")
    print("-" * 40)

    detected = False
    for idx, label in enumerate(preds):
        if label != 0:  # 0(해당없음)이 아닌 경우만 출력
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
    # 1. 데이터 로드 (0 잘림 방지를 위해 dtype=str 필수)
    print("📂 데이터 로딩 중...")
    try:
        df = pd.read_csv(DATA_PATH, dtype={'label': str})
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
        exit()

    # 데이터 분할
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Original_Review'].tolist(),
        df['label'].tolist(),
        test_size=0.1,
        random_state=SEED
    )

    # 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiOutputBert(MODEL_NAME).to(device)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir='./results',
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
        report_to="none",
        save_total_limit=2
    )

    # 학습 시작
    print("🚀 학습 시작...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    # 모델 저장
    print(f"💾 모델 저장 중... ({SAVE_PATH})")
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_PATH}/model_state_dict.pt")
    tokenizer.save_pretrained(SAVE_PATH)

    # ----------------------------------------
    # [테스트] 실제 예측 해보기
    # ----------------------------------------
    print("\n🔍 학습된 모델로 테스트를 진행합니다.")

    test_reviews = [
        "커피는 진짜 맛있는데 직원이 좀 불친절해서 기분 나빴음",
        "케이크랑 커피 둘 다 너무 맛있고 분위기도 짱 좋아요!",
        "가격은 비싼데 맛은 그냥 편의점 수준이네요 실망입니다.",
        "웨이팅이 너무 길어서 힘들었지만 소금빵 먹자마자 용서됨"
    ]

    for review in test_reviews:
        predict_review(model, tokenizer, review, device)

    # ----------------------------------------
    # 👇 여기부터 추가!
    # ----------------------------------------
    print("\n📊 검증 데이터셋 평가 중...")
    from sklearn.metrics import classification_report

    model.eval()
    all_preds = []
    all_labels = []

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in val_loader:
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

    # 📊 1. 측면별 정확도
    print("\n📈 측면별 정확도:")
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
    print(f"전체 정확도: {overall_accuracy:.2f}%\n")

    # 📊 2. 측면별 F1-Score
    print("\n📊 측면별 상세 평가 (F1-Score):")
    print("=" * 70)
    for i, aspect_name in enumerate(ASPECT_NAMES):
        y_true = all_labels[:, i].numpy()
        y_pred = all_preds[:, i].numpy()

        print(f"\n[ {aspect_name} ]")
        print(classification_report(y_true, y_pred,
                                    target_names=["해당없음", "긍정", "부정", "중립"],
                                    zero_division=0))

    # 📊 3. 측면 감지율
    print("\n📊 측면 감지 통계:")
    print("-" * 50)
    detected_true = (all_labels != 0).sum(dim=1)
    detected_pred = (all_preds != 0).sum(dim=1)
    print(f"평균 실제 측면 수: {detected_true.float().mean():.2f}")
    print(f"평균 예측 측면 수: {detected_pred.float().mean():.2f}")
