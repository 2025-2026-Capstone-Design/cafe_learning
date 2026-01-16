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
CHECKPOINT_PATH = "./checkpoints"  # 체크포인트 저장 경로
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# 중요: 모델에게 12개 숫자의 의미를 알려주는 매핑 테이블
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
# 🆕 Focal Loss 정의
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1.5, weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight  # 클래스 가중치
        self.gamma = gamma  # 어려운 샘플 집중도 (2.0 권장)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce_loss(inputs, targets)
        pt = torch.exp(log_pt)
        loss = ((1 - pt) ** self.gamma) * self.ce_loss(inputs, targets)
        return loss.mean()
# ==========================================
# 2. 멀티 출력 모델 정의
# ==========================================
# class MultiOutputBert(nn.Module):
#     def __init__(self, model_name):
#         super(MultiOutputBert, self).__init__()
#         self.bert = BertModel.from_pretrained(model_name)
#         self.drop = nn.Dropout(p=0.3)
#         self.out = nn.Linear(self.bert.config.hidden_size, 12 * 4)
#     def forward(self, input_ids, attention_mask, labels=None,class_weights=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         output = self.drop(pooled_output)
#         logits = self.out(output).view(-1, 12, 4)
#
#         loss = None
#         if labels is not None:
#             if class_weights is not None:
#                 weights = class_weights.to(labels.device)
#             else:
#                 weights = None
#             loss_fct = FocalLoss(gamma=1.5,weight=weights)
#             loss = 0
#             for i in range(12):
#                 loss += loss_fct(logits[:, i, :], labels[:, i])
#
#         return {'loss': loss, 'logits': logits}

# ==========================================
# 🆕 Dual-Head 모델 정의 (기존 91-113번 줄 교체)
# ==========================================
class DualHeadBert(nn.Module):
    def __init__(self, model_name):
        super(DualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)

        # Head 1: 측면 감지 (12개 측면에 대해 있다/없다)
        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)

        # Head 2: 감성 분류 (12개 측면 × 3개 감성 = 36)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 12 * 3)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        # 측면 감지 (sigmoid)
        aspect_logits = self.aspect_head(output)  # [batch, 12]

        # 감성 분류
        sentiment_logits = self.sentiment_head(output).view(-1, 12, 3)  # [batch, 12, 3]

        loss = None
        if labels is not None:
            # 라벨 분리
            aspect_labels = (labels != 0).float()  # [batch, 12]
            sentiment_labels = torch.where(
                labels > 0,
                labels - 1,  # 1→0, 2→1, 3→2
                torch.zeros_like(labels)
            )

            # Loss 1: 측면 감지
            aspect_loss_fct = nn.BCEWithLogitsLoss()
            aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)

            # Loss 2: 감성 분류
            if class_weights is not None:
                sentiment_weights = class_weights.to(labels.device)  # 긍정, 부정, 중립만
            else:
                sentiment_weights = None

            sentiment_loss_fct = FocalLoss(gamma=1.5, weight=sentiment_weights)
            sentiment_loss = 0

            for i in range(12):
                mask = aspect_labels[:, i] == 1
                if mask.sum() > 0:
                    sentiment_loss += sentiment_loss_fct(
                        sentiment_logits[mask, i, :],
                        sentiment_labels[mask, i]
                    )

            # 전체 Loss
            loss = aspect_loss + 2.0 * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }


# ==========================================
# 🆕 CustomTrainer 수정 (기존 116-124번 줄 교체)
# ==========================================
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_sampler = train_sampler

    def get_train_dataloader(self):
        """🆕 WeightedRandomSampler 사용"""
        if self.train_sampler is not None:
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=self.train_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels, class_weights=self.class_weights)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 🆕 Dual-Head 예측 함수 (기존 127-158번 줄 교체)
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

        # 측면 감지
        aspect_probs = torch.sigmoid(outputs['aspect_logits']).squeeze()
        aspect_detected = (aspect_probs > 0.5).cpu().numpy()

        # 감성 분류
        sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=2).squeeze()
        sentiment_preds = torch.argmax(sentiment_probs, dim=1).cpu().numpy()

    print(f"\n📝 리뷰: {review_text}")
    print("-" * 40)

    detected = False
    for idx in range(12):
        if aspect_detected[idx]:
            aspect = ASPECT_NAMES[idx]
            sentiment_code = sentiment_preds[idx] + 1  # 0→1(긍정), 1→2(부정), 2→3(중립)
            sentiment = SENTIMENT_NAMES[sentiment_code]
            confidence = aspect_probs[idx].item()
            print(f"👉 [{aspect}] : {sentiment} (신뢰도: {confidence:.1%})")
            detected = True

    if not detected:
        print("👉 분석된 감성이 없습니다.")
    print("-" * 40)

# ==========================================
# 메인 실행 코드
# ==========================================
if __name__ == "__main__":
    # 1. 각각의 데이터 로드 (파일 경로를 실제 파일명에 맞게 수정하세요)
    print("📂 데이터 로딩 중...")
    try:
        df_single = pd.read_csv("single_aspect_new.csv", dtype={'label': str})
        df_multi = pd.read_csv("multi_aspect_new.csv", dtype={'label': str})
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        exit()

    print(f"단일 측면 데이터: {len(df_single)}개")
    print(f"복합 측면 데이터: {len(df_multi)}개")
    # GPU 메모리 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"   메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("⚠️ CPU 사용 중")

    # 2. 복합 데이터에서 Test 500개 분할
    multi_train_temp, multi_test = train_test_split(
        df_multi,
        test_size=500,
        random_state=SEED,
        shuffle=True
    )

    # 3. 남은 복합 데이터에서 Val 500개 분할
    multi_train_final, multi_val = train_test_split(
        multi_train_temp,
        test_size=500,
        random_state=SEED,
        shuffle=True
    )

    # 4. 최종 학습 데이터 생성: 단일 데이터 전량 + 남은 복합 데이터 전량
    df_train = pd.concat([df_single, multi_train_final], ignore_index=True)

    # 5. 리스트로 변환 (기존 코드 호환용)
    train_texts = df_train['Original_Review'].tolist()
    train_labels = df_train['label'].tolist()

    val_texts = multi_val['Original_Review'].tolist()
    val_labels = multi_val['label'].tolist()

    test_texts = multi_test['Original_Review'].tolist()
    test_labels = multi_test['label'].tolist()

    print("-" * 30)
    print(f"✅ 최종 학습 데이터: {len(train_texts)}개 (단일 전량 + 남은 복합)")
    print(f"✅ 최종 검증 데이터: {len(val_texts)}개 (복합에서만 추출)")
    print(f"✅ 최종 테스트 데이터: {len(test_texts)}개 (복합에서만 추출)")
    print("-" * 30)

    # 💾 분할된 데이터 저장
    print("\n💾 분할된 데이터셋 저장 중...")
    pd.DataFrame({'Original_Review': train_texts, 'label': train_labels}).to_csv('split_train.csv', index=False,
                                                                                 encoding='utf-8-sig')
    pd.DataFrame({'Original_Review': val_texts, 'label': val_labels}).to_csv('split_val.csv', index=False,
                                                                             encoding='utf-8-sig')
    pd.DataFrame({'Original_Review': test_texts, 'label': test_labels}).to_csv('split_test.csv', index=False,
                                                                               encoding='utf-8-sig')
    print("✅ 저장 완료! (split_train.csv, split_val.csv, split_test.csv)")

    from sklearn.utils.class_weight import compute_class_weight

    print("\n⚖️ 클래스 가중치 계산 중...")

    # 전체 학습 데이터의 라벨을 1차원으로 펼치기
    all_labels_flat = []
    for label_str in train_labels:
        label_str = str(label_str).zfill(12)
        all_labels_flat.extend([int(c) for c in label_str])

    # # 가중치 계산 (개수의 역수)
    # class_weights = compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.unique(all_labels_flat),
    #     y=all_labels_flat
    # )

    # print(f"📊 계산된 클래스 가중치:")
    # for i, w in enumerate(class_weights):
    #     print(f"   {SENTIMENT_NAMES[i]:10s}: {w:.2f}배")

    # # 텐서로 변환
    # weights_tensor = torch.tensor([
    #     1.0,  # 해당없음
    #     1.0,  # 긍정
    #     2.0,  # 부정
    #     2.0  # 중립
    # ], dtype=torch.float)

    sentiment_weights = torch.tensor([
        1.0,  # 긍정 (index 0)
        4.0,  # 부정 (index 1)
        8.0  # 중립 (index 2)
    ], dtype=torch.float)
    # ==========================================
    # 🆕 WeightedRandomSampler 준비
    # ==========================================
    from torch.utils.data import WeightedRandomSampler

    print("\n⚖️ WeightedRandomSampler 가중치 계산 중...")

    # 각 샘플의 가중치 계산
    sample_weights = []

    for label_str in train_labels:
        label_str = str(label_str).zfill(12)
        label_list = [int(c) for c in label_str]

        # 이 샘플의 가중치 계산
        has_negative = 2 in label_list
        has_neutral = 3 in label_list
        has_positive = 1 in label_list

        # 가중치 부여 전략
        if has_negative:
            weight = 8.0  # 부정 최우선
        elif has_neutral:
            weight = 15.0  # 중립 차순위
        elif has_positive:
            weight = 1.0  # 긍정은 기본
        else:
            weight = 0.3  # 전부 해당없음은 낮게

        sample_weights.append(weight)

    # Sampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 중복 허용
    )

    print(f"✅ Sampler 생성 완료")
    print(f"   부정 포함 샘플 가중치: 5.0배")
    print(f"   중립 포함 샘플 가중치: 3.0배")
    print(f"   긍정만 샘플 가중치: 1.0배")
    print(f"   전부 해당없음 가중치: 0.3배")


    # 3. 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)  # 🆕

    # 4. 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadBert(MODEL_NAME).to(device)

    # 5. 체크포인트에서 재개할지 확인
    resume_from_checkpoint = None
    # if os.path.exists(CHECKPOINT_PATH):
    #     checkpoints = [d for d in os.listdir(CHECKPOINT_PATH) if d.startswith('checkpoint-')]
    #     if checkpoints:
    #         latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    #         resume_from_checkpoint = os.path.join(CHECKPOINT_PATH, latest_checkpoint)
    #         print(f"🔄 체크포인트에서 재개: {resume_from_checkpoint}")

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
        #resume_from_checkpoint=resume_from_checkpoint,  # 🆕 재개 설정
        # 🆕 GPU 최적화 옵션
        fp16 = True,  # 혼합 정밀도 (속도 2배↑, 메모리 절약)
        dataloader_num_workers = 2,  # 데이터 로딩 병렬화
        dataloader_pin_memory = False,  # 경고 제거
        gradient_checkpointing = False,  # 메모리 부족 시 True로
    )

    # 7. 학습 시작
    print("🚀 학습 시작...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=sentiment_weights.to(device),
        train_sampler=sampler
    )

    trainer.train()

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

            # 🆕 Dual-Head 예측 처리
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            aspect_detected = (aspect_probs > 0.5).long()

            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=2)

            # 최종 예측 조합
            preds = torch.where(
                aspect_detected == 1,
                sentiment_preds + 1,  # 0→1, 1→2, 2→3
                torch.zeros_like(sentiment_preds)
            )

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
