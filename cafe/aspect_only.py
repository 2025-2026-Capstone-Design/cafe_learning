import pandas as pd
import torch
import torch.nn as nn
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from torch.utils.data import Dataset, WeightedRandomSampler

# ==========================================
# 0. 설정 및 하이퍼파라미터
# ==========================================
MODEL_NAME = "klue/bert-base"
DATA_PATH = "final_data.csv"
SAVE_PATH = "./aspect_only_model"
CHECKPOINT_PATH = "./checkpoints_aspect_only"
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

TRAIN_ASPECT_ONLY = True

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
# 2. Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1.5, weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce_loss(inputs, targets)
        pt = torch.exp(log_pt)
        loss = ((1 - pt) ** self.gamma) * self.ce_loss(inputs, targets)
        return loss.mean()


# ==========================================
# 3. 🎯 측면 분류 전용 모델 (역수 기반 pos_weight)
# ==========================================
class DualHeadBert(nn.Module):
    def __init__(self, model_name, aspect_only=True, global_pos_weights=None):
        super(DualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.4)  # 0.3 → 0.4 (과적합 방지)
        self.aspect_only = aspect_only

        # 측면 감지 헤드
        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)
        self.aspect_head.bias.data.fill_(-4.0)

        # 감성 분류 헤드 (나중을 위해 유지하지만 동결)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 12 * 3)

        # ✅ 전역 pos_weight 저장 (역수 기반 자동 계산된 값)
        if global_pos_weights is not None:
            self.register_buffer('global_pos_weights', global_pos_weights)
        else:
            # 기본값 (사용하지 않을 예정)
            self.register_buffer('global_pos_weights', torch.ones(12) * 10.0)

        # 측면 분류만 학습할 때는 감성 헤드 동결
        if aspect_only:
            for param in self.sentiment_head.parameters():
                param.requires_grad = False
            print("🔒 감성 헤드가 동결되었습니다. (측면 분류만 학습)")

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        aspect_logits = self.aspect_head(output)
        sentiment_logits = self.sentiment_head(output).view(-1, 12, 3)

        loss = None
        if labels is not None:
            aspect_labels = (labels != 0).float()

            # 🆕 사전 계산된 역수 기반 pos_weight 사용
            aspect_loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.global_pos_weights)
            aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)

            if self.aspect_only:
                # 측면 분류만 학습
                loss = aspect_loss
            else:
                # 나중에 감성 분류 추가 학습할 때를 위한 코드
                sentiment_loss_fct = FocalLoss(gamma=2.0, weight=class_weights)
                sentiment_loss = 0
                valid_aspect_count = 0

                for i in range(12):
                    mask = aspect_labels[:, i] == 1
                    if mask.sum() > 0:
                        loss_s = sentiment_loss_fct(
                            sentiment_logits[mask, i, :],
                            torch.where(labels[mask, i] > 0, labels[mask, i] - 1,
                                        torch.zeros_like(labels[mask, i]))
                        )
                        sentiment_loss += loss_s
                        valid_aspect_count += 1

                if valid_aspect_count > 0:
                    sentiment_loss = sentiment_loss / valid_aspect_count

                loss = aspect_loss + 1.0 * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }


# ==========================================
# 4. CustomTrainer
# ==========================================
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, train_sampler=None, aspect_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_sampler = train_sampler
        self.aspect_only = aspect_only
        self.custom_step = 0

    def get_train_dataloader(self):
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

        if not hasattr(self, 'custom_step'):
            self.custom_step = 0

        if model.training and self.custom_step % 50 == 0:
            self._check_batch_distribution(labels)

        self.custom_step += 1

        outputs = model(**inputs, labels=labels, class_weights=self.class_weights)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def _check_batch_distribution(self, labels):
        """배치 내의 측면별 등장 횟수 출력"""
        batch_size = labels.size(0)

        print(f"\n📊 [Step {self.custom_step}] Batch Distribution Analysis")

        if self.aspect_only:
            print(f"   🎯 [모드: 측면 분류 전용]")

        # 측면별 등장 빈도
        print(f"\n   [측면별 등장 빈도 (총 {batch_size}개 샘플 중)]")
        aspect_counts = (labels != 0).sum(dim=0).cpu().numpy()

        for i in range(0, 12, 2):
            left_aspect = f"{ASPECT_NAMES[i]:12s}: {aspect_counts[i]:2d}개"
            right_aspect = f"{ASPECT_NAMES[i + 1]:12s}: {aspect_counts[i + 1]:2d}개" if i + 1 < 12 else ""
            print(f"   - {left_aspect}  |  {right_aspect}")

        # 전체 측면 비율
        total_elements = labels.numel()
        aspect_present = (labels != 0).sum().item()
        print(f"\n   [전체 측면 비율]")
        print(f"   - 측면 있음: {aspect_present:4d}개 ({aspect_present / total_elements:5.1%})")
        print(
            f"   - 측면 없음: {total_elements - aspect_present:4d}개 ({(total_elements - aspect_present) / total_elements:5.1%})")
        print("-" * 60)


# ==========================================
# 5. 예측 함수 (차등 임계값 적용)
# ==========================================
# 🆕 측면별 최적 임계값 (FP 방지)
EVAL_THRESHOLDS = {
    0: 0.50,  # 커피/음료
    1: 0.52,  # 베이커리/빵
    2: 0.68,  # 케이크 (FP 방지)
    3: 0.72,  # 쿠키 (가장 높게)
    4: 0.50,  # 빙수/과일
    5: 0.65,  # 기타디저트 (FP 방지)
    6: 0.58,  # 공간/편의시설
    7: 0.58,  # 분위기/감성
    8: 0.52,  # 서비스
    9: 0.55,  # 가격/가성비
    10: 0.68,  # 선물/포장 (FP 방지)
    11: 0.60,  # 혼잡도/웨이팅
}


def predict_review_aspect_only(model, tokenizer, review_text, device, use_optimal_thresholds=True):
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
        aspect_probs = torch.sigmoid(outputs['aspect_logits']).squeeze().cpu().numpy()

    print(f"\n📝 리뷰: {review_text}")
    print("-" * 60)

    detected = False
    for idx in range(12):
        threshold = EVAL_THRESHOLDS[idx] if use_optimal_thresholds else 0.5
        if aspect_probs[idx] > threshold:
            aspect = ASPECT_NAMES[idx]
            conf = aspect_probs[idx]
            print(f"✅ [{aspect:12s}] 감지됨 (신뢰도: {conf:.1%}, 임계값: {threshold:.2f})")
            detected = True

    if not detected:
        print("❌ 감지된 측면이 없습니다.")
    print("-" * 60)

#
# # ==========================================
# # 메인 실행 코드
# # ==========================================
# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("🎯 측면 분류 전용 모델 학습 시작 (역수 기반 pos_weight)")
#     print("=" * 60 + "\n")
#
#     # GPU 확인
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
#         print(f"   메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
#     else:
#         print("⚠️ CPU 사용 중")
#
#     # 1. 데이터 로드
#     print("\n📂 데이터 로딩 중...")
#     try:
#         df_single = pd.read_csv("single_aspect_new.csv", dtype={'label': str})
#         df_multi = pd.read_csv("multi_aspect_new_relabeled.csv", dtype={'label': str})
#     except FileNotFoundError as e:
#         print(f"❌ 파일을 찾을 수 없습니다: {e}")
#         exit()
#
#     print(f"단일 측면 데이터: {len(df_single)}개")
#     print(f"복합 측면 데이터: {len(df_multi)}개")
#
#     # 2. 데이터 분할
#     multi_train_temp, multi_test = train_test_split(
#         df_multi, test_size=500, random_state=SEED, shuffle=True
#     )
#     multi_train_final, multi_val = train_test_split(
#         multi_train_temp, test_size=500, random_state=SEED, shuffle=True
#     )
#     df_train = pd.concat([df_single, multi_train_final], ignore_index=True)
#
#     train_texts = df_train['Original_Review'].tolist()
#     train_labels = df_train['label'].tolist()
#     val_texts = multi_val['Original_Review'].tolist()
#     val_labels = multi_val['label'].tolist()
#     test_texts = multi_test['Original_Review'].tolist()
#     test_labels = multi_test['label'].tolist()
#
#     print("-" * 30)
#     print(f"✅ 학습 데이터: {len(train_texts)}개")
#     print(f"✅ 검증 데이터: {len(val_texts)}개")
#     print(f"✅ 테스트 데이터: {len(test_texts)}개")
#     print("-" * 30)
#
#     # 3. 분할 데이터 저장
#     print("\n💾 분할된 데이터셋 저장 중...")
#     pd.DataFrame({'Original_Review': train_texts, 'label': train_labels}).to_csv(
#         'split_train.csv', index=False, encoding='utf-8-sig')
#     pd.DataFrame({'Original_Review': val_texts, 'label': val_labels}).to_csv(
#         'split_val.csv', index=False, encoding='utf-8-sig')
#     pd.DataFrame({'Original_Review': test_texts, 'label': test_labels}).to_csv(
#         'split_test.csv', index=False, encoding='utf-8-sig')
#     print("✅ 저장 완료!")
#
#     # ========================================
#     # 4. 🆕 pos_weight 자동 계산 (역수 기반)
#     # ========================================
#     print("\n📊 pos_weight 자동 계산을 위한 전역 통계 생성 중...")
#
#     # 전체 학습 데이터의 측면별 긍정 비율 계산
#     train_aspect_presence = np.zeros(12)
#     for label_str in train_labels:
#         label_str = str(label_str).zfill(12)
#         for i, c in enumerate(label_str):
#             if int(c) > 0:
#                 train_aspect_presence[i] += 1
#
#     train_aspect_ratios = train_aspect_presence / len(train_labels)
#
#     # 역수 기반 pos_weight 계산 (FP 방지 보정 포함)
#     global_pos_weights = np.zeros(12)
#     FP_PRONE_ASPECTS_DICT = {2, 3, 5, 10}  # 케이크, 쿠키, 기타디저트, 선물/포장
#     MODERATE_FP_ASPECTS = {6, 7, 11}  # 공간, 분위기, 혼잡도
#
#     for i in range(12):
#         if train_aspect_ratios[i] > 0:
#             # 기본 역수 계산: (전체 - 긍정) / 긍정
#             base_weight = (1 - train_aspect_ratios[i]) / train_aspect_ratios[i]
#
#             # FP 방지 보정
#             if i in FP_PRONE_ASPECTS_DICT:
#                 base_weight *= 0.6  # 40% 감소
#             elif i in MODERATE_FP_ASPECTS:
#                 base_weight *= 0.8  # 20% 감소
#
#             # 범위 제한 (5.0 ~ 25.0)
#             global_pos_weights[i] = max(5.0, min(base_weight, 25.0))
#         else:
#             global_pos_weights[i] = 10.0
#
#     print("\n📈 계산된 pos_weight (역수 기반 + FP 보정):")
#     print("-" * 60)
#     for i, name in enumerate(ASPECT_NAMES):
#         fp_tag = ""
#         if i in FP_PRONE_ASPECTS_DICT:
#             fp_tag = " [FP 취약 -40%]"
#         elif i in MODERATE_FP_ASPECTS:
#             fp_tag = " [FP 주의 -20%]"
#         print(f"   {name:15s}: {global_pos_weights[i]:6.2f} (비율: {train_aspect_ratios[i]:5.1%}){fp_tag}")
#     print("-" * 60)
#
#     # 모델에 전달할 텐서로 변환
#     global_pos_weights_tensor = torch.tensor(global_pos_weights, dtype=torch.float32)
#
#     # ========================================
#     # 5. 🆕 역수 기반 Sampler 생성 (네거티브 보너스 강화)
#     # ========================================
#     print("\n⚖️ 희소성 역수 기반 Sampler 생성 중...")
#
#     # 측면별 등장 횟수 (이미 위에서 계산됨)
#     aspect_counts = train_aspect_presence
#
#     print("\n📊 측면별 등장 횟수:")
#     for i, name in enumerate(ASPECT_NAMES):
#         print(f"   {name:15s}: {int(aspect_counts[i]):5d}회")
#
#     # FP 취약 측면 정의
#     FP_PRONE_ASPECTS = {
#         5: "기타디저트",
#         6: "공간/편의시설",
#         7: "분위기/감성",
#         11: "혼잡도/웨이팅"
#     }
#
#     # 역수 기반 가중치 계산
#     sample_weights = []
#     alpha = 1.0
#
#     for label_str in train_labels:
#         label_str = str(label_str).zfill(12)
#         label_list = [int(c) for c in label_str]
#
#         current_weight = 0
#
#         # A. 희소 측면 가중치 (역수)
#         for i in range(12):
#             if label_list[i] > 0:
#                 rarity_weight = 1.0 / (aspect_counts[i] + 1e-5)
#                 current_weight += rarity_weight
#
#         # B. FP 취약 측면의 네거티브 샘플 강화 (5.0 → 8.0)
#         fp_negative_bonus = 0
#         for i in FP_PRONE_ASPECTS.keys():
#             if label_list[i] == 0:
#                 fp_negative_bonus += aspect_counts[i] / (aspect_counts.sum() + 1e-5)
#
#         current_weight += fp_negative_bonus * 8.0  # 네거티브 보너스 강화
#
#         # C. 기본 가중치
#         if current_weight == 0:
#             current_weight = 1.0
#
#         sample_weights.append(current_weight * alpha)
#
#     # 가중치 통계
#     sample_weights_array = np.array(sample_weights)
#     print(f"\n📊 샘플 가중치 통계:")
#     print(f"   최소: {sample_weights_array.min():.4f}")
#     print(f"   최대: {sample_weights_array.max():.4f}")
#     print(f"   평균: {sample_weights_array.mean():.4f}")
#     print(f"   중앙: {np.median(sample_weights_array):.4f}")
#
#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )
#     print("✅ 역수 기반 Sampler 생성 완료 (네거티브 보너스 8.0 적용)")
#
#     # 6. 토크나이저 및 데이터셋 준비
#     tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
#     train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
#     val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)
#     test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)
#
#     # 7. 모델 초기화 (역수 기반 pos_weight 주입)
#     model = DualHeadBert(
#         MODEL_NAME,
#         aspect_only=True,
#         global_pos_weights=global_pos_weights_tensor
#     ).to(device)
#
#     # 8. 학습 설정
#     training_args = TrainingArguments(
#         output_dir=CHECKPOINT_PATH,
#         num_train_epochs=EPOCHS,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         warmup_steps=100,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=50,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         report_to="none",
#         save_total_limit=2,
#         fp16=True,
#         dataloader_num_workers=2,
#         dataloader_pin_memory=False,
#     )
#
#     # 9. 학습 시작
#     print("\n🚀 학습 시작...")
#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         train_sampler=sampler,
#         aspect_only=True
#     )
#
#     trainer.train()
#
#     # 10. 모델 저장
#     print(f"\n💾 최종 모델 저장 중... ({SAVE_PATH})")
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     torch.save(model.state_dict(), f"{SAVE_PATH}/model_state_dict.pt")
#     tokenizer.save_pretrained(SAVE_PATH)
#
#     # pos_weight도 함께 저장
#     torch.save(global_pos_weights_tensor, f"{SAVE_PATH}/global_pos_weights.pt")
#     print("✅ 모델 및 pos_weight 저장 완료!")
#
#     # 11. 테스트 평가 (차등 임계값 적용)
#     print("\n" + "=" * 60)
#     print("📊 테스트 데이터셋 평가 (차등 임계값 적용)")
#     print("=" * 60)
#
#     model.eval()
#     all_aspect_preds = []
#     all_aspect_labels = []
#     all_aspect_probs = []
#
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
#
#     # 임계값 텐서 생성
#     thresholds_tensor = torch.tensor([EVAL_THRESHOLDS[i] for i in range(12)]).to(device)
#
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#
#             outputs = model(input_ids, attention_mask)
#             aspect_probs = torch.sigmoid(outputs['aspect_logits'])
#
#             # 측면별 차등 임계값 적용
#             aspect_preds = torch.zeros_like(aspect_probs, dtype=torch.long)
#             for i in range(12):
#                 aspect_preds[:, i] = (aspect_probs[:, i] > thresholds_tensor[i]).long()
#
#             aspect_labels = (labels != 0).long()
#
#             all_aspect_preds.append(aspect_preds.cpu())
#             all_aspect_labels.append(aspect_labels.cpu())
#             all_aspect_probs.append(aspect_probs.cpu())
#
#     all_aspect_preds = torch.cat(all_aspect_preds, dim=0)
#     all_aspect_labels = torch.cat(all_aspect_labels, dim=0)
#     all_aspect_probs = torch.cat(all_aspect_probs, dim=0)
#
#     # 측면별 상세 평가
#     print("\n📈 측면별 평가 지표 (차등 임계값):")
#     print("-" * 80)
#     print(
#         f"{'측면':<15} {'F1':>6} {'Prec':>6} {'Recall':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'임계값':>6}")
#     print("-" * 80)
#
#     for i, aspect_name in enumerate(ASPECT_NAMES):
#         tp = ((all_aspect_preds[:, i] == 1) & (all_aspect_labels[:, i] == 1)).sum().item()
#         fp = ((all_aspect_preds[:, i] == 1) & (all_aspect_labels[:, i] == 0)).sum().item()
#         tn = ((all_aspect_preds[:, i] == 0) & (all_aspect_labels[:, i] == 0)).sum().item()
#         fn = ((all_aspect_preds[:, i] == 0) & (all_aspect_labels[:, i] == 1)).sum().item()
#
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#         accuracy = (tp + tn) / (tp + fp + tn + fn)
#
#         threshold = EVAL_THRESHOLDS[i]
#
#         print(f"{aspect_name:<15} {f1:>6.3f} {precision:>6.3f} {recall:>6.3f} {accuracy:>6.3f} "
#               f"{tp:>4} {fp:>4} {tn:>4} {fn:>4} {threshold:>6.2f}")
#
#     total_correct = (all_aspect_preds == all_aspect_labels).sum().item()
#     total_elements = all_aspect_labels.numel()
#     overall_accuracy = total_correct / total_elements * 100
#     print("-" * 80)
#     print(f"전체 정확도: {overall_accuracy:.2f}%")
#
#     # 12. 샘플 예측 (차등 임계값 적용)
#     print("\n🔍 학습된 모델로 샘플 테스트 (차등 임계값)")
#
#     test_reviews = [
#         "커피는 진짜 맛있는데 직원이 좀 불친절해서 기분 나빴음",
#         "케이크랑 커피 둘 다 너무 맛있고 분위기도 짱 좋아요!",
#         "가격은 비싼데 맛은 그냥 편의점 수준이네요 실망입니다.",
#         "웨이팅이 너무 길어서 힘들었지만 소금빵 먹자마자 용서됨"
#     ]
#
#     for review in test_reviews:
#         predict_review_aspect_only(model, tokenizer, review, device)
#
#     print("\n" + "=" * 60)
#     print("✅ 측면 분류 모델 학습 완료!")
#     print("=" * 60)


# ==========================================
# 메인 실행 코드
# ==========================================
if __name__ == "__main__":

    # ============================================================
    # STAGE 1: 단일 측면 데이터만으로 학습
    # ============================================================
    print("\n" + "=" * 80)
    print("🎯 STAGE 1: 단일 측면 데이터 학습 시작")
    print("=" * 80 + "\n")

    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"   메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("⚠️ CPU 사용 중")

    # 1-1. 단일 측면 데이터 로드
    print("\n📂 [STAGE 1] 단일 측면 데이터 로딩 중...")
    try:
        df_single = pd.read_csv("single_aspect_new.csv", dtype={'label': str})
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        exit()

    print(f"단일 측면 데이터: {len(df_single)}개")

    # 1-2. 단일 측면 데이터 분할
    single_train_temp, single_test = train_test_split(
        df_single, test_size=500, random_state=SEED, shuffle=True
    )
    single_train, single_val = train_test_split(
        single_train_temp, test_size=500, random_state=SEED, shuffle=True
    )

    train_texts_single = single_train['Original_Review'].tolist()
    train_labels_single = single_train['label'].tolist()
    val_texts_single = single_val['Original_Review'].tolist()
    val_labels_single = single_val['label'].tolist()
    test_texts_single = single_test['Original_Review'].tolist()
    test_labels_single = single_test['label'].tolist()

    print("-" * 30)
    print(f"✅ 학습 데이터: {len(train_texts_single)}개")
    print(f"✅ 검증 데이터: {len(val_texts_single)}개")
    print(f"✅ 테스트 데이터: {len(test_texts_single)}개")
    print("-" * 30)

    # 1-3. 단일 측면 분할 데이터 저장
    print("\n💾 [STAGE 1] 분할된 데이터셋 저장 중...")
    pd.DataFrame({'Original_Review': train_texts_single, 'label': train_labels_single}).to_csv(
        'split_train_single.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'Original_Review': val_texts_single, 'label': val_labels_single}).to_csv(
        'split_val_single.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'Original_Review': test_texts_single, 'label': test_labels_single}).to_csv(
        'split_test_single.csv', index=False, encoding='utf-8-sig')
    print("✅ 저장 완료!")

    # 1-4. pos_weight 계산 (단일 측면)
    print("\n📊 [STAGE 1] pos_weight 자동 계산 중...")
    train_aspect_presence_single = np.zeros(12)
    for label_str in train_labels_single:
        label_str = str(label_str).zfill(12)
        for i, c in enumerate(label_str):
            if int(c) > 0:
                train_aspect_presence_single[i] += 1

    train_aspect_ratios_single = train_aspect_presence_single / len(train_labels_single)

    global_pos_weights_single = np.zeros(12)
    FP_PRONE_ASPECTS_DICT = {2, 3, 5, 10}
    MODERATE_FP_ASPECTS = {6, 7, 11}

    for i in range(12):
        if train_aspect_ratios_single[i] > 0:
            base_weight = (1 - train_aspect_ratios_single[i]) / train_aspect_ratios_single[i]
            if i in FP_PRONE_ASPECTS_DICT:
                base_weight *= 0.6
            elif i in MODERATE_FP_ASPECTS:
                base_weight *= 0.8
            global_pos_weights_single[i] = max(5.0, min(base_weight, 25.0))
        else:
            global_pos_weights_single[i] = 10.0

    print("\n📈 계산된 pos_weight (단일 측면):")
    print("-" * 60)
    for i, name in enumerate(ASPECT_NAMES):
        fp_tag = ""
        if i in FP_PRONE_ASPECTS_DICT:
            fp_tag = " [FP 취약 -40%]"
        elif i in MODERATE_FP_ASPECTS:
            fp_tag = " [FP 주의 -20%]"
        print(f"   {name:15s}: {global_pos_weights_single[i]:6.2f} (비율: {train_aspect_ratios_single[i]:5.1%}){fp_tag}")
    print("-" * 60)

    global_pos_weights_tensor_single = torch.tensor(global_pos_weights_single, dtype=torch.float32)

    # 1-5. Sampler 생성 (단일 측면)
    print("\n⚖️ [STAGE 1] 희소성 역수 기반 Sampler 생성 중...")
    aspect_counts_single = train_aspect_presence_single

    FP_PRONE_ASPECTS = {
        5: "기타디저트",
        6: "공간/편의시설",
        7: "분위기/감성",
        11: "혼잡도/웨이팅"
    }

    sample_weights_single = []
    alpha = 1.0

    for label_str in train_labels_single:
        label_str = str(label_str).zfill(12)
        label_list = [int(c) for c in label_str]
        current_weight = 0

        for i in range(12):
            if label_list[i] > 0:
                rarity_weight = 1.0 / (aspect_counts_single[i] + 1e-5)
                current_weight += rarity_weight

        fp_negative_bonus = 0
        for i in FP_PRONE_ASPECTS.keys():
            if label_list[i] == 0:
                fp_negative_bonus += aspect_counts_single[i] / (aspect_counts_single.sum() + 1e-5)

        current_weight += fp_negative_bonus * 8.0

        if current_weight == 0:
            current_weight = 1.0

        sample_weights_single.append(current_weight * alpha)

    sampler_single = WeightedRandomSampler(
        weights=sample_weights_single,
        num_samples=len(sample_weights_single),
        replacement=True
    )
    print("✅ 역수 기반 Sampler 생성 완료")

    # 1-6. 토크나이저 및 데이터셋 준비 (단일 측면)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset_single = CafeAspectDataset(train_texts_single, train_labels_single, tokenizer, MAX_LEN)
    val_dataset_single = CafeAspectDataset(val_texts_single, val_labels_single, tokenizer, MAX_LEN)
    test_dataset_single = CafeAspectDataset(test_texts_single, test_labels_single, tokenizer, MAX_LEN)

    # 1-7. 모델 초기화 (단일 측면)
    model_single = DualHeadBert(
        MODEL_NAME,
        aspect_only=True,
        global_pos_weights=global_pos_weights_tensor_single
    ).to(device)

    # 1-8. 학습 설정 (단일 측면)
    training_args_single = TrainingArguments(
        output_dir="./checkpoints_single_aspect",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_single',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
    )

    # 1-9. 학습 시작 (단일 측면)
    print("\n🚀 [STAGE 1] 학습 시작...")
    trainer_single = CustomTrainer(
        model=model_single,
        args=training_args_single,
        train_dataset=train_dataset_single,
        eval_dataset=val_dataset_single,
        train_sampler=sampler_single,
        aspect_only=True
    )

    trainer_single.train()

    # 1-10. 모델 저장 (단일 측면)
    SAVE_PATH_SINGLE = "./aspect_only_model_single"
    print(f"\n💾 [STAGE 1] 최종 모델 저장 중... ({SAVE_PATH_SINGLE})")
    os.makedirs(SAVE_PATH_SINGLE, exist_ok=True)
    torch.save(model_single.state_dict(), f"{SAVE_PATH_SINGLE}/model_state_dict.pt")
    tokenizer.save_pretrained(SAVE_PATH_SINGLE)
    torch.save(global_pos_weights_tensor_single, f"{SAVE_PATH_SINGLE}/global_pos_weights.pt")
    print("✅ 단일 측면 모델 저장 완료!")

    # 1-11. 테스트 평가 (단일 측면)
    print("\n" + "=" * 60)
    print("📊 [STAGE 1] 테스트 데이터셋 평가")
    print("=" * 60)

    model_single.eval()
    all_aspect_preds_single = []
    all_aspect_labels_single = []

    test_loader_single = torch.utils.data.DataLoader(test_dataset_single, batch_size=BATCH_SIZE)
    thresholds_tensor = torch.tensor([EVAL_THRESHOLDS[i] for i in range(12)]).to(device)

    with torch.no_grad():
        for batch in test_loader_single:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model_single(input_ids, attention_mask)
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])

            aspect_preds = torch.zeros_like(aspect_probs, dtype=torch.long)
            for i in range(12):
                aspect_preds[:, i] = (aspect_probs[:, i] > thresholds_tensor[i]).long()

            aspect_labels = (labels != 0).long()

            all_aspect_preds_single.append(aspect_preds.cpu())
            all_aspect_labels_single.append(aspect_labels.cpu())

    all_aspect_preds_single = torch.cat(all_aspect_preds_single, dim=0)
    all_aspect_labels_single = torch.cat(all_aspect_labels_single, dim=0)

    print("\n📈 [STAGE 1] 측면별 평가 지표:")
    print("-" * 80)
    print(f"{'측면':<15} {'F1':>6} {'Prec':>6} {'Recall':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print("-" * 80)

    for i, aspect_name in enumerate(ASPECT_NAMES):
        tp = ((all_aspect_preds_single[:, i] == 1) & (all_aspect_labels_single[:, i] == 1)).sum().item()
        fp = ((all_aspect_preds_single[:, i] == 1) & (all_aspect_labels_single[:, i] == 0)).sum().item()
        tn = ((all_aspect_preds_single[:, i] == 0) & (all_aspect_labels_single[:, i] == 0)).sum().item()
        fn = ((all_aspect_preds_single[:, i] == 0) & (all_aspect_labels_single[:, i] == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        print(f"{aspect_name:<15} {f1:>6.3f} {precision:>6.3f} {recall:>6.3f} {accuracy:>6.3f} "
              f"{tp:>4} {fp:>4} {tn:>4} {fn:>4}")

    print("-" * 80)
    print("✅ STAGE 1 완료!\n")

    # ============================================================
    # STAGE 2: 복합 측면 데이터 추가 학습
    # ============================================================
    print("\n" + "=" * 80)
    print("🎯 STAGE 2: 복합 측면 데이터 추가 학습 시작")
    print("=" * 80 + "\n")

    # 2-1. 복합 측면 데이터 로드
    print("\n📂 [STAGE 2] 복합 측면 데이터 로딩 중...")
    try:
        df_multi = pd.read_csv("multi_aspect_new_relabeled.csv", dtype={'label': str})
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        exit()

    print(f"복합 측면 데이터: {len(df_multi)}개")

    # 2-2. 복합 측면 데이터 분할 및 단일 측면과 결합
    multi_train_temp, multi_test = train_test_split(
        df_multi, test_size=500, random_state=SEED, shuffle=True
    )
    multi_train_final, multi_val = train_test_split(
        multi_train_temp, test_size=500, random_state=SEED, shuffle=True
    )

    # 단일 측면 + 복합 측면 결합
    df_train_combined = pd.concat([single_train, multi_train_final], ignore_index=True)

    train_texts_combined = df_train_combined['Original_Review'].tolist()
    train_labels_combined = df_train_combined['label'].tolist()
    val_texts_combined = multi_val['Original_Review'].tolist()
    val_labels_combined = multi_val['label'].tolist()
    test_texts_combined = multi_test['Original_Review'].tolist()
    test_labels_combined = multi_test['label'].tolist()

    print("-" * 30)
    print(f"✅ 학습 데이터 (단일+복합): {len(train_texts_combined)}개")
    print(f"   - 단일 측면: {len(single_train)}개")
    print(f"   - 복합 측면: {len(multi_train_final)}개")
    print(f"✅ 검증 데이터: {len(val_texts_combined)}개")
    print(f"✅ 테스트 데이터: {len(test_texts_combined)}개")
    print("-" * 30)

    # 2-3. 결합 데이터 저장
    print("\n💾 [STAGE 2] 분할된 데이터셋 저장 중...")
    pd.DataFrame({'Original_Review': train_texts_combined, 'label': train_labels_combined}).to_csv(
        'split_train_combined.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'Original_Review': val_texts_combined, 'label': val_labels_combined}).to_csv(
        'split_val_combined.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'Original_Review': test_texts_combined, 'label': test_labels_combined}).to_csv(
        'split_test_combined.csv', index=False, encoding='utf-8-sig')
    print("✅ 저장 완료!")

    # 2-4. pos_weight 재계산 (결합 데이터)
    print("\n📊 [STAGE 2] pos_weight 재계산 중 (단일+복합)...")
    train_aspect_presence_combined = np.zeros(12)
    for label_str in train_labels_combined:
        label_str = str(label_str).zfill(12)
        for i, c in enumerate(label_str):
            if int(c) > 0:
                train_aspect_presence_combined[i] += 1

    train_aspect_ratios_combined = train_aspect_presence_combined / len(train_labels_combined)

    global_pos_weights_combined = np.zeros(12)
    for i in range(12):
        if train_aspect_ratios_combined[i] > 0:
            base_weight = (1 - train_aspect_ratios_combined[i]) / train_aspect_ratios_combined[i]
            if i in FP_PRONE_ASPECTS_DICT:
                base_weight *= 0.6
            elif i in MODERATE_FP_ASPECTS:
                base_weight *= 0.8
            global_pos_weights_combined[i] = max(5.0, min(base_weight, 25.0))
        else:
            global_pos_weights_combined[i] = 10.0

    print("\n📈 계산된 pos_weight (단일+복합):")
    print("-" * 60)
    for i, name in enumerate(ASPECT_NAMES):
        fp_tag = ""
        if i in FP_PRONE_ASPECTS_DICT:
            fp_tag = " [FP 취약 -40%]"
        elif i in MODERATE_FP_ASPECTS:
            fp_tag = " [FP 주의 -20%]"
        print(
            f"   {name:15s}: {global_pos_weights_combined[i]:6.2f} (비율: {train_aspect_ratios_combined[i]:5.1%}){fp_tag}")
    print("-" * 60)

    global_pos_weights_tensor_combined = torch.tensor(global_pos_weights_combined, dtype=torch.float32)

    # 2-5. Sampler 생성 (결합 데이터)
    print("\n⚖️ [STAGE 2] Sampler 생성 중...")
    aspect_counts_combined = train_aspect_presence_combined

    sample_weights_combined = []
    for label_str in train_labels_combined:
        label_str = str(label_str).zfill(12)
        label_list = [int(c) for c in label_str]
        current_weight = 0

        for i in range(12):
            if label_list[i] > 0:
                rarity_weight = 1.0 / (aspect_counts_combined[i] + 1e-5)
                current_weight += rarity_weight

        fp_negative_bonus = 0
        for i in FP_PRONE_ASPECTS.keys():
            if label_list[i] == 0:
                fp_negative_bonus += aspect_counts_combined[i] / (aspect_counts_combined.sum() + 1e-5)

        current_weight += fp_negative_bonus * 8.0

        if current_weight == 0:
            current_weight = 1.0

        sample_weights_combined.append(current_weight * alpha)

    sampler_combined = WeightedRandomSampler(
        weights=sample_weights_combined,
        num_samples=len(sample_weights_combined),
        replacement=True
    )
    print("✅ Sampler 생성 완료")

    # 2-6. 데이터셋 준비 (결합 데이터)
    train_dataset_combined = CafeAspectDataset(train_texts_combined, train_labels_combined, tokenizer, MAX_LEN)
    val_dataset_combined = CafeAspectDataset(val_texts_combined, val_labels_combined, tokenizer, MAX_LEN)
    test_dataset_combined = CafeAspectDataset(test_texts_combined, test_labels_combined, tokenizer, MAX_LEN)

    # 2-7. 모델 초기화 (단일 측면 모델의 가중치로 시작)
    print("\n🔧 [STAGE 2] 모델 초기화 중 (STAGE 1 가중치 로드)...")
    model_combined = DualHeadBert(
        MODEL_NAME,
        aspect_only=True,
        global_pos_weights=global_pos_weights_tensor_combined
    ).to(device)

    # STAGE 1에서 학습한 가중치 로드
    model_combined.load_state_dict(model_single.state_dict())
    print("✅ STAGE 1 모델 가중치 로드 완료!")

    # 2-8. 학습 설정 (복합 측면)
    training_args_combined = TrainingArguments(
        output_dir=CHECKPOINT_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_combined',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
    )

    # 2-9. 학습 시작 (복합 측면)
    print("\n🚀 [STAGE 2] 학습 시작...")
    trainer_combined = CustomTrainer(
        model=model_combined,
        args=training_args_combined,
        train_dataset=train_dataset_combined,
        eval_dataset=val_dataset_combined,
        train_sampler=sampler_combined,
        aspect_only=True
    )

    trainer_combined.train()

    # 2-10. 최종 모델 저장
    print(f"\n💾 [STAGE 2] 최종 모델 저장 중... ({SAVE_PATH})")
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(model_combined.state_dict(), f"{SAVE_PATH}/model_state_dict.pt")
    tokenizer.save_pretrained(SAVE_PATH)
    torch.save(global_pos_weights_tensor_combined, f"{SAVE_PATH}/global_pos_weights.pt")
    print("✅ 최종 모델 저장 완료!")

    # 2-11. 테스트 평가 (복합 측면)
    print("\n" + "=" * 60)
    print("📊 [STAGE 2] 테스트 데이터셋 평가 (차등 임계값 적용)")
    print("=" * 60)

    model_combined.eval()
    all_aspect_preds = []
    all_aspect_labels = []
    all_aspect_probs = []

    test_loader = torch.utils.data.DataLoader(test_dataset_combined, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model_combined(input_ids, attention_mask)
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])

            aspect_preds = torch.zeros_like(aspect_probs, dtype=torch.long)
            for i in range(12):
                aspect_preds[:, i] = (aspect_probs[:, i] > thresholds_tensor[i]).long()

            aspect_labels = (labels != 0).long()

            all_aspect_preds.append(aspect_preds.cpu())
            all_aspect_labels.append(aspect_labels.cpu())
            all_aspect_probs.append(aspect_probs.cpu())

    all_aspect_preds = torch.cat(all_aspect_preds, dim=0)
    all_aspect_labels = torch.cat(all_aspect_labels, dim=0)
    all_aspect_probs = torch.cat(all_aspect_probs, dim=0)

    print("\n📈 측면별 평가 지표 (차등 임계값):")
    print("-" * 80)
    print(
        f"{'측면':<15} {'F1':>6} {'Prec':>6} {'Recall':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'임계값':>6}")
    print("-" * 80)

    for i, aspect_name in enumerate(ASPECT_NAMES):
        tp = ((all_aspect_preds[:, i] == 1) & (all_aspect_labels[:, i] == 1)).sum().item()
        fp = ((all_aspect_preds[:, i] == 1) & (all_aspect_labels[:, i] == 0)).sum().item()
        tn = ((all_aspect_preds[:, i] == 0) & (all_aspect_labels[:, i] == 0)).sum().item()
        fn = ((all_aspect_preds[:, i] == 0) & (all_aspect_labels[:, i] == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        threshold = EVAL_THRESHOLDS[i]

        print(f"{aspect_name:<15} {f1:>6.3f} {precision:>6.3f} {recall:>6.3f} {accuracy:>6.3f} "
              f"{tp:>4} {fp:>4} {tn:>4} {fn:>4} {threshold:>6.2f}")

    total_correct = (all_aspect_preds == all_aspect_labels).sum().item()
    total_elements = all_aspect_labels.numel()
    overall_accuracy = total_correct / total_elements * 100
    print("-" * 80)
    print(f"전체 정확도: {overall_accuracy:.2f}%")

    # 2-12. 샘플 예측
    print("\n🔍 학습된 모델로 샘플 테스트 (차등 임계값)")

    test_reviews = [
        "커피는 진짜 맛있는데 직원이 좀 불친절해서 기분 나빴음",
        "케이크랑 커피 둘 다 너무 맛있고 분위기도 짱 좋아요!",
        "가격은 비싼데 맛은 그냥 편의점 수준이네요 실망입니다.",
        "웨이팅이 너무 길어서 힘들었지만 소금빵 먹자마자 용서됨"
    ]

    for review in test_reviews:
        predict_review_aspect_only(model_combined, tokenizer, review, device)

    print("\n" + "=" * 80)
    print("✅ 전체 학습 완료!")
    print("=" * 80)
    print("\n📁 생성된 모델:")
    print(f"   - STAGE 1 (단일 측면): {SAVE_PATH_SINGLE}/")
    print(f"   - STAGE 2 (최종 모델): {SAVE_PATH}/")