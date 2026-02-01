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
SAVE_PATH_SINGLE="./aspect_only_model_single"
SAVE_PATH_SENTIMENT_SINGLE = "./aspect_sentiment_model"
SAVE_PATH_SENTIMENT_FINAL="./aspect_sentiment_final_model"
CHECKPOINT_PATH = "./checkpoints_aspect_only"
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42
# 🆕 실행할 STAGE 선택
RUN_STAGE_1 = False  # False로 설정하면 건너뜀
RUN_STAGE_2 = False # False로 설정하면 건너뜀
RUN_STAGE_3 = True  # True로 설정하면 실행
RUN_STAGE_4= True

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
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, weight=None):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha  # 클래스별 가중치
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

# ==========================================
# 3. 🎯 측면 분류 전용 모델 (역수 기반 pos_weight)
# ==========================================
class ImprovedDualHeadBert(nn.Module):
    def __init__(self, model_name, aspect_only=True, global_pos_weights=None, sentiment_class_weights=None):
        super(ImprovedDualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.aspect_only = aspect_only

        # 🔥 동적 감정 loss 가중치 (학습 초반에는 낮게, 후반에는 높게)
        self.register_buffer('sentiment_loss_weight', torch.tensor(3.0))  # 10.0 → 3.0

        # 측면 감지 헤드
        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)
        self.aspect_head.bias.data.fill_(-4.0)

        # 🔥 개선된 감정 헤드 (더 깊은 네트워크)
        self.sentiment_intermediate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LayerNorm(512),  # 추가
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 추가
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.sentiment_head = nn.Linear(256, 12 * 3)

        if sentiment_class_weights is not None:
            self.register_buffer('sentiment_class_weights', sentiment_class_weights)
        else:
            self.register_buffer('sentiment_class_weights', None)

        if global_pos_weights is not None:
            self.register_buffer('global_pos_weights', global_pos_weights)
        else:
            self.register_buffer('global_pos_weights', torch.ones(12) * 10.0)

        if aspect_only:
            for param in self.sentiment_head.parameters():
                param.requires_grad = False
            for param in self.sentiment_intermediate.parameters():
                param.requires_grad = False
            print("🔒 감성 헤드가 동결되었습니다.")

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        aspect_logits = self.aspect_head(output)
        sentiment_features = self.sentiment_intermediate(output)
        sentiment_logits = self.sentiment_head(sentiment_features).view(-1, 12, 3)

        loss = None
        if labels is not None:
            aspect_labels = (labels != 0).float()
            aspect_loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.global_pos_weights)
            aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)

            if self.aspect_only:
                loss = aspect_loss
            else:
                weights_to_use = class_weights if class_weights is not None else self.sentiment_class_weights

                # 🔥 개선된 Focal Loss 사용
                sentiment_loss_fct = ImprovedFocalLoss(gamma=2.5, weight=weights_to_use)
                sentiment_loss = 0
                valid_aspect_count = 0

                for i in range(12):
                    mask = aspect_labels[:, i] == 1
                    if mask.sum() > 0:
                        sentiment_targets = torch.where(
                            labels[mask, i] > 0,
                            labels[mask, i] - 1,
                            torch.zeros_like(labels[mask, i])
                        )
                        loss_s = sentiment_loss_fct(
                            sentiment_logits[mask, i, :],
                            sentiment_targets
                        )
                        sentiment_loss += loss_s
                        valid_aspect_count += 1

                if valid_aspect_count > 0:
                    sentiment_loss = sentiment_loss / valid_aspect_count

                loss = aspect_loss + self.sentiment_loss_weight * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }

    def update_sentiment_loss_weight(self, new_weight):
        """동적으로 감정 loss 가중치 업데이트"""
        self.sentiment_loss_weight = torch.tensor(new_weight).to(self.sentiment_loss_weight.device)


# ==========================================
# 4. CustomTrainer
# ==========================================
class ImprovedCustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, train_sampler=None, aspect_only=False,
                 sentiment_loss_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_sampler = train_sampler
        self.aspect_only = aspect_only
        self.custom_step = 0
        self.sentiment_loss_schedule = sentiment_loss_schedule  # 새로 추가

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
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(next(model.parameters()).device)
        labels = inputs.pop("labels")

        # 🔥 동적 감정 loss 가중치 조정
        if not self.aspect_only and self.sentiment_loss_schedule is not None:
            current_epoch = self.state.epoch if self.state.epoch is not None else 0
            if current_epoch in self.sentiment_loss_schedule:
                new_weight = self.sentiment_loss_schedule[current_epoch]
                model.module.update_sentiment_loss_weight(new_weight) if hasattr(model,
                                                                                 'module') else model.update_sentiment_loss_weight(
                    new_weight)
                print(f"\n🔥 Epoch {current_epoch}: 감정 loss 가중치 → {new_weight}")

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
        print(f"\n📊 [Step {self.custom_step}] Batch Distribution")

        if self.aspect_only:
            print(f"   🎯 [모드: 측면 분류 전용]")
        else:
            print(f"   🎭 [모드: 측면 + 감정 분류]")
            # 감정 분포 출력
            sentiment_counts = {1: 0, 2: 0, 3: 0}
            for i in range(12):
                for s in [1, 2, 3]:
                    sentiment_counts[s] += (labels[:, i] == s).sum().item()

            total = sum(sentiment_counts.values())
            if total > 0:
                print(f"   감정 분포 - 긍정:{sentiment_counts[1]}({sentiment_counts[1] / total * 100:.1f}%) "
                      f"부정:{sentiment_counts[2]}({sentiment_counts[2] / total * 100:.1f}%) "
                      f"중립:{sentiment_counts[3]}({sentiment_counts[3] / total * 100:.1f}%)")

        aspect_counts = (labels != 0).sum(dim=0).cpu().numpy()
        for i in range(0, 12, 2):
            left = f"{ASPECT_NAMES[i]:12s}: {aspect_counts[i]:2d}개"
            right = f"{ASPECT_NAMES[i + 1]:12s}: {aspect_counts[i + 1]:2d}개" if i + 1 < 12 else ""
            print(f"   - {left}  |  {right}")
        print("-" * 60)


# ==========================================
# 5. 예측 함수 (차등 임계값 적용)
# ==========================================
# 🆕 측면별 최적 임계값 (FP 방지)
EVAL_THRESHOLDS = {
    0: 0.50,  # 커피/음료
    1: 0.52,  # 베이커리/빵
    2: 0.68,  # 케이크 (FP 방지)
    3: 0.76,  # 쿠키 (가장 높게)
    4: 0.50,  # 빙수/과일
    5: 0.64,  # 기타디저트 (FP 방지)
    6: 0.55,  # 공간/편의시설
    7: 0.58,  # 분위기/감성
    8: 0.52,  # 서비스
    9: 0.55,  # 가격/가성비
    10: 0.70,  # 선물/포장 (FP 방지)
    11: 0.65,  # 혼잡도/웨이팅
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


def create_improved_sampler(labels, aspect_counts, focus_on_minority=True):
    """
    개선된 샘플러 생성
    focus_on_minority=True: 부정/중립 집중 샘플링
    focus_on_minority=False: 균등 샘플링
    """
    sample_weights = []

    for label_str in labels:
        label = str(label_str).zfill(12)
        label_list = [int(c) for c in label]

        if focus_on_minority:
            # 🔥 부정/중립에 극단적 가중치
            has_negative = any(label_list[i] == 2 for i in range(12))
            has_neutral = any(label_list[i] == 3 for i in range(12))

            if has_negative:
                current_weight = 15.0  # 부정 매우 높게
            elif has_neutral:
                current_weight = 10.0  # 중립 높게
            else:
                current_weight = 1.0  # 긍정 낮게
        else:
            # 균등 샘플링 (측면 개수만 고려)
            current_weight = sum(1.0 for i in range(12) if label_list[i] > 0)
            current_weight = max(current_weight, 1.0)

        sample_weights.append(current_weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

# ==========================================
# 메인 실행 코드
# ==========================================
if __name__ == "__main__":

    if RUN_STAGE_1:
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
            df_single, test_size=100, random_state=SEED, shuffle=True
        )
        single_train, single_val = train_test_split(
            single_train_temp, test_size=100, random_state=SEED, shuffle=True
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
        FP_PRONE_ASPECTS_DICT = {2}  # 🔧 케이크, 기타디저트만 (쿠키, 선물/포장 제외)
        MODERATE_FP_ASPECTS = {7, 11}

        # 🆕 Recall 향상이 필요한 측면 (쿠키, 선물/포장)
        LOW_RECALL_ASPECTS = {3, 10}

        for i in range(12):
            if train_aspect_ratios_single[i] > 0:
                base_weight = (1 - train_aspect_ratios_single[i]) / train_aspect_ratios_single[i]

                # 🆕 Recall 향상 측면은 가중치 증가
                if i in LOW_RECALL_ASPECTS:
                    if i == 3:  # 쿠키
                        base_weight *= 1.2  # 1.3 → 1.2 (완화)
                    elif i == 10:  # 선물/포장
                        base_weight *= 1.5  # 1.3 → 1.5 (강화)

                # FP 방지 보정
                elif i in FP_PRONE_ASPECTS_DICT:
                    base_weight *= 0.7
                elif i in MODERATE_FP_ASPECTS:
                    base_weight *= 0.85

                global_pos_weights_single[i] = max(5.0, min(base_weight, 25.0))
            else:
                global_pos_weights_single[i] = 10.0

        print("\n📈 계산된 pos_weight (단일 측면):")
        print("-" * 60)
        for i, name in enumerate(ASPECT_NAMES):
            fp_tag = ""
            if i in LOW_RECALL_ASPECTS:
                fp_tag = " [Recall 향상 +30%]"
            elif i in FP_PRONE_ASPECTS_DICT:
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

        # 🆕 Recall 향상이 필요한 측면
        RECALL_BOOST_ASPECTS = {
            3: "쿠키/구움과자",
            10: "선물/포장"
        }

        sample_weights_single = []
        alpha = 1.0

        for label_str in train_labels_single:
            label_str = str(label_str).zfill(12)
            label_list = [int(c) for c in label_str]
            current_weight = 1.0

            #  모든 감정에 동일 가중치
            for i in range(12):
                if label_list[i] > 0:
                    current_weight += 1.0

            # # A. 희소 측면 가중치 (역수)
            # for i in range(12):
            #     if label_list[i] > 0:
            #         rarity_weight = 1.0 / (aspect_counts_single[i] + 1e-5)
            #
            #         # 🆕 Recall 향상 측면은 추가 가중치
            #         if i in RECALL_BOOST_ASPECTS.keys():
            #             if i == 10:  # 선물/포장
            #                 rarity_weight *= 3.0  # 2.0 → 3.0
            #             else:  # 쿠키
            #                 rarity_weight *= 1.5  # 2.0 → 1.5 (완화)
            #
            #         current_weight += rarity_weight
            #
            # # B. FP 취약 측면의 네거티브 샘플 강화
            # fp_negative_bonus = 0
            # for i in FP_PRONE_ASPECTS.keys():
            #     if label_list[i] == 0:
            #         fp_negative_bonus += aspect_counts_single[i] / (aspect_counts_single.sum() + 1e-5)
            #
            # current_weight += fp_negative_bonus * 8.0
            #
            # # C. 기본 가중치
            # if current_weight == 0:
            #     current_weight = 1.0
            #
            sample_weights_single.append(current_weight)

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
        model_single = ImprovedDualHeadBert(
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
            no_cuda=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=False,
        )

        # 1-9. 학습 시작 (단일 측면)
        print("\n🚀 [STAGE 1] 학습 시작...")
        trainer_single = ImprovedCustomTrainer(
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

    if RUN_STAGE_2:
        # ============================================================
        # STAGE 2: 복합 측면 데이터 추가 학습
        # ============================================================
        print("\n" + "=" * 80)
        print("🎯 STAGE 2: 복합 측면 데이터 추가 학습 시작")
        print("=" * 80 + "\n")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
            print(f"   메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        else:
            print("⚠️ CPU 사용 중")

        # 🆕 단일 측면 데이터 로드 (CSV에서)
        print("\n📂 [STAGE 2] 단일 측면 데이터 로딩 중...")
        try:
            df_single_train = pd.read_csv('split_train_single.csv', dtype={'label': str})
            print(f"✅ 단일 측면 학습 데이터: {len(df_single_train)}개")
        except FileNotFoundError:
            print(f"❌ split_train_single.csv를 찾을 수 없습니다.")
            print("💡 먼저 STAGE 1을 실행해주세요.")
            exit()

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

                # 🆕 Recall 향상 측면은 가중치 증가
                if i in LOW_RECALL_ASPECTS:
                    base_weight *= 1.3
                elif i in FP_PRONE_ASPECTS_DICT:
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
            if i in LOW_RECALL_ASPECTS:
                fp_tag = " [Recall 향상 +30%]"
            elif i in FP_PRONE_ASPECTS_DICT:
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

                    # 🆕 Recall 향상 측면은 추가 가중치
                    if i in RECALL_BOOST_ASPECTS.keys():
                        rarity_weight *= 2.0

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
        model_combined = ImprovedDualHeadBert(
            MODEL_NAME,
            aspect_only=True,
            global_pos_weights=global_pos_weights_tensor_combined
        ).to(device)

        # 🆕 STAGE 2를 이미 완료했다면 저장된 모델 로드
        if os.path.exists(f"{SAVE_PATH}/model_state_dict.pt"):
            print("✅ 저장된 STAGE 2 모델을 발견했습니다. 로드 중...")
            model_combined.load_state_dict(torch.load(f"{SAVE_PATH}/model_state_dict.pt"))
            print("✅ STAGE 2 모델 로드 완료! STAGE 3로 바로 진행합니다.")
        else:
            # STAGE 1에서 학습한 가중치 로드 (처음 실행할 때)
            model_combined.load_state_dict(model_single.state_dict())
            print("✅ STAGE 1 모델 가중치 로드 완료!")

        # 🆕 pos_weight를 STAGE 2 값으로 업데이트
        model_combined.global_pos_weights = global_pos_weights_tensor_combined.to(device)
        print("✅ pos_weight를 STAGE 2 값으로 업데이트 완료!")

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
        trainer_combined = ImprovedCustomTrainer(
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

    if RUN_STAGE_3:
        print("\n" + "=" * 80)
        print("🎯 STAGE 3: 감정 분류 학습 (개선 버전)")
        print("=" * 80 + "\n")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # 데이터 로드
        print("\n📂 데이터 로딩...")
        try:
            df_train = pd.read_csv('split_train_single.csv', dtype={'label': str})
            df_val = pd.read_csv('split_val_single.csv', dtype={'label': str})

            train_texts = df_train['Original_Review'].tolist()
            train_labels = df_train['label'].tolist()
            val_texts = df_val['Original_Review'].tolist()
            val_labels = df_val['label'].tolist()

            print(f"✅ 학습: {len(train_texts)}개, 검증: {len(val_texts)}개")
        except FileNotFoundError:
            print("❌ 데이터 파일 없음. STAGE 1-2 먼저 실행하세요.")
            exit()

        # 🔥 개선된 감정 가중치 계산 (더 극단적으로)
        print("\n📊 감정 분포 분석...")
        sentiment_counts = {1: 0, 2: 0, 3: 0}

        for label_str in train_labels:
            label = str(label_str).zfill(12)
            for i in range(12):
                s = int(label[i])
                if s > 0:
                    sentiment_counts[s] += 1

        total = sum(sentiment_counts.values())
        print(f"긍정: {sentiment_counts[1]} ({sentiment_counts[1] / total * 100:.1f}%)")
        print(f"부정: {sentiment_counts[2]} ({sentiment_counts[2] / total * 100:.1f}%)")
        print(f"중립: {sentiment_counts[3]} ({sentiment_counts[3] / total * 100:.1f}%)")

        # 🔥 극단적 가중치 적용
        sentiment_weights = np.zeros(3)
        for i in range(3):
            s_idx = i + 1
            ratio = sentiment_counts[s_idx] / total
            if ratio > 0:
                base_weight = (1 - ratio) / ratio

                # 부정/중립에 추가 배수
                if s_idx == 2:  # 부정
                    base_weight *= 2.5  # 극단적 강화
                elif s_idx == 3:  # 중립
                    base_weight *= 3.0  # 더 극단적 강화

                sentiment_weights[i] = max(1.0, min(base_weight, 30.0))
            else:
                sentiment_weights[i] = 15.0

        print(f"\n📈 감정 가중치: 긍정={sentiment_weights[0]:.1f}, "
              f"부정={sentiment_weights[1]:.1f}, 중립={sentiment_weights[2]:.1f}")

        sentiment_class_weights = torch.tensor(sentiment_weights, dtype=torch.float32).to(device)

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        # 데이터셋
        train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)

        # 모델 로드
        print("\n🔧 STAGE 2 모델 로딩...")
        global_pos_weights = torch.load(f"{SAVE_PATH}/global_pos_weights.pt", weights_only=True)

        model = ImprovedDualHeadBert(
            MODEL_NAME,
            aspect_only=False,
            global_pos_weights=global_pos_weights,
            sentiment_class_weights=sentiment_class_weights
        ).to(device)

        # STAGE 2 가중치 로드
        stage2_state = torch.load(f"{SAVE_PATH}/model_state_dict.pt", weights_only=False)
        model.load_state_dict(stage2_state, strict=False)
        print("✅ STAGE 2 가중치 로드 완료")

        # 🔥 Aspect head 동결, Sentiment head 활성화
        for param in model.aspect_head.parameters():
            param.requires_grad = False
        for param in model.sentiment_head.parameters():
            param.requires_grad = True
        for param in model.sentiment_intermediate.parameters():
            param.requires_grad = True

        # BERT 상위 레이어만 활성화
        for name, param in model.bert.named_parameters():
            if any(f'encoder.layer.{i}' in name for i in [11, 10, 9, 8]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("🔓 BERT 상위 4개 레이어 + Sentiment Head 활성화")

        # 🔥 부정/중립 집중 샘플러
        print("\n⚖️ 부정/중립 집중 Sampler 생성...")
        aspect_counts = np.zeros(12)
        for label_str in train_labels:
            label = str(label_str).zfill(12)
            for i in range(12):
                if int(label[i]) > 0:
                    aspect_counts[i] += 1

        sampler = create_improved_sampler(train_labels, aspect_counts, focus_on_minority=True)

        # 🔥 동적 감정 loss 가중치 스케줄
        sentiment_loss_schedule = {
            0: 3.0,  # 초반에는 낮게
            2: 5.0,  # 중반에 증가
            4: 7.0  # 후반에 높게
        }

        # 학습 설정
        training_args = TrainingArguments(
            output_dir="./checkpoints_sentiment_improved",
            num_train_epochs=6,  # 5 → 6
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=3e-5,  # 5e-5 → 3e-5 (약간 낮춤)
            warmup_steps=150,  # 100 → 150
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            save_total_limit=2,
            fp16=True,
            dataloader_num_workers=2,
        )

        # 학습
        print("\n🚀 학습 시작 (부정/중립 집중)...")
        trainer = ImprovedCustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            train_sampler=sampler,
            class_weights=sentiment_class_weights,
            aspect_only=False,
            sentiment_loss_schedule=sentiment_loss_schedule
        )

        trainer.train()

        # 저장
        print(f"\n💾 모델 저장... ({SAVE_PATH_SENTIMENT_SINGLE})")
        os.makedirs(SAVE_PATH_SENTIMENT_SINGLE, exist_ok=True)
        torch.save(model.state_dict(), f"{SAVE_PATH_SENTIMENT_SINGLE}/model_state_dict.pt")
        tokenizer.save_pretrained(SAVE_PATH_SENTIMENT_SINGLE)
        torch.save(global_pos_weights, f"{SAVE_PATH_SENTIMENT_SINGLE}/global_pos_weights.pt")
        torch.save(sentiment_class_weights, f"{SAVE_PATH_SENTIMENT_SINGLE}/sentiment_class_weights.pt")
        print("✅ STAGE 3 완료!")

    if RUN_STAGE_4:
        print("\n" + "=" * 80)
        print("🎯 STAGE 4: 최종 학습 (개선 버전)")
        print("=" * 80 + "\n")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 데이터 로드
        print("\n📂 결합 데이터 로딩...")
        try:
            df_train = pd.read_csv('split_train_combined.csv', dtype={'label': str})
            df_val = pd.read_csv('split_val_combined.csv', dtype={'label': str})

            train_texts = df_train['Original_Review'].tolist()
            train_labels = df_train['label'].tolist()
            val_texts = df_val['Original_Review'].tolist()
            val_labels = df_val['label'].tolist()

            print(f"✅ 학습: {len(train_texts)}개, 검증: {len(val_texts)}개")
        except FileNotFoundError:
            print("❌ 데이터 파일 없음.")
            exit()

        # 감정 가중치 재계산
        print("\n📊 감정 분포 재분석...")
        sentiment_counts = {1: 0, 2: 0, 3: 0}

        for label_str in train_labels:
            label = str(label_str).zfill(12)
            for i in range(12):
                s = int(label[i])
                if s > 0:
                    sentiment_counts[s] += 1

        total = sum(sentiment_counts.values())
        print(f"긍정: {sentiment_counts[1]} ({sentiment_counts[1] / total * 100:.1f}%)")
        print(f"부정: {sentiment_counts[2]} ({sentiment_counts[2] / total * 100:.1f}%)")
        print(f"중립: {sentiment_counts[3]} ({sentiment_counts[3] / total * 100:.1f}%)")

        sentiment_weights = np.zeros(3)
        for i in range(3):
            s_idx = i + 1
            ratio = sentiment_counts[s_idx] / total
            if ratio > 0:
                base_weight = (1 - ratio) / ratio
                if s_idx == 2:
                    base_weight *= 2.0
                elif s_idx == 3:
                    base_weight *= 2.5
                sentiment_weights[i] = max(1.0, min(base_weight, 25.0))
            else:
                sentiment_weights[i] = 12.0

        sentiment_class_weights = torch.tensor(sentiment_weights, dtype=torch.float32).to(device)

        # STAGE 3 모델 로드
        print("\n🔧 STAGE 3 모델 로딩...")
        global_pos_weights = torch.load(f"{SAVE_PATH_SENTIMENT_SINGLE}/global_pos_weights.pt", weights_only=True)

        model = ImprovedDualHeadBert(
            MODEL_NAME,
            aspect_only=False,
            global_pos_weights=global_pos_weights,
            sentiment_class_weights=sentiment_class_weights
        ).to(device)

        model.load_state_dict(torch.load(f"{SAVE_PATH_SENTIMENT_SINGLE}/model_state_dict.pt", weights_only=False))
        print("✅ STAGE 3 가중치 로드 완료")

        # Aspect head 해제
        for param in model.aspect_head.parameters():
            param.requires_grad = True
        print("🔓 Aspect head 활성화")

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)

        # ============================================================
        # Phase 1: 부정/중립 집중 (Epoch 1-3)
        # ============================================================
        print("\n" + "=" * 60)
        print("📚 Phase 1: 부정/중립 집중 학습")
        print("=" * 60)

        aspect_counts = np.zeros(12)
        for label_str in train_labels:
            label = str(label_str).zfill(12)
            for i in range(12):
                if int(label[i]) > 0:
                    aspect_counts[i] += 1

        sampler_phase1 = create_improved_sampler(train_labels, aspect_counts, focus_on_minority=True)

        training_args_phase1 = TrainingArguments(
            output_dir="./checkpoints_phase1_improved",
            num_train_epochs=3,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            save_total_limit=2,
            fp16=True,
        )

        trainer_phase1 = ImprovedCustomTrainer(
            model=model,
            args=training_args_phase1,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            train_sampler=sampler_phase1,
            class_weights=sentiment_class_weights,
            aspect_only=False,
            sentiment_loss_schedule={0: 4.0, 1: 6.0, 2: 8.0}
        )

        print("🚀 Phase 1 학습...")
        trainer_phase1.train()

        # ============================================================
        # Phase 2: 균등 학습 (Epoch 4-6)
        # ============================================================
        print("\n" + "=" * 60)
        print("📚 Phase 2: 균등 학습")
        print("=" * 60)

        sampler_phase2 = create_improved_sampler(train_labels, aspect_counts, focus_on_minority=False)

        training_args_phase2 = TrainingArguments(
            output_dir="./checkpoints_phase2_improved",
            num_train_epochs=3,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=1e-5,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            save_total_limit=2,
            fp16=True,
        )

        trainer_phase2 = ImprovedCustomTrainer(
            model=model,
            args=training_args_phase2,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            train_sampler=sampler_phase2,
            class_weights=sentiment_class_weights,
            aspect_only=False,
        )

        print("🚀 Phase 2 학습...")
        trainer_phase2.train()

        # ============================================================
        # Phase 3: Fine-tuning (Epoch 7-8)
        # ============================================================
        print("\n" + "=" * 60)
        print("📚 Phase 3: Fine-tuning")
        print("=" * 60)

        training_args_phase3 = TrainingArguments(
            output_dir="./checkpoints_phase3_improved",
            num_train_epochs=2,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=5e-6,
            warmup_steps=0,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            save_total_limit=2,
            fp16=True,
        )

        trainer_phase3 = ImprovedCustomTrainer(
            model=model,
            args=training_args_phase3,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            train_sampler=sampler_phase2,
            class_weights=sentiment_class_weights,
            aspect_only=False,
        )

        print("🚀 Phase 3 학습...")
        trainer_phase3.train()

        # 최종 저장
        print(f"\n💾 최종 모델 저장... ({SAVE_PATH_SENTIMENT_FINAL})")
        os.makedirs(SAVE_PATH_SENTIMENT_FINAL, exist_ok=True)
        torch.save(model.state_dict(), f"{SAVE_PATH_SENTIMENT_FINAL}/model_state_dict.pt")
        tokenizer.save_pretrained(SAVE_PATH_SENTIMENT_FINAL)
        torch.save(global_pos_weights, f"{SAVE_PATH_SENTIMENT_FINAL}/global_pos_weights.pt")
        torch.save(sentiment_class_weights, f"{SAVE_PATH_SENTIMENT_FINAL}/sentiment_class_weights.pt")
        print("✅ STAGE 4 완료!")

    print("\n" + "=" * 80)
    print("✅ 전체 학습 완료!")
    print("=" * 80)

    # 완료 메시지
    print("\n" + "=" * 80)
    print("✅ 감정 분류 학습 완료!")
    print("=" * 80)
    print("\n📁 생성된 모델:")
    if RUN_STAGE_3:
        print(f"   - STAGE 3 (감정-단일): {SAVE_PATH_SENTIMENT_SINGLE}/")
    if RUN_STAGE_4:
        print(f"   - STAGE 4 (감정-최종): {SAVE_PATH_SENTIMENT_FINAL}/")
    print("\n📈 학습 구조:")
    print("   - STAGE 3: 단일 측면 감정 학습 (5 epoch)")
    print("   - STAGE 4: 단일+복합 측면 감정 학습 (3 epoch)")

    # 2-11. 테스트 평가 (복합 측면)
    print("\n" + "=" * 60)
    print("📊 [STAGE 2] 테스트 데이터셋 평가 (차등 임계값 적용)")
    print("=" * 60)

    if RUN_STAGE_2:

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