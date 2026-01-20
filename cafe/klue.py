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
#
# # ==========================================
# # 🆕 Hybrid 모델: 측면 정보를 감성 분류에 활용
# # ==========================================
# class HybridAspectSentiment(nn.Module):
#     def __init__(self, model_name):
#         super(HybridAspectSentiment, self).__init__()
#         self.bert = BertModel.from_pretrained(model_name)
#         self.drop = nn.Dropout(p=0.3)
#
#         # Step 1: 측면 감지 (보조 정보)
#         self.aspect_detector = nn.Linear(self.bert.config.hidden_size, 12)
#
#         # Step 2: 측면 정보 + BERT 임베딩 결합
#         # 768(BERT) + 12(aspect probs) = 780
#         self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size + 12, 12 * 4)
#
#     def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output  # [batch, 768]
#         output = self.drop(pooled_output)
#
#         # Step 1: 측면 확률 계산
#         aspect_logits = self.aspect_detector(output)  # [batch, 12]
#         aspect_probs = torch.sigmoid(aspect_logits)  # [batch, 12]
#
#         # Step 2: BERT 임베딩 + 측면 확률 결합
#         combined = torch.cat([output, aspect_probs], dim=1)  # [batch, 780]
#
#         # Step 3: 감성 분류
#         sentiment_logits = self.sentiment_classifier(combined)
#         sentiment_logits = sentiment_logits.view(-1, 12, 4)  # [batch, 12, 4]
#
#         loss = None
#         if labels is not None:
#             # Loss 1: 측면 감지 (보조 loss)
#             aspect_labels = (labels != 0).float()  # [batch, 12]
#             # 측면이 있는 경우(1)에 5배 더 가중치를 줌 (측면을 더 적극적으로 찾게 함)
#             pos_weight = torch.ones([12]).to(labels.device) * 5.0
#             aspect_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#             aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)
#
#             # Loss 2: 감성 분류 (주 loss)
#             if class_weights is not None:
#                 weights = class_weights.to(labels.device)
#             else:
#                 weights = None
#
#             sentiment_loss_fct = FocalLoss(gamma=2.0,weight=weights)
#             sentiment_loss = 0
#             for i in range(12):
#                 sentiment_loss += sentiment_loss_fct(
#                     sentiment_logits[:, i, :],
#                     labels[:, i]
#                 )
#
#             # 전체 Loss: 감성 분류가 주, 측면 감지가 보조
#             loss = 0.2 * aspect_loss + sentiment_loss
#
#         return {
#             'loss': loss,
#             'logits': sentiment_logits,
#             'aspect_probs': aspect_probs
#         }
#

# ==========================================
# 🆕 Phase별 학습 전략 구현
# ==========================================

class PhaseConfig:
    """각 Phase의 학습 전략 정의"""

    def __init__(self, phase_name, epochs, aspect_weight, sentiment_weight, freeze_sentiment):
        self.phase_name = phase_name
        self.epochs = epochs
        self.aspect_weight = aspect_weight
        self.sentiment_weight = sentiment_weight
        self.freeze_sentiment = freeze_sentiment

    def get_safe_dirname(self):
        """Windows 호환 디렉토리명 생성"""
        return self.phase_name.replace(':', '').replace(' ', '_')

    def __str__(self):
        freeze_status = "🔒 Frozen" if self.freeze_sentiment else "🔓 Active"
        return (f"\n{'=' * 60}\n"
                f"📍 {self.phase_name}\n"
                f"{'=' * 60}\n"
                f"   Epochs: {self.epochs}\n"
                f"   Loss = {self.aspect_weight:.1f}×Aspect + {self.sentiment_weight:.1f}×Sentiment\n"
                f"   Sentiment Head: {freeze_status}\n"
                f"{'=' * 60}")


# Phase 설정
PHASES = [
    PhaseConfig(
        phase_name="Phase 1 - Aspect Focus",  # 🔧 콜론 제거
        epochs=10,
        aspect_weight=1.0,
        sentiment_weight=0.0,
        freeze_sentiment=True  # 🔒 감성 헤드 동결
    ),
    PhaseConfig(
        phase_name="Phase 2 - Sentiment Warm-up",  # 🔧 콜론 제거
        epochs=3,
        aspect_weight=1.0,
        sentiment_weight=0.3,  # 서서히 깨우기
        freeze_sentiment=False  # 🔓 감성 헤드 활성화
    ),
    PhaseConfig(
        phase_name="Phase 3 - Full Training",  # 🔧 콜론 제거
        epochs=4,
        aspect_weight=1.0,
        sentiment_weight=1.0,  # 완전 학습
        freeze_sentiment=False
    )
]

class DualHeadBert(nn.Module):
    def __init__(self, model_name):
        super(DualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)

        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 12 * 3)

        # [핵심] Bias 초기화: 시작하자마자 확률 0.01(-4.59) 정도를 뱉게 함
        # 이렇게 하면 초반에 "대부분 없음"인 정답을 보고도 Loss가 폭발하지 않아
        # "있는 것"을 찾을 여유가 생김.
        self.aspect_head.bias.data.fill_(-4.0)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        aspect_logits = self.aspect_head(output)
        sentiment_logits = self.sentiment_head(output).view(-1, 12, 3)

        loss = None
        if labels is not None:
            aspect_labels = (labels != 0).float()

            # [수정 1] pos_weight를 동적으로 계산하거나 훨씬 크게 설정 (예: 15.0)
            # 여기서는 배치 내 비율로 계산하는 안전장치를 둠 + 최소값 보장
            num_pos = aspect_labels.sum()
            num_neg = aspect_labels.numel() - num_pos
            # pos가 0개면 에러나므로 1e-5 더함. 비율이 100:1이면 weight는 100이 됨.
            pos_weight_val = (num_neg / (num_pos + 1e-5)).item()
            # 너무 크면 튀니까 최대 20정도로 클리핑하거나, 고정값 15.0 추천
            final_pos_weight = torch.tensor([min(max(pos_weight_val, 10.0), 30.0)] * 12).to(labels.device)

            # BCE Loss 계산
            aspect_loss_fct = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
            aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)

            # Loss 2: 감성 분류
            if class_weights is not None:
                sentiment_weights = class_weights.to(labels.device)
            else:
                sentiment_weights = None

            sentiment_loss_fct = FocalLoss(gamma=2.0, weight=sentiment_weights)  # Gamma 2.0 추천
            sentiment_loss = 0
            valid_aspect_count = 0  # [수정 3] 평균을 내기 위한 카운터

            for i in range(12):
                mask = aspect_labels[:, i] == 1
                if mask.sum() > 0:
                    loss_s = sentiment_loss_fct(
                        sentiment_logits[mask, i, :],
                        # labels -1 처리 (0:긍정, 1:부정, 2:중립)
                        torch.where(labels[mask, i] > 0, labels[mask, i] - 1, torch.zeros_like(labels[mask, i]))
                    )
                    sentiment_loss += loss_s
                    valid_aspect_count += 1

            # [수정 3] 합이 아니라 평균으로 처리하여 Loss 스케일 유지
            if valid_aspect_count > 0:
                sentiment_loss = sentiment_loss / valid_aspect_count

            # Sentiment Loss의 중요도 더 높임 (Aspect는 쉬운 편이므로)
            loss = aspect_loss + 0.0 * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }


# ==========================================
# CustomTrainer 수정 (Phase 적용)
# ==========================================
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, train_sampler=None,
                 current_phase=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_sampler = train_sampler
        self.current_phase = current_phase  # 🆕 현재 Phase 정보
        self.custom_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        if not hasattr(self, 'custom_step'):
            self.custom_step = 0

        if model.training and self.custom_step % 50 == 0:
            self._check_batch_distribution(labels)

        self.custom_step += 1

        # 🆕 Phase별 가중치 적용
        outputs = model(**inputs, labels=labels, class_weights=self.class_weights)

        # Phase 1일 때는 aspect_loss만 사용
        if self.current_phase and self.current_phase.sentiment_weight == 0.0:
            loss = outputs['loss']  # 이미 aspect_loss만 계산됨
        else:
            loss = outputs['loss']  # aspect + sentiment 조합 loss

        return (loss, outputs) if return_outputs else loss

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

    def _check_batch_distribution(self, labels):
        """배치 내의 감성 비율 및 측면별 등장 횟수 출력"""
        total_elements = labels.numel()
        none_count = (labels == 0).sum().item()
        pos_count = (labels == 1).sum().item()
        neg_count = (labels == 2).sum().item()
        neu_count = (labels == 3).sum().item()

        batch_size = labels.size(0)

        print(f"\n📊 [Step {self.custom_step}] Batch Distribution Analysis")

        # 🆕 현재 Phase 정보 표시
        if self.current_phase:
            print(f"   [Current Phase: {self.current_phase.phase_name}]")
            print(f"   [Loss Weight: {self.current_phase.aspect_weight:.1f}×Aspect + "
                  f"{self.current_phase.sentiment_weight:.1f}×Sentiment]")

        print(f"   [1. 감성 비율]")
        print(f"   - 긍정(1): {pos_count:3d}개 ({pos_count / total_elements:4.1%})")
        print(f"   - 부정(2): {neg_count:3d}개 ({neg_count / total_elements:4.1%})")
        print(f"   - 중립(3): {neu_count:3d}개 ({neu_count / total_elements:4.1%})")
        print(f"   - 없음(0): {none_count:3d}개 ({none_count / total_elements:4.1%})")

        print(f"\n   [2. 측면별 등장 빈도 (총 {batch_size}개 샘플 중)]")
        aspect_counts = (labels != 0).sum(dim=0).cpu().numpy()

        for i in range(0, 12, 2):
            left_aspect = f"{ASPECT_NAMES[i]:12s}: {aspect_counts[i]:2d}개"
            right_aspect = f"{ASPECT_NAMES[i + 1]:12s}: {aspect_counts[i + 1]:2d}개" if i + 1 < 12 else ""
            print(f"   - {left_aspect}  |  {right_aspect}")
        print("-" * 60)


# ==========================================
# DualHeadBert 모델 수정 (Phase별 Loss 계산)
# ==========================================
class DualHeadBert(nn.Module):
    def __init__(self, model_name):
        super(DualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)

        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 12 * 3)

        # Bias 초기화
        self.aspect_head.bias.data.fill_(-4.0)

        # 🆕 Phase 정보 저장용
        self.current_phase = None

    def set_phase(self, phase_config):
        """Phase 전환 시 호출"""
        self.current_phase = phase_config

        # Sentiment Head Freeze/Unfreeze
        for param in self.sentiment_head.parameters():
            param.requires_grad = not phase_config.freeze_sentiment

        print(f"\n🔧 Model Phase Updated:")
        print(f"   {phase_config.phase_name}")
        print(f"   Sentiment Head: {'🔒 Frozen' if phase_config.freeze_sentiment else '🔓 Active'}")

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        aspect_logits = self.aspect_head(output)
        sentiment_logits = self.sentiment_head(output).view(-1, 12, 3)

        loss = None
        if labels is not None:
            aspect_labels = (labels != 0).float()

            # Aspect Loss 계산
            num_pos = aspect_labels.sum()
            num_neg = aspect_labels.numel() - num_pos
            pos_weight_val = (num_neg / (num_pos + 1e-5)).item()
            final_pos_weight = torch.tensor([min(max(pos_weight_val, 10.0), 30.0)] * 12).to(labels.device)

            aspect_loss_fct = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
            aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)

            # Sentiment Loss 계산
            sentiment_loss = 0
            if self.current_phase is None or self.current_phase.sentiment_weight > 0:
                if class_weights is not None:
                    sentiment_weights = class_weights.to(labels.device)
                else:
                    sentiment_weights = None

                sentiment_loss_fct = FocalLoss(gamma=2.0, weight=sentiment_weights)
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

            # 🆕 Phase별 Loss 가중치 적용
            if self.current_phase:
                loss = (self.current_phase.aspect_weight * aspect_loss +
                        self.current_phase.sentiment_weight * sentiment_loss)
            else:
                # Phase 미지정 시 기본값
                loss = aspect_loss + 1.0 * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }

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

    class_weights = torch.tensor([
        1.0,  # 긍정 (index 0)
        5.0,  # 부정 (index 1)
        15.0  # 중립 (index 2)
    ], dtype=torch.float)
    # ==========================================
    # 🆕 WeightedRandomSampler 준비
    # ==========================================
    from torch.utils.data import WeightedRandomSampler
    import numpy as np

    print("\n⚖️ 희소성 가중치(Sparsity Weight) 기반 Sampler 계산 중...")

    # 1. 각 측면 및 감성의 출현 빈도 미리 계산 (전체 학습 데이터 기준)
    aspect_counts = np.zeros(12)
    sentiment_counts = np.zeros(4)  # 0:없음, 1:긍정, 2:부정, 3:중립

    for label_str in train_labels:
        label_str = str(label_str).zfill(12)
        for i, c in enumerate(label_str):
            val = int(c)
            if val > 0:
                aspect_counts[i] += 1  # 해당 측면이 등장한 횟수
            sentiment_counts[val] += 1  # 각 감성이 등장한 총 횟수

    # 2. 샘플별 가중치 계산 루프
    sample_weights = []
    alpha = 10.0  # 가중치 증폭 계수 (희귀할수록 점수가 확 뛰게 함)

    for label_str in train_labels:
        label_str = str(label_str).zfill(12)
        label_list = [int(c) for c in label_str]

        current_sample_weight = 0

        for i, s_val in enumerate(label_list):
            if s_val > 0:
                # 💡 이미지 수식 적용: 1/Count(측면) + 1/Count(감성)
                # 해당 샘플이 가진 측면과 감성이 희귀할수록 분모가 작아져 weight가 커짐
                a_weight = 1.0 / (aspect_counts[i] + 1e-5)
                s_weight = 1.0 / (sentiment_counts[s_val] + 1e-5)

                # 측면 + 감성의 교집합 가중치 합산
                current_sample_weight += (a_weight + s_weight)

        # 아무것도 없는 샘플(전부 0)은 최소 가중치 부여
        if current_sample_weight == 0:
            current_sample_weight = 1.0 / (sentiment_counts[0] + 1e-5)

        # 최종 점수에 알파(증폭) 적용
        sample_weights.append(current_sample_weight * alpha)

    # 3. Sampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"✅ 교집합 가중치 기반 Sampler 생성 완료")
    print(f"   (가장 희귀한 측면+감성 조합이 우선적으로 배치에 채워집니다.)")


    # 3. 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CafeAspectDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = CafeAspectDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)  # 🆕

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadBert(MODEL_NAME).to(device)

    # 🆕 Phase별 순차 학습
    for phase in PHASES:
        print(phase)  # Phase 정보 출력

        # 모델에 Phase 설정
        model.set_phase(phase)

        # TrainingArguments 설정
        training_args = TrainingArguments(
            output_dir=f"./checkpoints/{phase.phase_name.replace(' ', '_')}",
            num_train_epochs=phase.epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
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

        # Trainer 생성
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=class_weights.to(device),
            train_sampler=sampler if phase.phase_name == "Phase 1: Aspect Focus" else None,
            current_phase=phase  # 🆕 Phase 정보 전달
        )

        # 학습 시작
        trainer.train()

        print(f"✅ {phase.phase_name} 완료!\n")

    # 최종 모델 저장
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

            # 측면 감지 결과 [batch, 12]
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            aspect_detected = (aspect_probs > 0.5).long()

            # 감성 분류 결과 [batch, 12] (0, 1, 2 중 하나)
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=2)

            # 최종 예측 값 복원 (측면이 없으면 0, 있으면 감성+1)
            preds = torch.where(
                aspect_detected == 1,
                sentiment_preds + 1,
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

    # for review in test_reviews:
    #     predict_review(model, tokenizer, review, device)
