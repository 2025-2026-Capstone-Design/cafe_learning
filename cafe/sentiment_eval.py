import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import os

# ==========================================
# 0. 설정
# ==========================================
MODEL_NAME = "klue/bert-base"
MODEL_PATH = "./aspect_sentiment_final_model"  # STAGE 3 모델 경로
TEST_DATA_PATH = ["split_test_combined.csv", "test_hard_clean.csv"]  # 테스트 데이터
MAX_LEN = 128
BATCH_SIZE = 32
SEED = 42

ASPECT_NAMES = [
    "커피/음료", "베이커리/빵", "케이크", "쿠키/구움과자",
    "빙수/과일", "기타 디저트", "공간/편의시설", "분위기/감성",
    "서비스", "가격/가성비", "선물/포장", "혼잡도/웨이팅"
]

SENTIMENT_NAMES = ["해당없음", "긍정", "부정", "중립"]

EVAL_THRESHOLDS = {
    0: 0.50, 1: 0.52, 2: 0.68, 3: 0.76, 4: 0.50, 5: 0.64,
    6: 0.55, 7: 0.58, 8: 0.52, 9: 0.55, 10: 0.70, 11: 0.65,
}


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
# 3. 모델 클래스
# ==========================================
class DualHeadBert(nn.Module):
    def __init__(self, model_name, aspect_only=False, global_pos_weights=None, sentiment_class_weights=None):
        super(DualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.4)
        self.aspect_only = aspect_only
        self.sentiment_loss_weight = 3.0  # 학습 코드와 동일하게 설정

        # 측면 감지 헤드
        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)

        # 🆕 학습 코드와 동일한 중간 레이어 추가
        self.sentiment_intermediate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.sentiment_head = nn.Linear(256, 12 * 3)  # 768이 아닌 256을 입력으로 받음

        if sentiment_class_weights is not None:
            self.register_buffer('sentiment_class_weights', sentiment_class_weights)
        else:
            self.register_buffer('sentiment_class_weights', None)

        if global_pos_weights is not None:
            self.register_buffer('global_pos_weights', global_pos_weights)
        else:
            self.register_buffer('global_pos_weights', torch.ones(12) * 10.0)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        aspect_logits = self.aspect_head(output)

        # 🆕 수정: sentiment_intermediate를 거친 피처를 헤드에 전달
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
                sentiment_loss_fct = FocalLoss(gamma=2.0, weight=weights_to_use)
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

                loss = aspect_loss + self.sentiment_loss_weight * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }


# ==========================================
# 4. 평가 함수
# ==========================================
def evaluate_sentiment_per_aspect(model, dataloader, device, thresholds):
    """측면별 감정 분류 평가"""
    model.eval()

    # 측면별 예측/라벨 저장
    aspect_sentiment_preds = {i: [] for i in range(12)}
    aspect_sentiment_labels = {i: [] for i in range(12)}

    # 전체 측면 감지 성능 저장
    all_aspect_preds = []
    all_aspect_labels = []

    print("\n🔍 테스트 데이터 평가 중...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            sentiment_logits = outputs['sentiment_logits']  # (batch, 12, 3)
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)  # (batch, 12)

            # 측면 감지 (차등 임계값)
            aspect_detected = torch.zeros_like(aspect_probs, dtype=torch.bool)
            for i in range(12):
                aspect_detected[:, i] = aspect_probs[:, i] > thresholds[i]

            # 실제 측면 라벨 (0이 아닌 것)
            aspect_labels = (labels != 0)

            # 전체 측면 감지 성능 저장
            all_aspect_preds.append(aspect_detected.cpu())
            all_aspect_labels.append(aspect_labels.cpu())

            # 측면별로 분리
            for i in range(12):
                # 🆕 모든 샘플을 평가에 포함 (mask 제거)
                batch_size = labels.size(0)

                # 실제 라벨: 0(해당없음), 1(긍정), 2(부정), 3(중립)
                true_sentiment = labels[:, i].clone()  # 전체 배치

                # 예측 라벨 생성:
                # - 측면 감지 안 됨: 0(해당없음)
                # - 측면 감지 됨: 감정 예측값 + 1
                pred_sentiment = torch.zeros(batch_size, dtype=torch.long, device=labels.device)
                detected_mask = aspect_detected[:, i]
                pred_sentiment[detected_mask] = sentiment_preds[detected_mask, i] + 1

                aspect_sentiment_labels[i].extend(true_sentiment.cpu().numpy().tolist())
                aspect_sentiment_preds[i].extend(pred_sentiment.cpu().numpy().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"   진행: {batch_idx + 1}/{len(dataloader)} 배치")

    return aspect_sentiment_preds, aspect_sentiment_labels, all_aspect_preds, all_aspect_labels


def print_aspect_detection_performance(all_aspect_preds, all_aspect_labels, thresholds):
    """측면 감지 성능 출력"""
    all_aspect_preds = torch.cat(all_aspect_preds, dim=0)
    all_aspect_labels = torch.cat(all_aspect_labels, dim=0)

    print("\n" + "=" * 90)
    print("📊 측면 감지 성능 (Aspect Detection Performance)")
    print("=" * 90)
    print(f"{'측면':<15} {'F1':>6} {'Prec':>6} {'Recall':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'임계값':>6}")
    print("-" * 90)

    for i, aspect_name in enumerate(ASPECT_NAMES):
        tp = ((all_aspect_preds[:, i] == 1) & (all_aspect_labels[:, i] == 1)).sum().item()
        fp = ((all_aspect_preds[:, i] == 1) & (all_aspect_labels[:, i] == 0)).sum().item()
        fn = ((all_aspect_preds[:, i] == 0) & (all_aspect_labels[:, i] == 1)).sum().item()
        tn = ((all_aspect_preds[:, i] == 0) & (all_aspect_labels[:, i] == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        print(f"{aspect_name:<15} {f1:>6.3f} {precision:>6.3f} {recall:>6.3f} {accuracy:>6.3f} "
              f"{tp:>4} {fp:>4} {fn:>4} {thresholds[i]:>6.2f}")

    print("-" * 90)


def print_sentiment_results(aspect_sentiment_preds, aspect_sentiment_labels):
    """측면별 감정 분류 결과 출력"""
    print("\n" + "=" * 80)
    print("🎭 측면별 감정 분류 성능 (Sentiment Classification per Aspect)")
    print("=" * 80)

    overall_results = []

    for i, aspect_name in enumerate(ASPECT_NAMES):
        if len(aspect_sentiment_labels[i]) == 0:
            print(f"\n[{aspect_name}] - 테스트 샘플 없음")
            continue

        y_true = np.array(aspect_sentiment_labels[i])
        y_pred = np.array(aspect_sentiment_preds[i])

        # F1 Score (macro)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0,labels=[0,1,2,3])
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0,labels=[0,1,2,3])

        # 클래스별 F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0,labels=[0,1,2,3])

        print(f"\n{'=' * 80}")
        print(f"[{aspect_name}]")
        print(f"  샘플 수: {len(y_true)}개")
        print(f"  F1 (Macro): {f1_macro:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")
        print(f"\n  클래스별 F1:")
        for j, sentiment in enumerate(SENTIMENT_NAMES):
            print(f"    {sentiment}: {f1_per_class[j]:.4f}")

        # Classification Report
        print(f"\n  상세 리포트:")
        report = classification_report(
            y_true, y_pred,
            target_names=SENTIMENT_NAMES,
            labels=[0,1,2,3],
            zero_division=0,
            digits=4
        )
        print("  " + report.replace("\n", "\n  "))

        overall_results.append({
            'aspect': aspect_name,
            'samples': len(y_true),
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_none':f1_per_class[0],
            'f1_positive': f1_per_class[1],
            'f1_negative': f1_per_class[2],
            'f1_neutral': f1_per_class[3]
        })

    # 전체 평균
    print("\n" + "=" * 80)
    print("📈 전체 평균 성능")
    print("=" * 80)

    df_results = pd.DataFrame(overall_results)

    # 샘플 가중 평균
    total_samples = df_results['samples'].sum()
    weighted_f1_macro = (df_results['f1_macro'] * df_results['samples']).sum() / total_samples
    weighted_f1_weighted = (df_results['f1_weighted'] * df_results['samples']).sum() / total_samples

    print(f"\n전체 F1 (Macro, 샘플 가중): {weighted_f1_macro:.4f}")
    print(f"전체 F1 (Weighted, 샘플 가중): {weighted_f1_weighted:.4f}")

    print(f"\n측면별 평균:")
    print(f"  해당없음 F1: {df_results['f1_none'].mean():.4f}")
    print(f"  긍정 F1: {df_results['f1_positive'].mean():.4f}")
    print(f"  부정 F1: {df_results['f1_negative'].mean():.4f}")
    print(f"  중립 F1: {df_results['f1_neutral'].mean():.4f}")

    # 결과 테이블
    print(f"\n측면별 요약 테이블:")
    print("-" * 80)
    print(f"{'측면':<15} {'샘플':>6} {'F1(M)':>8} {'F1(W)':>8} {'없음':>6} {'긍정':>6} {'부정':>6} {'중립':>6}")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"{row['aspect']:<15} {row['samples']:>6} "
              f"{row['f1_macro']:>8.4f} {row['f1_weighted']:>8.4f} "
              f"{row['f1_none']:>6.4f} {row['f1_positive']:>6.4f} {row['f1_negative']:>6.4f} {row['f1_neutral']:>6.4f}")
    print("-" * 80)

    return df_results


def plot_confusion_matrices(aspect_sentiment_preds, aspect_sentiment_labels, save_dir='./evaluation_results'):
    """측면별 혼동 행렬 시각화"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📊 혼동 행렬 생성 중... (저장 경로: {save_dir})")

    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    # plt.rcParams['font.family'] = 'AppleGothic'  # Mac
    plt.rcParams['axes.unicode_minus'] = False

    # 4x3 그리드로 12개 측면 표시
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle('측면별 감정 분류 혼동 행렬 (Confusion Matrix per Aspect)',
                 fontsize=20, fontweight='bold', y=0.995)

    axes = axes.flatten()

    for i, aspect_name in enumerate(ASPECT_NAMES):
        ax = axes[i]

        if len(aspect_sentiment_labels[i]) == 0:
            ax.text(0.5, 0.5, '샘플 없음',
                    ha='center', va='center', fontsize=14)
            ax.set_title(f'{aspect_name}\n(0개)', fontsize=12, fontweight='bold')
            ax.axis('off')
            continue

        y_true = np.array(aspect_sentiment_labels[i])
        y_pred = np.array(aspect_sentiment_preds[i])

        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2 , 3])

        # 정규화 (행 기준)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # F1 스코어 계산
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 히트맵 그리기
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=SENTIMENT_NAMES, yticklabels=SENTIMENT_NAMES,
                    cbar=True, ax=ax, vmin=0, vmax=1,
                    annot_kws={'size': 8})

        ax.set_title(f'{aspect_name}\n(샘플: {len(y_true)}개, F1: {f1_macro:.3f})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('예측', fontsize=10)
        ax.set_ylabel('실제', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrices_per_aspect.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 혼동 행렬 저장 완료: {save_path}")
    plt.close()

    # 개별 측면 상세 혼동 행렬 (큰 사이즈)
    for i, aspect_name in enumerate(ASPECT_NAMES):
        if len(aspect_sentiment_labels[i]) == 0:
            continue

        y_true = np.array(aspect_sentiment_labels[i])
        y_pred = np.array(aspect_sentiment_preds[i])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2 , 3])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=SENTIMENT_NAMES, yticklabels=SENTIMENT_NAMES,
                    cbar=True, ax=ax, vmin=0, vmax=1, annot_kws={'size': 12})

        ax.set_title(f'{aspect_name} - 감정 분류 혼동 행렬\n샘플: {len(y_true)}개, F1 (Macro): {f1_macro:.4f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('예측', fontsize=12)
        ax.set_ylabel('실제', fontsize=12)

        plt.tight_layout()
        safe_aspect_name=aspect_name.replace('/','_').replace('\\','_')
        save_path = os.path.join(save_dir, f'cm_{i:02d}_{safe_aspect_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ 개별 혼동 행렬 저장 완료 ({save_dir}/)")


def plot_f1_comparison(df_results, save_dir='./evaluation_results'):
    """측면별 F1 스코어 비교 그래프"""
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 측면별 전체 F1 스코어
    ax1 = axes[0]
    x = np.arange(len(df_results))
    width = 0.35

    ax1.bar(x - width / 2, df_results['f1_macro'], width, label='F1 (Macro)', alpha=0.8)
    ax1.bar(x + width / 2, df_results['f1_weighted'], width, label='F1 (Weighted)', alpha=0.8)

    ax1.set_xlabel('측면', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('측면별 F1 스코어 비교', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_results['aspect'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # 2. 감정 클래스별 F1 스코어
    ax2 = axes[1]
    x = np.arange(len(df_results))
    width = 0.25

    ax2.bar(x - 1.5 * width, df_results['f1_none'], width, label='해당없음', alpha=0.8, color='lightgray')
    ax2.bar(x - 0.5 * width, df_results['f1_positive'], width, label='긍정', alpha=0.8, color='green')
    ax2.bar(x + 0.5 * width, df_results['f1_negative'], width, label='부정', alpha=0.8, color='red')
    ax2.bar(x + 1.5 * width, df_results['f1_neutral'], width, label='중립', alpha=0.8, color='gray')

    ax2.set_xlabel('측면', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('측면별 감정 클래스 F1 스코어', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_results['aspect'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'f1_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ F1 비교 그래프 저장: {save_path}")
    plt.close()


# ==========================================
# 평가 함수에 신뢰도 정보 추가
# ==========================================
def evaluate_sentiment_per_aspect(model, dataloader, device, thresholds):
    """측면별 감정 분류 평가 + 신뢰도 및 misclassified 샘플 저장"""
    model.eval()

    aspect_sentiment_preds = {i: [] for i in range(12)}
    aspect_sentiment_labels = {i: [] for i in range(12)}

    # 🆕 신뢰도 및 misclassified 샘플 저장용
    aspect_confidences = {i: [] for i in range(12)}  # 예측 신뢰도
    misclassified_samples = []  # (텍스트, 측면, 실제라벨, 예측라벨, 신뢰도)
    all_samples_info = []  # 모든 샘플 정보

    all_aspect_preds = []
    all_aspect_labels = []

    sample_idx = 0  # 전체 샘플 인덱스

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            sentiment_logits = outputs['sentiment_logits']
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)

            # 🆕 최대 신뢰도 추출
            sentiment_confidence = torch.max(sentiment_probs, dim=-1)[0]  # (batch, 12)

            aspect_detected = torch.zeros_like(aspect_probs, dtype=torch.bool)
            for i in range(12):
                aspect_detected[:, i] = aspect_probs[:, i] > thresholds[i]

            aspect_labels = (labels != 0)
            all_aspect_preds.append(aspect_detected.cpu())
            all_aspect_labels.append(aspect_labels.cpu())

            # 배치 내 각 샘플 처리
            for sample_in_batch in range(labels.size(0)):
                current_text = dataloader.dataset.texts[sample_idx]

                for i in range(12):
                    true_sentiment = labels[sample_in_batch, i].item()

                    # 예측 라벨 생성
                    if aspect_detected[sample_in_batch, i]:
                        pred_sentiment = sentiment_preds[sample_in_batch, i].item() + 1
                        confidence = sentiment_confidence[sample_in_batch, i].item()
                    else:
                        pred_sentiment = 0
                        confidence = 1.0 - aspect_probs[sample_in_batch, i].item()

                    aspect_sentiment_labels[i].append(true_sentiment)
                    aspect_sentiment_preds[i].append(pred_sentiment)
                    aspect_confidences[i].append(confidence)

                    # 🆕 모든 샘플 정보 저장
                    sample_info = {
                        'sample_idx': sample_idx,
                        'text': current_text,
                        'aspect': ASPECT_NAMES[i],
                        'aspect_idx': i,
                        'true_label': true_sentiment,
                        'pred_label': pred_sentiment,
                        'confidence': confidence,
                        'is_misclassified': (true_sentiment != pred_sentiment)
                    }
                    all_samples_info.append(sample_info)

                    # 🆕 Misclassified 샘플 저장
                    if true_sentiment != pred_sentiment:
                        misclassified_samples.append(sample_info)

                sample_idx += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"   진행: {batch_idx + 1}/{len(dataloader)} 배치")

    return (aspect_sentiment_preds, aspect_sentiment_labels,
            all_aspect_preds, all_aspect_labels,
            aspect_confidences, misclassified_samples, all_samples_info)


# ==========================================
# 🆕 Misclassified 샘플 분석 함수
# ==========================================
def save_misclassified_samples(misclassified_samples, save_dir='./evaluation_results'):
    """오분류 샘플을 CSV로 저장"""
    os.makedirs(save_dir, exist_ok=True)

    df_misc = pd.DataFrame(misclassified_samples)

    # 신뢰도 낮은 순으로 정렬
    df_misc = df_misc.sort_values('confidence', ascending=True)

    save_path = os.path.join(save_dir, 'misclassified_samples.csv')
    df_misc.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n📋 오분류 샘플 분석:")
    print(f"   총 오분류: {len(df_misc)}개")
    print(f"   저장 경로: {save_path}")

    # 측면별 오분류 통계
    print(f"\n   측면별 오분류 개수:")
    for aspect in ASPECT_NAMES:
        count = len(df_misc[df_misc['aspect'] == aspect])
        print(f"      {aspect}: {count}개")

    return df_misc


# ==========================================
# 🆕 신뢰도 분석 함수
# ==========================================
def analyze_confidence(aspect_confidences, aspect_sentiment_labels,
                       aspect_sentiment_preds, save_dir='./evaluation_results'):
    """신뢰도 분석 및 저장"""
    os.makedirs(save_dir, exist_ok=True)

    confidence_analysis = []

    for i, aspect_name in enumerate(ASPECT_NAMES):
        if len(aspect_confidences[i]) == 0:
            continue

        confs = np.array(aspect_confidences[i])
        labels = np.array(aspect_sentiment_labels[i])
        preds = np.array(aspect_sentiment_preds[i])

        # 정분류/오분류별 신뢰도
        correct_mask = (labels == preds)
        correct_confs = confs[correct_mask]
        incorrect_confs = confs[~correct_mask]

        analysis = {
            'aspect': aspect_name,
            'total_samples': len(confs),
            'avg_confidence': confs.mean(),
            'std_confidence': confs.std(),
            'min_confidence': confs.min(),
            'max_confidence': confs.max(),
            'correct_avg_conf': correct_confs.mean() if len(correct_confs) > 0 else 0,
            'incorrect_avg_conf': incorrect_confs.mean() if len(incorrect_confs) > 0 else 0,
            'num_correct': len(correct_confs),
            'num_incorrect': len(incorrect_confs),
        }
        confidence_analysis.append(analysis)

    df_conf = pd.DataFrame(confidence_analysis)
    save_path = os.path.join(save_dir, 'confidence_analysis.csv')
    df_conf.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n📊 신뢰도 분석:")
    print(f"   저장 경로: {save_path}")
    print(f"\n   측면별 평균 신뢰도:")
    for _, row in df_conf.iterrows():
        print(f"      {row['aspect']}: {row['avg_confidence']:.4f} "
              f"(정분류: {row['correct_avg_conf']:.4f}, "
              f"오분류: {row['incorrect_avg_conf']:.4f})")

    return df_conf


# ==========================================
# 🆕 Low Confidence 샘플 추출
# ==========================================
def extract_low_confidence_samples(all_samples_info, threshold=0.5,
                                   save_dir='./evaluation_results'):
    """낮은 신뢰도 샘플 추출"""
    os.makedirs(save_dir, exist_ok=True)

    df_all = pd.DataFrame(all_samples_info)
    df_low_conf = df_all[df_all['confidence'] < threshold].copy()
    df_low_conf = df_low_conf.sort_values('confidence', ascending=True)

    save_path = os.path.join(save_dir, f'low_confidence_samples_th{threshold}.csv')
    df_low_conf.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n⚠️ 낮은 신뢰도 샘플 (threshold < {threshold}):")
    print(f"   개수: {len(df_low_conf)}개")
    print(f"   저장 경로: {save_path}")

    return df_low_conf

# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    print("=" * 80)
    print("🎭 감정 분류 모델 평가 (STAGE 3)")
    print("=" * 80)

    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CPU 사용 중")

    # # 1. 테스트 데이터 로드
    # print(f"\n📂 테스트 데이터 로딩 중... ({TEST_DATA_PATH})")
    # try:
    #     df_test = pd.read_csv(TEST_DATA_PATH, dtype={'label': str})
    #     print(f"✅ 테스트 데이터: {len(df_test)}개")
    # except FileNotFoundError:
    #     print(f"❌ 파일을 찾을 수 없습니다: {TEST_DATA_PATH}")
    #     exit()
    #
    # test_texts = df_test['Original_Review'].tolist()
    # test_labels = df_test['label'].tolist()
    # 1. 🆕 여러 테스트 데이터를 합쳐서 로드
    print(f"\n📂 테스트 데이터 로딩 중...")
    df_test_list = []
    for path in TEST_DATA_PATH:
        try:
            df = pd.read_csv(path, dtype={'label': str})
            df_test_list.append(df)
            print(f"   ✅ {path}: {len(df)}개")
        except FileNotFoundError:
            print(f"   ⚠️ 파일을 찾을 수 없습니다: {path}, 스킵합니다.")
            continue

    if len(df_test_list) == 0:
        print("❌ 로드된 테스트 데이터가 없습니다.")
        exit()

    # 🆕 데이터 합치기
    df_test = pd.concat(df_test_list, ignore_index=True)
    print(f"\n✅ 전체 테스트 데이터: {len(df_test)}개 (총 {len(df_test_list)}개 파일 병합)")

    test_texts = df_test['Original_Review'].tolist()
    test_labels = df_test['label'].tolist()

    # 2. 토크나이저 로드
    print(f"\n🔧 토크나이저 로딩 중...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # 3. 데이터셋 준비
    test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 모델 로드
    print(f"\n🤖 모델 로딩 중... ({MODEL_PATH})")

    if not os.path.exists(f"{MODEL_PATH}/model_state_dict.pt"):
        print(f"❌ 모델을 찾을 수 없습니다: {MODEL_PATH}/model_state_dict.pt")
        exit()

    # pos_weight 로드
    try:
        global_pos_weights = torch.load(f"{MODEL_PATH}/global_pos_weights.pt", weights_only=True)
    except:
        global_pos_weights = torch.ones(12) * 10.0

    # sentiment_class_weights 로드
    try:
        sentiment_class_weights = torch.load(f"{MODEL_PATH}/sentiment_class_weights.pt", weights_only=True)
    except:
        sentiment_class_weights = None

    model = DualHeadBert(
        MODEL_NAME,
        aspect_only=False,
        global_pos_weights=global_pos_weights,
        sentiment_class_weights=sentiment_class_weights
    ).to(device)

    model.load_state_dict(torch.load(f"{MODEL_PATH}/model_state_dict.pt", weights_only=False),strict=False)
    print("✅ 모델 로드 완료!")

    # 5. 평가 실행 (반환값 추가)
    (aspect_sentiment_preds, aspect_sentiment_labels,
     all_aspect_preds, all_aspect_labels,
     aspect_confidences, misclassified_samples, all_samples_info) = \
        evaluate_sentiment_per_aspect(model, test_loader, device, EVAL_THRESHOLDS)

    # 6. 기존 결과 출력
    print_aspect_detection_performance(all_aspect_preds, all_aspect_labels, EVAL_THRESHOLDS)
    df_results = print_sentiment_results(aspect_sentiment_preds, aspect_sentiment_labels)

    # 7. 🆕 신뢰도 및 오분류 분석
    df_conf = analyze_confidence(aspect_confidences, aspect_sentiment_labels,
                                 aspect_sentiment_preds)
    df_misc = save_misclassified_samples(misclassified_samples)
    df_low_conf = extract_low_confidence_samples(all_samples_info, threshold=0.5)

    # 8. 시각화
    plot_confusion_matrices(aspect_sentiment_preds, aspect_sentiment_labels)
    plot_f1_comparison(df_results)

    # 9. 결과 저장
    print("\n💾 결과 저장 중...")
    df_results.to_csv('./evaluation_results/sentiment_evaluation_summary.csv',
                      index=False, encoding='utf-8-sig')
    print(f"   - 혼동 행렬: ./evaluation_results/confusion_matrices_per_aspect.png")
    print(f"   - F1 비교: ./evaluation_results/f1_comparison.png")

