import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from collections import Counter
import re

# ==========================================
# 0. 설정
# ==========================================
MODEL_NAME = "klue/bert-base"
DATA_PATH = "multi_aspect_new.csv"
SAVED_MODEL_PATH = "./final_cafe_model"
MAX_LEN = 128
BATCH_SIZE = 32
SEED = 42

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

        label_raw = str(self.labels[item])
        label_clean = label_raw.strip().replace('"', '').replace("'", "")
        label_str = label_clean.zfill(12)

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
# 🔧 DualHeadBert (Phase 학습 버전과 동일)
# ==========================================
class DualHeadBert(nn.Module):
    def __init__(self, model_name):
        super(DualHeadBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)

        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 12)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 12 * 3)

        self.aspect_head.bias.data.fill_(-4.0)

        # Phase 관련 속성 (테스트 시에는 불필요하지만 호환성 유지)
        self.current_phase = None

    def set_phase(self, phase_config):
        """Phase 전환 (테스트에서는 사용 안 함)"""
        self.current_phase = phase_config
        for param in self.sentiment_head.parameters():
            param.requires_grad = not phase_config.freeze_sentiment

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)

        aspect_logits = self.aspect_head(output)
        sentiment_logits = self.sentiment_head(output).view(-1, 12, 3)

        loss = None
        if labels is not None:
            aspect_labels = (labels != 0).float()

            num_pos = aspect_labels.sum()
            num_neg = aspect_labels.numel() - num_pos
            pos_weight_val = (num_neg / (num_pos + 1e-5)).item()
            final_pos_weight = torch.tensor([min(max(pos_weight_val, 10.0), 30.0)] * 12).to(labels.device)

            aspect_loss_fct = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
            aspect_loss = aspect_loss_fct(aspect_logits, aspect_labels)

            if class_weights is not None:
                sentiment_weights = class_weights.to(labels.device)
            else:
                sentiment_weights = None

            sentiment_loss_fct = FocalLoss(gamma=2.0, weight=sentiment_weights)
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

            # Phase 정보가 있으면 그에 맞게, 없으면 기본값
            if self.current_phase:
                loss = (self.current_phase.aspect_weight * aspect_loss +
                        self.current_phase.sentiment_weight * sentiment_loss)
            else:
                loss = aspect_loss + 1.0 * sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }


# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    # 한글 폰트 설정
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumBarunGothic'

    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 70)
    print("🔍 모델 평가 및 신뢰도 분석")
    print("=" * 70)

    # 데이터 로드
    print(f"📂 데이터 로딩 중...")

    df = pd.read_csv(DATA_PATH, dtype={'label': str})
    temp_texts, test_texts, temp_labels, test_labels = train_test_split(
        df['Original_Review'].tolist(),
        df['label'].tolist(),
        test_size=500,
        random_state=SEED,
        shuffle=True
    )

    # 고난도 데이터 추가
    try:
        df_hard = pd.read_csv("test_hard.csv", dtype={'label': str})
        test_texts.extend(df_hard['Original_Review'].tolist())
        test_labels.extend(df_hard['label'].tolist())
        print(f"총 테스트 데이터: {len(test_texts)}개 (기본 500 + 추가 {len(df_hard)})")
    except FileNotFoundError:
        print(f"총 테스트 데이터: {len(test_texts)}개 (기본만)")

    # 토크나이저 및 데이터셋
    tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_PATH)
    test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💾 모델 로드 중... ({SAVED_MODEL_PATH})")

    model = DualHeadBert(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(f"{SAVED_MODEL_PATH}/model_state_dict.pt",
                                     map_location=device))
    model.eval()
    print("✅ 모델 로드 완료!")

    # 테스트셋 평가
    print("\n📊 테스트 데이터셋 평가 중...")
    all_preds = []
    all_labels = []

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            # 측면 감지
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            aspect_detected = (aspect_probs > 0.5).long()

            # 감성 분류 (0,1,2 -> 1,2,3 복원)
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=2)

            # 최종 예측
            preds = torch.where(aspect_detected == 1, sentiment_preds + 1,
                                torch.zeros_like(sentiment_preds))

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 측면별 정확도
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
    print(f"전체 정확도: {overall_accuracy:.2f}%")

    # ==========================================
    # 신뢰도 분석
    # ==========================================
    print("\n" + "=" * 70)
    print("🔍 상세 성능 분석")
    print("=" * 70)

    print("\n📊 1. 신뢰도 분석")
    print("-" * 70)

    all_confidences = []
    low_confidence_cases = []

    test_loader_single = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader_single):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            aspect_probs = torch.sigmoid(outputs['aspect_logits']).squeeze()
            sentiment_logits = outputs['sentiment_logits'].squeeze()
            sentiment_probs = torch.softmax(sentiment_logits, dim=1)
            sentiment_preds = torch.argmax(sentiment_logits, dim=1)

            confidences = torch.zeros(12)
            final_preds = []

            for i in range(12):
                if aspect_probs[i] > 0.5:
                    confidences[i] = aspect_probs[i] * sentiment_probs[i, sentiment_preds[i]]
                    final_preds.append(sentiment_preds[i].item() + 1)
                else:
                    confidences[i] = (1.0 - aspect_probs[i])
                    final_preds.append(0)

            avg_conf = confidences.mean().item()
            all_confidences.append(avg_conf)

            if avg_conf < 0.7:
                low_confidence_cases.append({
                    'idx': idx,
                    'text': test_texts[idx],
                    'confidence': avg_conf,
                    'pred': final_preds,
                    'true': labels.squeeze().cpu().tolist()
                })

    all_confidences = np.array(all_confidences)
    print(f"평균 신뢰도: {all_confidences.mean():.2%}")
    print(f"중앙값 신뢰도: {np.median(all_confidences):.2%}")
    print(f"최소 신뢰도: {all_confidences.min():.2%}")
    print(f"최대 신뢰도: {all_confidences.max():.2%}")
    print(f"\n낮은 신뢰도(<70%) 케이스: {len(low_confidence_cases)}개 "
          f"({len(low_confidence_cases) / len(test_texts) * 100:.1f}%)")

    if low_confidence_cases:
        print("\n⚠️ 신뢰도가 낮은 TOP 10 케이스:")
        print("-" * 70)
        low_confidence_cases.sort(key=lambda x: x['confidence'])
        for i, case in enumerate(low_confidence_cases[:10], 1):
            print(f"\n[{i}] 신뢰도: {case['confidence']:.1%}")
            print(f"    리뷰: {case['text'][:80]}...")

            print(f"    차이점:")
            for j in range(12):
                if case['pred'][j] != case['true'][j]:
                    print(f"      - {ASPECT_NAMES[j]}: "
                          f"정답({SENTIMENT_NAMES[case['true'][j]]}) vs "
                          f"예측({SENTIMENT_NAMES[case['pred'][j]]})")

    # 측면별 F1-Score
    print("\n" + "=" * 70)
    print("📊 2. 측면별 상세 평가 (Precision, Recall, F1-Score)")
    print("=" * 70)

    for i, aspect_name in enumerate(ASPECT_NAMES):
        y_true = all_labels[:, i].numpy()
        y_pred = all_preds[:, i].numpy()

        print(f"\n[ {aspect_name} ]")

        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [SENTIMENT_NAMES[l] for l in present_labels]
        try:
            print(classification_report(
                y_true, y_pred,
                labels=present_labels,
                target_names=target_names,
                zero_division=0,
                digits=3
            ))
        except Exception as e:
            print(f"  ⚠️ 보고서 생성 실패: {e}")

    # 혼동 행렬 시각화
    print("\n" + "=" * 70)
    print("📊 3. 모든 측면 혼동 행렬 시각화 중...")
    print("=" * 70)

    import math

    cols = 3
    rows = math.ceil(len(ASPECT_NAMES) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 20))
    fig.suptitle('모든 측면별 혼동 행렬 (Confusion Matrix)', fontsize=25, y=1.02)

    labels_cm = ["해당없음", "긍정", "부정", "중립"]

    for i, aspect_name in enumerate(ASPECT_NAMES):
        r, c = i // cols, i % cols
        y_true = all_labels[:, i].numpy()
        y_pred = all_preds[:, i].numpy()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r, c],
                    xticklabels=labels_cm, yticklabels=labels_cm, cbar=False)

        axes[r, c].set_title(f'[{aspect_name}]', fontsize=15)
        axes[r, c].set_xlabel('예측값')
        axes[r, c].set_ylabel('실제값')

    for j in range(i + 1, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.savefig(f"{SAVED_MODEL_PATH}/all_aspects_confusion_matrix.png", dpi=300)
    print(f"✅ 혼동 행렬 저장: {SAVED_MODEL_PATH}/all_aspects_confusion_matrix.png")

    # 불일치 케이스 분석
    print("\n" + "=" * 70)
    print("📊 4. 가장 많이 틀린 케이스 TOP 10")
    print("=" * 70)

    disagreements = []
    for idx in range(len(test_texts)):
        true_label = all_labels[idx]
        pred_label = all_preds[idx]

        diff_count = (pred_label != true_label).sum().item()

        if diff_count > 0:
            disagreements.append({
                'idx': idx,
                'text': test_texts[idx],
                'true': true_label.tolist(),
                'pred': pred_label.tolist(),
                'diff_count': diff_count
            })

    disagreements.sort(key=lambda x: x['diff_count'], reverse=True)

    print(f"\n전체 불일치 케이스: {len(disagreements)}개 "
          f"({len(disagreements) / len(test_texts) * 100:.1f}%)")
    print("\n가장 많이 틀린 TOP 10:")
    print("-" * 70)

    for i, case in enumerate(disagreements[:10], 1):
        print(f"\n[{i}] 불일치 개수: {case['diff_count']}/12")
        print(f"    리뷰: {case['text'][:80]}...")

        print("    차이점:")
        for j in range(12):
            if case['true'][j] != case['pred'][j]:
                print(f"      - {ASPECT_NAMES[j]}: "
                      f"정답({SENTIMENT_NAMES[case['true'][j]]}) vs "
                      f"예측({SENTIMENT_NAMES[case['pred'][j]]})")

    # 측면 감지 통계
    print("\n" + "=" * 70)
    print("📊 5. 측면 감지 통계")
    print("=" * 70)

    detected_true = (all_labels != 0).sum(dim=1).float()
    detected_pred = (all_preds != 0).sum(dim=1).float()

    print(f"평균 실제 측면 수: {detected_true.mean():.2f}")
    print(f"평균 예측 측면 수: {detected_pred.mean():.2f}")
    print(f"측면 수 차이: {abs(detected_pred.mean() - detected_true.mean()):.2f}")

    over_detection = []
    under_detection = []

    for idx in range(len(test_texts)):
        true_count = detected_true[idx].item()
        pred_count = detected_pred[idx].item()
        diff = pred_count - true_count

        if diff >= 3:
            over_detection.append({'text': test_texts[idx], 'diff': diff})
        elif diff <= -3:
            under_detection.append({'text': test_texts[idx], 'diff': diff})

    if over_detection:
        print(f"\n⚠️ 측면을 과도하게 많이 감지한 케이스: {len(over_detection)}개")
        for case in over_detection[:3]:
            print(f"   {case['text'][:60]}... (차이: +{case['diff']}개)")

    if under_detection:
        print(f"\n⚠️ 측면을 너무 적게 감지한 케이스: {len(under_detection)}개")
        for case in under_detection[:3]:
            print(f"   {case['text'][:60]}... (차이: {case['diff']}개)")

    # F1-Score 시각화
    print("\n" + "=" * 70)
    print("📊 6. 시각화 리포트 생성")
    print("=" * 70)

    aspect_f1_scores = []
    for i in range(12):
        report = classification_report(all_labels[:, i], all_preds[:, i],
                                       output_dict=True, zero_division=0)
        aspect_f1_scores.append(report['weighted avg']['f1-score'])

    plt.figure(figsize=(12, 8))
    perf_df = pd.DataFrame({'Aspect': ASPECT_NAMES, 'F1': aspect_f1_scores}).sort_values('F1', ascending=False)

    sns.barplot(data=perf_df, x='F1', y='Aspect', palette='coolwarm')

    for i, v in enumerate(perf_df['F1']):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.title('측면별 모델 성능 (F1-Score)', fontsize=15)
    plt.xlabel('F1-Score (0.0 ~ 1.0)')
    plt.ylabel('평가 항목 (Aspect)')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"{SAVED_MODEL_PATH}/aspect_f1_report.png")
    print(f"✅ F1-Score 그래프 저장: aspect_f1_report.png")

    # 에러 분석 CSV 저장
    error_analysis_df = pd.DataFrame(disagreements)
    error_analysis_df.to_csv(f"{SAVED_MODEL_PATH}/error_analysis.csv",
                             index=False, encoding='utf-8-sig')
    print(f"✅ 에러 리스트 저장: error_analysis.csv")

    # 감성별 F1-Score 히트맵
    print("\n" + "=" * 70)
    print("📊 7. 감성별 상세 성능(F1-Score) 분석 중...")
    print("=" * 70)

    sentiment_labels = ['1', '2', '3']
    sentiment_names = ['긍정', '부정', '중립']
    f1_matrix = []

    for i in range(12):
        report = classification_report(all_labels[:, i], all_preds[:, i],
                                       output_dict=True, zero_division=0)
        scores = []
        for s_code in sentiment_labels:
            score = report.get(s_code, {}).get('f1-score', 0.0)
            scores.append(score)
        f1_matrix.append(scores)

    df_f1 = pd.DataFrame(f1_matrix, index=ASPECT_NAMES, columns=sentiment_names)

    plt.figure(figsize=(10, 10))
    sns.heatmap(df_f1, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)

    plt.title('측면별/감성별 상세 성능 (F1-Score)', fontsize=18, pad=20)
    plt.ylabel('평가 항목 (Aspect)', fontsize=12)
    plt.xlabel('감성 (Sentiment)', fontsize=12)
    plt.tight_layout()

    plt.savefig(f"{SAVED_MODEL_PATH}/sentiment_detailed_f1.png", dpi=300)
    print(f"✅ 상세 히트맵 저장: {SAVED_MODEL_PATH}/sentiment_detailed_f1.png")

    # 데이터 밸런스 경고
    print("\n⚠️ 성능 최적화 제언:")
    negative_recalls = []
    for i in range(12):
        report = classification_report(all_labels[:, i], all_preds[:, i],
                                       output_dict=True, zero_division=0)
        if '2' in report:
            negative_recalls.append(report['2']['recall'])

    avg_neg_recall = np.mean(negative_recalls) if negative_recalls else 0
    if avg_neg_recall < 0.5:
        print(f"👉 [위험] 부정(Negative) 재현율이 {avg_neg_recall:.2%}로 매우 낮습니다.")
        print(f"   학습 데이터에 '부정' 사례를 더 추가해야 실전에서 악플을 잡아낼 수 있습니다.")

    # 중립 오답 분석
    print("\n" + "=" * 70)
    print("🕵️‍♂️ '중립(Neutral)' 오답 집중 추적")
    print("=" * 70)

    neutral_errors = []

    for idx in range(len(test_texts)):
        for aspect_idx in range(12):
            true_val = all_labels[idx, aspect_idx].item()
            pred_val = all_preds[idx, aspect_idx].item()

            if true_val == 3 and pred_val != 3:
                neutral_errors.append({
                    'Type': '놓친 중립(Missed)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': '중립',
                    'Pred': SENTIMENT_NAMES[pred_val]
                })

            elif pred_val == 3 and true_val != 3:
                neutral_errors.append({
                    'Type': '잘못된 중립(False Neutral)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': SENTIMENT_NAMES[true_val],
                    'Pred': '중립'
                })

    if neutral_errors:
        print(f"⚠️ 총 {len(neutral_errors)}개의 중립 관련 오류가 발견되었습니다.\n")

        for i, err in enumerate(neutral_errors[:20], 1):
            print(f"[{i}] {err['Type']}")
            print(f"   측면: {err['Aspect']}")
            print(f"   문장: {err['Text']}")
            print(f"   결과: 정답({err['True']}) vs 예측({err['Pred']})")
            print("-" * 50)

        err_df = pd.DataFrame(neutral_errors)
        err_df.to_csv(f"{SAVED_MODEL_PATH}/neutral_errors.csv",
                      index=False, encoding='utf-8-sig')
        print(f"\n✅ 중립 오답 리스트 저장: {SAVED_MODEL_PATH}/neutral_errors.csv")
    else:
        print("🎉 와우! 중립 감성에 대한 오류가 하나도 없습니다.")

    # 부정 오답 분석
    print("\n" + "=" * 70)
    print("🕵️‍♂️ '부정(Negative)' 오답 집중 추적")
    print("=" * 70)

    negative_errors = []

    for idx in range(len(test_texts)):
        for aspect_idx in range(12):
            true_val = all_labels[idx, aspect_idx].item()
            pred_val = all_preds[idx, aspect_idx].item()

            if true_val == 2 and pred_val != 2:
                negative_errors.append({
                    'Type': '놓친 부정(Missed)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': '부정',
                    'Pred': SENTIMENT_NAMES[pred_val]
                })

            elif pred_val == 2 and true_val != 2:
                negative_errors.append({
                    'Type': '잘못된 부정(False Negative)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': SENTIMENT_NAMES[true_val],
                    'Pred': '부정'
                })

    if negative_errors:
        print(f"⚠️ 총 {len(negative_errors)}개의 부정 관련 오류가 발견되었습니다.\n")

        # 보기 좋게 상위 20개만 출력
        for i, err in enumerate(negative_errors[:20], 1):
            print(f"[{i}] {err['Type']}")
            print(f"   측면: {err['Aspect']}")
            print(f"   문장: {err['Text']}")
            print(f"   결과: 정답({err['True']}) vs 예측({err['Pred']})")
            print("-" * 50)

        # CSV로 전체 저장 (엑셀에서 확인용)
        err_df = pd.DataFrame(negative_errors)
        err_df.to_csv(f"{SAVED_MODEL_PATH}/negative_errors.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ 전체 부정 오답 리스트 저장 완료: {SAVED_MODEL_PATH}/negative_errors.csv")

    else:
        print("🎉 와우! 부정 감성에 대한 오류가 하나도 없습니다.")

# ==========================================
# 🆕 부정/중립 오답의 실제 원인 분석
# ==========================================
print("\n" + "=" * 70)
print("🔬 부정/중립 오답 키워드 분석")
print("=" * 70)

from collections import Counter
import re

# 부정/중립 오답 텍스트만 추출
negative_missed_texts = [err['Text'] for err in negative_errors if err['Type'] == '놓친 부정(Missed)']
neutral_missed_texts = [err['Text'] for err in neutral_errors if err['Type'] == '놓친 중립(Missed)']


# 한국어 형태소 추출 (간단 버전)
def extract_keywords(text):
    # 공백, 특수문자 기준으로 단어 분리
    words = re.findall(r'[가-힣]+', text)
    # 2글자 이상 단어만
    return [w for w in words if len(w) >= 2]


# 부정 오답에서 자주 나오는 단어
if negative_missed_texts:
    print("\n[ 놓친 부정 리뷰에서 자주 나온 표현 TOP 20 ]")
    all_words = []
    for text in negative_missed_texts:
        all_words.extend(extract_keywords(text))

    word_freq = Counter(all_words)
    for word, count in word_freq.most_common(20):
        print(f"  '{word}' : {count}번")

# 중립 오답에서 자주 나오는 단어
if neutral_missed_texts:
    print("\n[ 놓친 중립 리뷰에서 자주 나온 표현 TOP 20 ]")
    all_words = []
    for text in neutral_missed_texts:
        all_words.extend(extract_keywords(text))

    word_freq = Counter(all_words)
    for word, count in word_freq.most_common(20):
        print(f"  '{word}' : {count}번")

# 🔥 핵심: 부정/중립 특유의 표현 찾기
print("\n" + "=" * 70)
print("💡 학습 데이터에 추가해야 할 표현 후보")
print("=" * 70)

# 부정 키워드인데 자주 놓친 것
negative_keywords = ['별로', '실망', '아쉽', '그냥', '비싸', '불친절', '짜증']
for keyword in negative_keywords:
    missed_count = sum(1 for text in negative_missed_texts if keyword in text)
    if missed_count > 0:
        print(f"⚠️ '{keyword}' 포함 부정 리뷰를 {missed_count}번 놓침")

# 중립 키워드인데 자주 놓친 것
neutral_keywords = ['괜찮', '그냥', '평범', '무난', '보통', '나쁘지않', '애매']
for keyword in neutral_keywords:
    missed_count = sum(1 for text in neutral_missed_texts if keyword in text)
    if missed_count > 0:
        print(f"⚠️ '{keyword}' 포함 중립 리뷰를 {missed_count}번 놓침")

# ==========================================
# 🆕 대표 오답 케이스 상세 분석
# ==========================================
print("\n" + "=" * 70)
print("🔎 대표 오답 케이스 5개 상세 분석")
print("=" * 70)

# 부정을 놓친 케이스 5개
if negative_missed_texts:
    print("\n[ 부정을 놓친 케이스 ]")
    for i, case in enumerate(negative_errors[:5], 1):
        if case['Type'] == '놓친 부정(Missed)':
            print(f"\n{i}. {case['Text']}")
            print(f"   측면: {case['Aspect']}")
            print(f"   실제: {case['True']} → 예측: {case['Pred']}")

            # 이 리뷰를 모델에 다시 넣어서 신뢰도 확인
            encoding = tokenizer.encode_plus(
                case['Text'],
                add_special_tokens=True,
                max_length=MAX_LEN,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            with torch.no_grad():
                outputs = model(
                    encoding['input_ids'].to(device),
                    encoding['attention_mask'].to(device)
                )

                aspect_idx = ASPECT_NAMES.index(case['Aspect'])
                aspect_prob = outputs['aspect_probs'][0, aspect_idx].item()
                logits = outputs['logits'][0, aspect_idx]
                sent_probs = torch.softmax(logits, dim=0)

                print(f"   📊 모델 판단:")
                print(f"      측면 감지: {aspect_prob:.1%}")
                print(f"      해당없음: {sent_probs[0]:.1%}")
                print(f"      긍정: {sent_probs[1]:.1%}")
                print(f"      부정: {sent_probs[2]:.1%}")  # ← 이게 낮으면 학습 부족
                print(f"      중립: {sent_probs[3]:.1%}")