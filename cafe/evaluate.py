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

# ==========================================
# 0. 설정
# ==========================================
MODEL_NAME = "klue/bert-base"
DATA_PATH = "final_data.csv"
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
# 2. 모델 정의
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
# 메인 실행
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("🔍 모델 평가 및 신뢰도 분석")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n📂 데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH, dtype={'label': str})
    print(f"총 데이터 개수: {len(df)}개")

    # 2. 데이터 분할 (학습과 동일하게)
    temp_texts, test_texts, temp_labels, test_labels = train_test_split(
        df['Original_Review'].tolist(),
        df['label'].tolist(),
        test_size=500,
        random_state=SEED,
        shuffle=True
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=500,
        random_state=SEED,
        shuffle=True
    )

    print(f"테스트 데이터: {len(test_texts)}개")

    # 3. 토크나이저 및 데이터셋
    tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_PATH)
    test_dataset = CafeAspectDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    # 4. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💾 모델 로드 중... ({SAVED_MODEL_PATH})")

    model = MultiOutputBert(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(f"{SAVED_MODEL_PATH}/model_state_dict.pt",
                                     map_location=device))
    model.eval()
    print("✅ 모델 로드 완료!")

    # 5. 테스트셋 평가
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
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=2)

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

    # -----------------------------------------
    # 1. 신뢰도 분석
    # -----------------------------------------
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
            logits = outputs['logits']

            # 확률 계산
            probs = torch.softmax(logits, dim=2).squeeze()
            preds = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1).values

            # 평균 신뢰도
            avg_conf = confidences.mean().item()
            all_confidences.append(avg_conf)

            # 낮은 신뢰도 케이스 저장 (0.7 미만)
            if avg_conf < 0.7:
                low_confidence_cases.append({
                    'idx': idx,
                    'text': test_texts[idx],
                    'confidence': avg_conf,
                    'pred': preds.cpu().tolist(),
                    'true': labels.squeeze().cpu().tolist()
                })

    # 신뢰도 통계
    all_confidences = np.array(all_confidences)
    print(f"평균 신뢰도: {all_confidences.mean():.2%}")
    print(f"중앙값 신뢰도: {np.median(all_confidences):.2%}")
    print(f"최소 신뢰도: {all_confidences.min():.2%}")
    print(f"최대 신뢰도: {all_confidences.max():.2%}")
    print(
        f"\n낮은 신뢰도(<70%) 케이스: {len(low_confidence_cases)}개 ({len(low_confidence_cases) / len(test_texts) * 100:.1f}%)")

    # 낮은 신뢰도 케이스 상위 10개 출력
    if low_confidence_cases:
        print("\n⚠️ 신뢰도가 낮은 TOP 10 케이스:")
        print("-" * 70)
        low_confidence_cases.sort(key=lambda x: x['confidence'])
        for i, case in enumerate(low_confidence_cases[:10], 1):
            print(f"\n[{i}] 신뢰도: {case['confidence']:.1%}")
            print(f"    리뷰: {case['text'][:80]}...")

            # 예측 vs 정답 비교
            print(f"    차이점:")
            for j in range(12):
                if case['pred'][j] != case['true'][j]:
                    print(
                        f"      - {ASPECT_NAMES[j]}: 정답({SENTIMENT_NAMES[case['true'][j]]}) vs 예측({SENTIMENT_NAMES[case['pred'][j]]})")

    # -----------------------------------------
    # 2. 측면별 F1-Score
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("📊 2. 측면별 상세 평가 (Precision, Recall, F1-Score)")
    print("=" * 70)

    for i, aspect_name in enumerate(ASPECT_NAMES):
        y_true = all_labels[:, i].numpy()
        y_pred = all_preds[:, i].numpy()

        print(f"\n[ {aspect_name} ]")
        try:
            print(classification_report(
                y_true, y_pred,
                target_names=["해당없음", "긍정", "부정", "중립"],
                zero_division=0,
                digits=3
            ))
        except Exception as e:
            print(f"  ⚠️ 보고서 생성 실패: {e}")

    # -----------------------------------------
    # 3. 혼동 행렬 (서비스 측면)
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("📊 3. 혼동 행렬 분석 (서비스 측면)")
    print("=" * 70)

    service_idx = 8  # 서비스
    y_true_service = all_labels[:, service_idx].numpy()
    y_pred_service = all_preds[:, service_idx].numpy()

    cm = confusion_matrix(y_true_service, y_pred_service)
    print("\n서비스 측면 혼동 행렬:")
    print("          예측→")
    print("실제↓   해당없음  긍정   부정   중립")
    labels_cm = ["해당없음", "긍정", "부정", "중립"]
    for i, label in enumerate(labels_cm):
        print(f"{label:6s}  ", end="")
        for j in range(4):
            if i < cm.shape[0] and j < cm.shape[1]:
                print(f"{cm[i][j]:6d} ", end="")
            else:
                print(f"{'0':6s} ", end="")
        print()

    # 부정을 긍정으로 잘못 예측한 경우 찾기
    wrong_negative_to_positive = []
    for idx in range(len(test_texts)):
        true_val = all_labels[idx, service_idx].item()
        pred_val = all_preds[idx, service_idx].item()

        if true_val == 2 and pred_val == 1:  # 실제 부정인데 긍정으로 예측
            wrong_negative_to_positive.append({
                'text': test_texts[idx],
                'true': true_val,
                'pred': pred_val
            })

    if wrong_negative_to_positive:
        print(f"\n⚠️ 서비스 부정을 긍정으로 잘못 예측한 경우: {len(wrong_negative_to_positive)}개")
        print("-" * 70)
        for i, case in enumerate(wrong_negative_to_positive[:5], 1):
            print(f"\n[{i}] {case['text'][:100]}...")

    # -----------------------------------------
    # 4. 불일치 케이스 분석
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("📊 4. 가장 많이 틀린 케이스 TOP 20")
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

    print(f"\n전체 불일치 케이스: {len(disagreements)}개 ({len(disagreements) / len(test_texts) * 100:.1f}%)")
    print("\n가장 많이 틀린 TOP 20:")
    print("-" * 70)

    for i, case in enumerate(disagreements[:20], 1):
        print(f"\n[{i}] 불일치 개수: {case['diff_count']}/12")
        print(f"    리뷰: {case['text'][:80]}...")

        # 어떤 측면에서 틀렸는지 표시
        print("    차이점:")
        for j in range(12):
            if case['true'][j] != case['pred'][j]:
                print(
                    f"      - {ASPECT_NAMES[j]}: 정답({SENTIMENT_NAMES[case['true'][j]]}) vs 예측({SENTIMENT_NAMES[case['pred'][j]]})")

    # -----------------------------------------
    # 5. 측면 감지 통계
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("📊 5. 측면 감지 통계")
    print("=" * 70)

    detected_true = (all_labels != 0).sum(dim=1).float()
    detected_pred = (all_preds != 0).sum(dim=1).float()

    print(f"평균 실제 측면 수: {detected_true.mean():.2f}")
    print(f"평균 예측 측면 수: {detected_pred.mean():.2f}")
    print(f"측면 수 차이: {abs(detected_pred.mean() - detected_true.mean()):.2f}")

    # 측면을 너무 많이/적게 예측하는 케이스
    over_detection = []
    under_detection = []

    for idx in range(len(test_texts)):
        true_count = detected_true[idx].item()
        pred_count = detected_pred[idx].item()
        diff = pred_count - true_count

        if diff >= 3:  # 3개 이상 더 많이 예측
            over_detection.append({'text': test_texts[idx], 'diff': diff})
        elif diff <= -3:  # 3개 이상 적게 예측
            under_detection.append({'text': test_texts[idx], 'diff': diff})

    if over_detection:
        print(f"\n⚠️ 측면을 과도하게 많이 감지한 케이스: {len(over_detection)}개")
        for case in over_detection[:3]:
            print(f"   {case['text'][:60]}... (차이: +{case['diff']}개)")

    if under_detection:
        print(f"\n⚠️ 측면을 너무 적게 감지한 케이스: {len(under_detection)}개")
        for case in under_detection[:3]:
            print(f"   {case['text'][:60]}... (차이: {case['diff']}개)")

            # -----------------------------------------
            # [수정] 6. 시각화 리포트 생성 (한글 폰트 대응)
            # -----------------------------------------
            print("\n" + "=" * 70)
            print("📊 6. 시각화 리포트 생성")
            print("=" * 70)

            # 윈도우/맥/리눅스 환경에 따른 폰트 설정
            import platform

            if platform.system() == 'Windows':
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif platform.system() == 'Darwin':  # Mac
                plt.rcParams['font.family'] = 'AppleGothic'
            else:  # Linux/Colab 등
                plt.rcParams['font.family'] = 'NanumBarunGothic'

            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

            aspect_f1_scores = []
            for i in range(12):
                report = classification_report(all_labels[:, i], all_preds[:, i], output_dict=True, zero_division=0)
                # 가중 평균 F1-Score 수집
                aspect_f1_scores.append(report['weighted avg']['f1-score'])

            # 시각화: 측면별 F1-Score 막대 그래프
            plt.figure(figsize=(12, 8))
            # 값에 따라 색상 변화 (낮을수록 붉은색, 높을수록 푸른색)
            colors = sns.color_palette("RdYlGn", len(aspect_f1_scores))
            # F1 score 기준으로 정렬해서 그리면 더 보기 좋습니다
            perf_df = pd.DataFrame({'Aspect': ASPECT_NAMES, 'F1': aspect_f1_scores}).sort_values('F1', ascending=False)

            sns.barplot(data=perf_df, x='F1', y='Aspect', palette='coolwarm')

            # 그래프에 숫자 표시
            for i, v in enumerate(perf_df['F1']):
                plt.text(v + 0.01, i, f'{v:.3f}', va='center')

            plt.title('측면별 모델 성능 (F1-Score)', fontsize=15)
            plt.xlabel('F1-Score (0.0 ~ 1.0)')
            plt.ylabel('평가 항목 (Aspect)')
            plt.xlim(0, 1.1)
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()

            plt.savefig(f"{SAVED_MODEL_PATH}/aspect_f1_report.png")
            print(f"✅ 한글 폰트 적용 그래프 저장 완료: aspect_f1_report.png")

            # 틀린 케이스만 CSV로 따로 저장 (나중에 분석용)
            error_analysis_df = pd.DataFrame(disagreements)
            error_analysis_df.to_csv(f"{SAVED_MODEL_PATH}/error_analysis.csv", index=False, encoding='utf-8-sig')
            print(f"✅ 분석용 에러 리스트 저장 완료: error_analysis.csv")

        # -----------------------------------------
        # [추가] 7. 데이터 밸런스 경고 요약
        # -----------------------------------------
        print("\n⚠️ 성능 최적화 제언:")
        negative_recalls = []
        for i in range(12):
            report = classification_report(all_labels[:, i], all_preds[:, i], output_dict=True, zero_division=0)
            if '2' in report:  # '부정' 라벨이 있다면
                negative_recalls.append(report['2']['recall'])

        avg_neg_recall = np.mean(negative_recalls) if negative_recalls else 0
        if avg_neg_recall < 0.5:
            print(f"👉 [위험] 부정(Negative) 재현율이 {avg_neg_recall:.2%}로 매우 낮습니다.")
            print(f"   학습 데이터에 '부정' 사례를 더 추가해야 실전에서 악플을 잡아낼 수 있습니다.")

        print("\n" + "=" * 70)
        print("✅ 모든 분석 및 시각화 완료!")
        print("=" * 70)