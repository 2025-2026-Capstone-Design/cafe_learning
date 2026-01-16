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

        # 🔧 라벨 전처리 (따옴표 제거)
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
    import matplotlib.pyplot as plt
    import platform
    # 1. 한글 폰트 설정 (Windows 기준)
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    elif platform.system() == 'Darwin':  # Mac
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumBarunGothic'

    # 2. 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 70)
    print("🔍 모델 평가 및 신뢰도 분석")
    print("=" * 70)

    # 2. 파일 로드
    file_old = "split_test.csv"

    print(f"📂 데이터 로딩 중...")

    # 기본 분할 (학습 시와 동일)
    df = pd.read_csv(DATA_PATH, dtype={'label': str})
    temp_texts, test_texts, temp_labels, test_labels = train_test_split(
        df['Original_Review'].tolist(),
        df['label'].tolist(),
        test_size=500,
        random_state=SEED,
        shuffle=True
    )

    # 🔧 고난도 데이터 추가
    df_hard = pd.read_csv("test_hard.csv", dtype={'label': str})
    test_texts.extend(df_hard['Original_Review'].tolist())
    test_labels.extend(df_hard['label'].tolist())

    print(f"총 테스트 데이터: {len(test_texts)}개 (기본 500 + 추가 {len(df_hard)})")
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
    #
    # # -----------------------------------------
    # # 3. 혼동 행렬 (서비스 측면)
    # # -----------------------------------------
    # print("\n" + "=" * 70)
    # print("📊 3. 혼동 행렬 분석 (서비스 측면)")
    # print("=" * 70)
    #
    # service_idx = 8  # 서비스
    # y_true_service = all_labels[:, service_idx].numpy()
    # y_pred_service = all_preds[:, service_idx].numpy()
    #
    # cm = confusion_matrix(y_true_service, y_pred_service)
    # print("\n서비스 측면 혼동 행렬:")
    # print("          예측→")
    # print("실제↓   해당없음  긍정   부정   중립")
    # labels_cm = ["해당없음", "긍정", "부정", "중립"]
    # for i, label in enumerate(labels_cm):
    #     print(f"{label:6s}  ", end="")
    #     for j in range(4):
    #         if i < cm.shape[0] and j < cm.shape[1]:
    #             print(f"{cm[i][j]:6d} ", end="")
    #         else:
    #             print(f"{'0':6s} ", end="")
    #     print()
    #
    # # 부정을 긍정으로 잘못 예측한 경우 찾기
    # wrong_negative_to_positive = []
    # for idx in range(len(test_texts)):
    #     true_val = all_labels[idx, service_idx].item()
    #     pred_val = all_preds[idx, service_idx].item()
    #
    #     if true_val == 2 and pred_val == 1:  # 실제 부정인데 긍정으로 예측
    #         wrong_negative_to_positive.append({
    #             'text': test_texts[idx],
    #             'true': true_val,
    #             'pred': pred_val
    #         })
    #
    # if wrong_negative_to_positive:
    #     print(f"\n⚠️ 서비스 부정을 긍정으로 잘못 예측한 경우: {len(wrong_negative_to_positive)}개")
    #     print("-" * 70)
    #     for i, case in enumerate(wrong_negative_to_positive[:5], 1):
    #         print(f"\n[{i}] {case['text'][:100]}...")
    # -----------------------------------------
    # 3. 모든 측면 혼동 행렬 분석 (시각화 리포트)
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("📊 3. 모든 측면 혼동 행렬 시각화 중...")
    print("=" * 70)

    import math

    # 12개 측면을 4행 3열로 배치
    cols = 3
    rows = math.ceil(len(ASPECT_NAMES) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 20))
    fig.suptitle('모든 측면별 혼동 행렬 (Confusion Matrix)', fontsize=25, y=1.02)

    labels_cm = ["해당없음", "긍정", "부정", "중립"]

    for i, aspect_name in enumerate(ASPECT_NAMES):
        r, c = i // cols, i % cols
        y_true = all_labels[:, i].numpy()
        y_pred = all_preds[:, i].numpy()

        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        # 히트맵 그리기
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r, c],
                    xticklabels=labels_cm, yticklabels=labels_cm, cbar=False)

        axes[r, c].set_title(f'[{aspect_name}]', fontsize=15)
        axes[r, c].set_xlabel('예측값')
        axes[r, c].set_ylabel('실제값')

    # 남는 빈 칸 제거 (12개면 딱 맞지만 혹시 모르니)
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.savefig(f"{SAVED_MODEL_PATH}/all_aspects_confusion_matrix.png", dpi=300)
    print(f"✅ 모든 측면 혼동 행렬 저장 완료: {SAVED_MODEL_PATH}/all_aspects_confusion_matrix.png")
    # -----------------------------------------
    # 4. 불일치 케이스 분석
    # -----------------------------------------
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

    print(f"\n전체 불일치 케이스: {len(disagreements)}개 ({len(disagreements) / len(test_texts) * 100:.1f}%)")
    print("\n가장 많이 틀린 TOP 10:")
    print("-" * 70)

    for i, case in enumerate(disagreements[:10], 1):
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
    # 6. 감성별 상세 F1-Score 히트맵 생성
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("📊 6. 감성별 상세 성능(F1-Score) 분석 중...")
    print("=" * 70)

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import platform

    # 1. 폰트 설정 (오류 방지)
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumBarunGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 데이터 추출 (긍정:1, 부정:2, 중립:3 별로 F1-score 수집)
    sentiment_labels = ['1', '2', '3']
    sentiment_names = ['긍정', '부정', '중립']
    f1_matrix = []

    for i in range(12):
        report = classification_report(all_labels[:, i], all_preds[:, i], output_dict=True, zero_division=0)
        scores = []
        for s_code in sentiment_labels:
            # 해당 감성 라벨이 데이터에 있으면 점수 가져오기, 없으면 0.0
            score = report.get(s_code, {}).get('f1-score', 0.0)
            scores.append(score)
        f1_matrix.append(scores)

    # 데이터프레임 변환
    df_f1 = pd.DataFrame(f1_matrix, index=ASPECT_NAMES, columns=sentiment_names)

    # 3. 히트맵 시각화
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_f1, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)

    plt.title('측면별/감성별 상세 성능 (F1-Score)', fontsize=18, pad=20)
    plt.ylabel('평가 항목 (Aspect)', fontsize=12)
    plt.xlabel('감성 (Sentiment)', fontsize=12)
    plt.tight_layout()

    # 이미지 저장
    plt.savefig(f"{SAVED_MODEL_PATH}/sentiment_detailed_f1.png", dpi=300)
    print(f"✅ 상세 분석 히트맵 저장 완료: {SAVED_MODEL_PATH}/sentiment_detailed_f1.png")
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

    # -----------------------------------------
    # [추가] 8. 중립(Neutral, 3) 오답 정밀 분석
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("🕵️‍♂️ '중립(Neutral)' 오답 집중 추적")
    print("=" * 70)

    neutral_errors = []

    for idx in range(len(test_texts)):
        for aspect_idx in range(12):
            true_val = all_labels[idx, aspect_idx].item()
            pred_val = all_preds[idx, aspect_idx].item()

            # 케이스 1: 정답은 중립(3)인데, 모델이 다른 걸로 예측함 (Recall 0의 원인)
            if true_val == 3 and pred_val != 3:
                neutral_errors.append({
                    'Type': '놓친 중립(Missed)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': '중립',
                    'Pred': SENTIMENT_NAMES[pred_val]
                })

            # 케이스 2: 모델이 중립(3)이라고 예측했는데, 정답은 아님 (Precision 0의 원인)
            elif pred_val == 3 and true_val != 3:
                neutral_errors.append({
                    'Type': '잘못된 중립(False Neutral)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': SENTIMENT_NAMES[true_val],
                    'Pred': '중립'
                })

    # 결과 출력
    if neutral_errors:
        print(f"⚠️ 총 {len(neutral_errors)}개의 중립 관련 오류가 발견되었습니다.\n")

        # 보기 좋게 상위 20개만 출력
        for i, err in enumerate(neutral_errors[:20], 1):
            print(f"[{i}] {err['Type']}")
            print(f"   측면: {err['Aspect']}")
            print(f"   문장: {err['Text']}")
            print(f"   결과: 정답({err['True']}) vs 예측({err['Pred']})")
            print("-" * 50)

        # CSV로 전체 저장 (엑셀에서 확인용)
        err_df = pd.DataFrame(neutral_errors)
        err_df.to_csv(f"{SAVED_MODEL_PATH}/neutral_errors.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ 전체 중립 오답 리스트 저장 완료: {SAVED_MODEL_PATH}/neutral_errors.csv")

    else:
        print("🎉 와우! 중립 감성에 대한 오류가 하나도 없습니다.")

    # -----------------------------------------
    # [추가] 8. 중립(Neutral, 3) 오답 정밀 분석
    # -----------------------------------------
    print("\n" + "=" * 70)
    print("🕵️‍♂️ '중립(Neutral)' 오답 집중 추적")
    print("=" * 70)

    negative_errors = []

    for idx in range(len(test_texts)):
        for aspect_idx in range(12):
            true_val = all_labels[idx, aspect_idx].item()
            pred_val = all_preds[idx, aspect_idx].item()

            # 케이스 1: 정답은 중립(3)인데, 모델이 다른 걸로 예측함 (Recall 0의 원인)
            if true_val == 2 and pred_val != 2:
                negative_errors.append({
                    'Type': '놓친 부정(Missed)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': '부정',
                    'Pred': SENTIMENT_NAMES[pred_val]
                })

            # 케이스 2: 모델이 중립(3)이라고 예측했는데, 정답은 아님 (Precision 0의 원인)
            elif pred_val == 2 and true_val != 2:
                negative_errors.append({
                    'Type': '잘못된 부정(False Neutral)',
                    'Text': test_texts[idx],
                    'Aspect': ASPECT_NAMES[aspect_idx],
                    'True': SENTIMENT_NAMES[true_val],
                    'Pred': '부정'
                })

    # 결과 출력
    if negative_errors:
        print(f"⚠️ 총 {len(negative_errors)}개의 중립 관련 오류가 발견되었습니다.\n")

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