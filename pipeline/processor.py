"""
processors.py
텍스트 전처리, 샘플링, 번역, 분류, 감성분석 등 핵심 로직
"""

import re
import time
import json
import glob
import requests
import pandas as pd
import numpy as np
from collections import Counter
from kiwipiepy import Kiwi
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation._base import BaseRepresentation
import spacy

# Kiwi 전역 초기화
kiwi = Kiwi()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ============================================================
# 1. 텍스트 전처리
# ============================================================

def clean_text(text):
    """한글 리뷰 텍스트 정제"""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\^+', '', text)
    text = re.sub(r'[ㅋㅎㅠㅜㅡㅗ]+', '', text)
    text = re.sub(r'([~!?.])\1{2,}', r'\1', text)
    text = re.sub(r'([a-zA-Z])\1{3,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) >= 5 else ""


def clean_english_text(text):
    """영문 번역 후 노이즈 제거"""
    if not isinstance(text, str):
        return ""

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub('', text)
    text = re.sub(r'([~!?.,:;])\1{2,}', r'\1', text)
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', text)
    text = re.sub(r'[가-힣]+', '', text)
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) >= 5 else ""


# ============================================================
# 2. 데이터 로드
# ============================================================

def load_and_split_sentences(folder_path, max_reviews=10000):
    """JSONL 폴더에서 문장 단위 데이터 로드"""
    all_bodies = []
    jsonl_files = glob.glob(f"{folder_path}/*.jsonl")

    print(f"총 {len(jsonl_files)}개 파일 발견")

    for file_path in jsonl_files:
        if len(all_bodies) >= max_reviews:
            break

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    body = data.get('body', '')
                    if body and len(body) > 5:
                        all_bodies.append(body)

                    if len(all_bodies) >= max_reviews:
                        break
                except:
                    continue

    # 문장 분리
    sentences = []
    for review in all_bodies:
        sents = kiwi.split_into_sents(review)
        sentences.extend([sent.text for sent in sents])

    df = pd.DataFrame({'text': sentences})
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

    print(f"{len(all_bodies)}건 리뷰 → {len(df)}개 고유 문장 추출")
    return df


# ============================================================
# 3. 계층적 샘플링
# ============================================================

def stratified_sampling_for_translation(df, rare_keywords, sample_size=2000, seed=42):
    """다양성을 보장하는 계층적 샘플링"""
    print("\n🎯 계층적 샘플링 시작...")

    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() >= 10].copy()
    df['text_len'] = df['text_clean'].str.len()

    print(f"전처리 후: {len(df)}개 문장")

    samples = []

    # 희귀 측면 우선 샘플링
    print("\n📌 희귀 측면 우선 샘플링:")
    for category, keywords in rare_keywords.items():
        pattern = '|'.join(keywords)
        mask = df['text_clean'].str.contains(pattern, case=False, na=False)

        if mask.sum() > 0:
            n_samples = min(50, mask.sum())
            category_samples = df[mask].sample(n=n_samples, random_state=seed)
            samples.append(category_samples)
            print(f"  - {category}: {n_samples}개 선택")

    # 선택된 문장 제외
    selected_indices = pd.concat(samples).index if samples else pd.Index([])
    df_remaining = df[~df.index.isin(selected_indices)].copy()

    # 길이별 균등 샘플링
    current_count = len(selected_indices)
    remaining_needed = sample_size - current_count

    if remaining_needed > 0:
        print(f"\n📊 길이별 균등 샘플링 ({remaining_needed}개):")

        length_targets = {
            'short': (0, 30, 0.2),
            'medium': (30, 60, 0.5),
            'long': (60, 999, 0.3)
        }

        for length_type, (min_len, max_len, ratio) in length_targets.items():
            mask = (df_remaining['text_len'] >= min_len) & (df_remaining['text_len'] < max_len)
            available = df_remaining[mask]
            n_samples = min(int(remaining_needed * ratio), len(available))

            if n_samples > 0:
                length_samples = available.sample(n=n_samples, random_state=seed)
                samples.append(length_samples)
                print(f"  - {length_type} ({min_len}~{max_len}자): {n_samples}개")

    df_sample = pd.concat(samples).drop_duplicates(subset=['text_clean'])
    df_sample = df_sample.sample(n=min(sample_size, len(df_sample)), random_state=seed)

    print(f"\n✅ 최종 샘플: {len(df_sample)}개")

    df_sample['text'] = df_sample['text_clean']
    return df_sample.drop(columns=['text_clean', 'text_len'])


# ============================================================
# 4. 번역
# ============================================================

def sample_and_translate(df, rare_keywords, sample_size=2000, cache_file='translated_sample.csv'):
    """샘플링 + 번역 + 정제"""

    # 캐시 확인
    try:
        df_cached = pd.read_csv(cache_file)
        if len(df_cached) >= sample_size * 0.9:
            print(f"✅ 캐시 로드: {len(df_cached)}건")
            df_cached['text_en'] = df_cached['text_en'].apply(clean_english_text)
            df_cached = df_cached[df_cached['text_en'].str.len() > 0].copy()
            return df_cached
    except FileNotFoundError:
        pass

    # 샘플링
    df_sample = stratified_sampling_for_translation(df, rare_keywords, sample_size)

    # 번역
    print(f"\n🌐 번역 시작: {len(df_sample)}건...")
    translator = GoogleTranslator(source='ko', target='en')

    df_sample['text_en'] = None
    failed_count = 0

    for idx, row in df_sample.iterrows():
        try:
            translated = translator.translate(str(row['text']))
            df_sample.at[idx, 'text_en'] = translated

            if len(df_sample[df_sample['text_en'].notna()]) % 100 == 0:
                print(f"번역 진행: {len(df_sample[df_sample['text_en'].notna()])} / {len(df_sample)}")
                df_sample.to_csv(cache_file, index=False)

            time.sleep(0.1)
        except Exception as e:
            df_sample.at[idx, 'text_en'] = ""
            failed_count += 1
            time.sleep(1)

    # 영문 정제
    print("\n🧹 영문 텍스트 정제 중...")
    df_sample['text_en'] = df_sample['text_en'].apply(clean_english_text)
    df_sample = df_sample[df_sample['text_en'].str.len() > 0].copy()

    df_sample.to_csv(cache_file, index=False)
    print(f"\n✅ 번역 완료: {len(df_sample)}건")

    return df_sample


# ============================================================
# 5. 키워드 확장 (SBERT)
# ============================================================

def expand_english_keyword_dict(df_sample, base_dict_en, stopwords, noise_words):
    """SBERT로 영문 키워드 확장"""
    print("\n🔍 SBERT 기반 키워드 확장 시작...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    clean_sentences = df_sample['text_en'].dropna().astype(str).tolist()

    expanded_dict = {aspect: list(keywords) for aspect, keywords in base_dict_en.items()}
    aspect_embs = {cat: model.encode(" ".join(words)) for cat, words in base_dict_en.items()}
    sentence_embeddings = model.encode(clean_sentences, show_progress_bar=True)

    for aspect, target_emb in aspect_embs.items():
        sims = cosine_similarity([target_emb], sentence_embeddings)[0]
        top_indices = np.where(sims > 0.45)[0]

        new_kws = set()
        for idx in top_indices:
            doc = nlp(clean_sentences[idx].lower())
            for token in doc:
                if token.pos_ in ["NOUN", "ADJ"] and len(token.text) > 2:
                    if token.text not in stopwords and token.text not in noise_words:
                        new_kws.add(token.text)

        expanded_dict[aspect].extend(list(new_kws))
        expanded_dict[aspect] = list(set(expanded_dict[aspect]))
        print(f"✅ {aspect}: {len(expanded_dict[aspect])}개")

    return expanded_dict


# ============================================================
# 6. 한글 번역
# ============================================================

def translate_keywords_to_korean(aspect_keywords_en, essential_kr, cache_file='korean_keywords.json'):
    """영문 키워드 → 한글 번역"""

    # 캐시 확인
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            korean_dict = json.load(f)
        if korean_dict:
            print(f"✅ 한글 사전 로드: {cache_file}")
            return korean_dict
    except FileNotFoundError:
        pass

    translator = GoogleTranslator(source='en', target='ko')
    raw_korean_dict = {}

    print("\n🌐 영문 → 한글 번역 시작...")

    for aspect, keywords_en in aspect_keywords_en.items():
        print(f"\n🔍 [{aspect}] {len(keywords_en)}개 번역...", end=" ")
        translated_set = set()

        for kw in keywords_en:
            try:
                translated = translator.translate(kw).strip()
                if len(translated) > 1 and not translated.isdigit():
                    translated = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', translated)
                    translated_set.add(translated)
                time.sleep(0.1)
            except:
                time.sleep(1)
                continue

        raw_korean_dict[aspect] = list(translated_set)
        print(f"완료 ({len(translated_set)}개)")

    # 중복 제거
    print("\n⚖️ 중복 키워드 제거 중...")
    all_words = []
    for kws in raw_korean_dict.values():
        all_words.extend(kws)

    counts = Counter(all_words)
    duplicates = {kw for kw, count in counts.items() if count > 1}

    final_korean_dict = {}
    for aspect, kws in raw_korean_dict.items():
        clean_kws = [kw for kw in kws if kw not in duplicates]
        final_korean_dict[aspect] = sorted(clean_kws)

    # 필수 단어 추가
    for aspect, essentials in essential_kr.items():
        if aspect in final_korean_dict:
            final_korean_dict[aspect] = list(set(final_korean_dict[aspect] + essentials))

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(final_korean_dict, f, ensure_ascii=False, indent=2)

    print(f"\n💾 한글 사전 저장: {cache_file}")
    return final_korean_dict


# ============================================================
# 7. 어간 추출
# ============================================================

def extract_korean_stems(korean_keywords):
    """한글 키워드에서 어간 추출"""
    print("\n✂️ 한글 어간 추출 중...")

    stems_dict = {}
    for aspect, kws in korean_keywords.items():
        stems = set()
        for kw in kws:
            tokens = kiwi.tokenize(kw)
            for t in tokens:
                if t.tag in ['VA', 'VV', 'XR', 'NNG', 'NNP']:
                    stems.add(t.form)
        stems_dict[aspect] = list(stems)
        print(f"  [{aspect}]: {len(stems)}개 어간")

    return stems_dict


# ============================================================
# 8. 키워드 기반 분류
# ============================================================

def classify_with_korean_keywords(df, aspect_keywords_kr):
    """한글 키워드 기반 분류 (다중 라벨)"""
    print("\n🔍 키워드 기반 분류 시작...")

    df['aspect'] = None
    df['classification_method'] = None

    aspect_sets = {aspect: set(keywords) for aspect, keywords in aspect_keywords_kr.items()}

    for idx, row in df.iterrows():
        text = row['text']
        tokens = [t.form for t in kiwi.tokenize(text)]
        token_set = set(tokens)

        found_aspects = []
        for aspect, k_set in aspect_sets.items():
            if not token_set.isdisjoint(k_set):
                found_aspects.append(aspect)

        if found_aspects:
            df.at[idx, 'aspect'] = list(set(found_aspects))
            df.at[idx, 'classification_method'] = 'keyword'

    labeled = df[df['aspect'].notna()].copy()
    unlabeled = df[df['aspect'].isna()].copy()

    print(f"✅ 라벨링: {len(labeled)}건 ({len(labeled)/len(df)*100:.1f}%)")
    print(f"❌ 미분류: {len(unlabeled)}건")

    return labeled, unlabeled


# ============================================================
# 9. SBERT 의미 기반 분류
# ============================================================

def classify_with_multilingual_sbert(df_unlabeled, aspect_keywords_en, threshold=0.45):
    """다국어 SBERT로 의미 기반 분류 (다중 라벨)"""
    if df_unlabeled.empty:
        return pd.DataFrame(), df_unlabeled

    print(f"\n🧐 SBERT 의미 분석: {len(df_unlabeled)}건...")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    aspect_names = list(aspect_keywords_en.keys())
    aspect_descriptions = [" ".join(aspect_keywords_en[asp]) for asp in aspect_names]
    aspect_embs = model.encode(aspect_descriptions, convert_to_tensor=True)

    sentences = df_unlabeled['text'].tolist()
    sentence_embs = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

    cos_scores = util.cos_sim(sentence_embs, aspect_embs)

    sbert_results = []
    for i in range(len(sentences)):
        matched_indices = (cos_scores[i] >= threshold).nonzero(as_tuple=True)[0]

        if len(matched_indices) > 0:
            found_aspects = [aspect_names[idx] for idx in matched_indices]
            row_data = df_unlabeled.iloc[i].to_dict()
            row_data['aspect'] = found_aspects
            row_data['classification_method'] = 'sbert_semantic'
            sbert_results.append(row_data)

    df_sbert_labeled = pd.DataFrame(sbert_results)

    if not df_sbert_labeled.empty:
        labeled_texts = set(df_sbert_labeled['text'])
        df_still_unlabeled = df_unlabeled[~df_unlabeled['text'].isin(labeled_texts)].copy()
    else:
        df_still_unlabeled = df_unlabeled.copy()

    print(f"✅ SBERT 추가 라벨링: {len(df_sbert_labeled)}건")
    return df_sbert_labeled, df_still_unlabeled


# ============================================================
# 10. BERTopic + Groq 신규 측면 발견
# ============================================================

class GroqRepresentation(BaseRepresentation):
    """Groq API를 사용하는 BERTopic 커스텀 표현 모델"""

    def __init__(self, api_key, model="llama-3.3-70b-versatile", delay=0.5):
        self.api_key = api_key
        self.model = model
        self.delay = delay
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """BERTopic이 호출하는 메서드 - 각 토픽의 대표 단어 생성"""
        updated_topics = {}

        # documents를 안전하게 리스트로 변환
        if isinstance(documents, pd.DataFrame):
            docs_list = documents.iloc[:, 0].tolist()
        elif hasattr(documents, 'values'):
            docs_list = documents.values.flatten().tolist()
        elif isinstance(documents, list):
            docs_list = documents
        else:
            docs_list = list(documents)

        print(f"\n📊 총 {len(set(topics)) - (1 if -1 in topics else 0)}개 토픽 발견")

        for topic_id in sorted(set(topics)):
            if topic_id == -1:
                continue

            topic_docs = [docs_list[i] for i, t in enumerate(topics) if t == topic_id]
            print(f"🔍 토픽 {topic_id}: {len(topic_docs)}개 문서 분석 중...", end=" ")

            if len(topic_docs) > 10:
                import random
                topic_docs = random.sample(topic_docs, 10)

            representative_word = self._get_representative_word(topic_docs)

            if representative_word:
                updated_topics[topic_id] = [(representative_word, 1.0)]
                print(f"✅ '{representative_word}'")
            else:
                print("❌ 실패")

            time.sleep(self.delay)

        return updated_topics

    def _get_representative_word(self, documents):
        """문서 리스트를 받아 하나의 대표 단어를 생성"""
        context = "\n".join(documents[:5])[:1000]

        prompt = f"""You are analyzing cafe reviews. Here are sample reviews from a cluster:

{context}

Task: Provide ONE single English word that best represents the common aspect/theme in these reviews.

Examples of good outputs:
- Toilet
- Parking
- Pet
- Waiting
- Music
- Wifi
- Seating

Output ONLY the word, nothing else:"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.3
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=15
            )

            result = response.json()
            word = result['choices'][0]['message']['content'].strip()
            word = word.split()[0].strip('.,!?"\'').capitalize()

            if word.isalpha() and len(word) >= 2:
                return word

        except Exception as e:
            print(f"\n⚠️  Groq API 오류: {e}")

        return None


def discover_new_aspects_with_bertopic(df_sample, groq_api_key, top_n=5, min_cluster_size=30):
    """
    BERTopic + Groq를 사용하여 새로운 측면 발견

    Args:
        df_sample: 번역된 영문 샘플 데이터프레임
        groq_api_key: Groq API 키
        top_n: 반환할 최대 측면 개수
        min_cluster_size: 최소 클러스터 크기
    """
    print("\n[데이터 탐사] BERTopic + Groq로 새로운 카테고리 후보 찾는 중...")

    try:
        from bertopic import BERTopic
    except ImportError:
        print("⚠️  bertopic 패키지가 설치되지 않았습니다. pip install bertopic")
        return []

    docs = df_sample['text_en'].dropna().tolist()
    print(f"분석 대상 문서: {len(docs)}건")

    # 임베딩 모델
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Groq 표현 모델
    groq_repr = GroqRepresentation(api_key=groq_api_key)

    # BERTopic 모델
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=groq_repr,
        min_topic_size=min_cluster_size,
        nr_topics="auto",
        verbose=True,
        calculate_probabilities=False
    )

    # 토픽 모델링 실행
    print("\n🔄 클러스터링 및 토픽 추출 중...")
    topics, _ = topic_model.fit_transform(docs)

    # 토픽 정보 추출
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1].copy()

    print(f"\n발견된 토픽 수: {len(valid_topics)}개")
    print("\n토픽별 문서 수:")
    print(valid_topics[['Topic', 'Count', 'Name']].head(10))

    # 대표 단어 추출
    candidates = []

    for idx, row in valid_topics.head(top_n * 2).iterrows():
        topic_id = row['Topic']
        topic_words = topic_model.get_topic(topic_id)

        if topic_words:
            word = topic_words[0][0].strip().capitalize()
            if word and word.isalpha() and len(word) >= 2:
                candidates.append(word)

    unique_candidates = list(dict.fromkeys(candidates))[:top_n]
    print(f"\n✨ 발견된 새로운 측면 후보: {unique_candidates}")

    return unique_candidates


# ============================================================
# 11. 감성 분석 (규칙 기반)
# ============================================================

def add_sentiment_labels_rule_based(df, positive_words, negative_words, negation_patterns):
    """규칙 기반 감성 레이블링"""
    print("\n😊 규칙 기반 감성 분석...")

    df['sentiment'] = 1  # 기본: 중립

    for idx, row in df.iterrows():
        text = row['text']

        has_negation = any(neg in text for neg in negation_patterns)
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        if has_negation:
            if pos_count > neg_count:
                df.at[idx, 'sentiment'] = 0
            elif neg_count > pos_count:
                df.at[idx, 'sentiment'] = 2
        else:
            if pos_count > neg_count:
                df.at[idx, 'sentiment'] = 2
            elif neg_count > pos_count:
                df.at[idx, 'sentiment'] = 0

    print(f"감성 분포: {df['sentiment'].value_counts().to_dict()}")
    return df