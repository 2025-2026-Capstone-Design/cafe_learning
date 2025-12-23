"""
config.py
설정, 상수, 키워드 사전 관리
"""

import os
import json
from dotenv import load_dotenv
from pathlib import Path

# 환경 변수 로드
load_dotenv()

# ============================================================
# API 키 및 경로 설정
# ============================================================

class Config:
    """전역 설정"""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    JSONL_FOLDER_PATH = project_root / "cafe_reviews"

    # Sampling & Translation
    SAMPLE_SIZE = 2000
    MAX_REVIEWS = 10000
    MAX_TEST_SAMPLES = 30000

    # Classification Thresholds
    SBERT_THRESHOLD = 0.45
    MIN_CLUSTER_SIZE = 30
    BERTOPIC_MERGE_MODE = '2'

    # LLM Limits (Groq 무료 제한)
    MAX_LLM_SAMPLES = 800
    MAX_SENTIMENT_SAMPLES = 800

    # Model Settings
    MODEL_NAME = 'klue/roberta-base'
    MAX_LENGTH = 128
    TRAIN_EPOCHS = 3
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32

    # Cache Files
    CACHE_TRANSLATED = 'translated_sample.csv'
    CACHE_KOREAN_KW = 'korean_keywords.json'
    CACHE_EXPANDED_EN = 'expanded_keywords_en.json'
    CACHE_KOREAN_STEMS = 'final_korean_stems.json'


# ============================================================
# 기본 측면 사전 (Base Aspect Dictionary)
# ============================================================

BASE_ASPECT_DICT_EN = {
    "Coffee/Drink": [
        "coffee", "espresso", "latte", "americano", "caffeine",
        "acidity", "beans", "brew", "tea", "ade", "smoothie"
    ],
    "Dessert/Bakery": [
        "bread", "cake", "dessert", "scone", "pastry", "bakery",
        "bakery", "croissant", "tiramisu", "salt bread", "sweet"
    ],
    "Vibe": ["atmosphere", "interior", "mood", "lighting", "music"],
    "Service": ["service", "staff", "friendly", "kind", "waiting"],
    "Price": ["price", "expensive", "reasonable", "value"],
    "Position": ["location", "parking", "access", "station"]
}

# ============================================================
# 수동 보강 키워드 (Manual Additions)
# ============================================================

MANUAL_ADDITIONS_EN = {
    "Coffee/Drink": [
        "ade", "affogato", "americano", "cappuccino", "cold brew", "decaf",
        "espresso", "grapefruit", "hojicha", "hot chocolate", "latte",
        "lemon", "milk tea", "smoothie", "sorbet", "strawberry", "vanilla",
        "acidity", "bitter", "aroma", "scent", "sourness", "caffeine"
    ],
    "Dessert/Bakery": [
        "apple", "basque", "berry", "black sesame", "bread", "brownie", "butter bar",
        "cake", "cheese", "chocolate", "croissant", "dessert", "financiers",
        "ganache", "gateau", "ice cream", "macarpone", "mugwort", "pastry",
        "pudding", "scone", "sherbet", "tiramisu", "waffle",
        "buttery", "chewy", "creamy", "crispy", "crunchiness", "moist",
        "nutty", "savory", "texture", "soggy", "sweetness"
    ],
    "Service": [
        "clerk", "checkout", "delicately served", "explanation", "friendly", "hospitality",
        "kiosk", "management", "number ticket", "opening", "operation", "order form",
        "organized well", "pickup line", "polite", "queue", "reservation", "revisit",
        "staff", "takeout", "turnover rate", "waiting system", "welcome"
    ],
    "Vibe": [
        "ambiance", "atmosphere", "background music", "calm", "clean", "comfortable",
        "compact", "cozy", "decor", "decorated", "design", "editing shop", "emotional",
        "european", "fancy", "furniture", "hip", "hot place", "interior", "lighting",
        "lo-fi", "media art", "minimal", "modern", "mood", "neat", "night view",
        "outdoor", "photo zone", "quiet", "retro", "seating", "showroom", "sophisticated",
        "spacious", "studio", "stylish", "terrace", "vintage", "view", "windowed", "wood tone"
    ],
    "Price": [
        "affordable", "cheap", "fair price", "gift", "goods zone", "high quality",
        "merchandise", "overpriced", "packaging", "pricey", "reasonable", "satisfaction",
        "value for money", "worth"
    ],
    "Position": [
        "access", "accessibility", "address", "alley", "close", "convenient", "distance",
        "exit", "landmark", "location", "nearby", "parking fee", "park", "station", "traffic", "walk"
    ]
}


# ============================================================
# 필수 한글 키워드 (보강용)
# ============================================================

ESSENTIAL_KOREAN = {
    "Coffee/Drink": ["커피", "음료", "라떼", "에이드", "티", "차", "원두", "산미", "슬러시", "그라니따", "시음", "믹스"],
    "Dessert/Bakery": ["빵", "디저트", "케이크", "구움과자", "스콘", "디저트", "달콤","오란다", "건빵", "뉴욕롤"],
    "Vibe": ["분위기", "인테리어", "매장", "공간", "힙", "예쁘", "이쁘"],
    "Service": ["서비스", "사장님", "직원", "친절", "대기", "웨이팅", "줄"],
    "Price": ["가격", "가성비", "비싸", "저렴", "혜자"],
    "Position": ["위치", "주차", "가깝", "멀", "역", "출구"],
    "Event/Goods": ["굿즈", "교동", "인형", "키링", "팝업", "이벤트", "생일", "전시", "피규어"],
}

# ============================================================
# 희귀 측면 키워드 (샘플링용)
# ============================================================

RARE_ASPECT_KEYWORDS = {
    '주차': ['주차', '주차장', '주차공간', '차', '파킹'],
    '화장실': ['화장실', '변기', '세면대', '휴지', '깨끗'],
    '애완동물': ['강아지', '반려동물', '펫', '애완', '개'],
    '와이파이': ['와이파이', 'wifi', '인터넷', '무선'],
    '콘센트': ['콘센트', '충전', '플러그', '전원'],
    '음악': ['음악', '노래', 'bgm', '재즈', '팝송'],
    '대기': ['웨이팅', '대기', '줄', '기다림', '예약'],
    '좌석': ['좌석', '자리', '테이블', '의자', '앉'],
    '빵': ['빵', '스콘', '크루아상', '베이글', '식빵'],
    '디저트': ['디저트', '케이크', '마카롱', '쿠키'],
    '온도': ['따뜻', '차가', '뜨거', '시원', '미지근'],
    '혼잡도': ['붐비', '사람많', '한적', '조용', '시끄']
}


# ============================================================
# 노이즈 필터링용 불용어
# ============================================================

STOPWORDS_EN = {
    # [기본 및 일반어]
    'good', 'great', 'better', 'best', 'most', 'very', 'really', 'quite',
    'only', 'just', 'last', 'first', 'second', 'third', 'time', 'place',
    'stores', 'store', 'again', 'also', 'however', 'though', 'many',
    'much', 'more', 'less', 'other', 'another', 'same', 'different',
    'lot', 'nice', 'bit', 'open', 'drinks', 'various', 'easy', 'stay',
    'surprised', 'particular', 'types', 'while', 'part', 'right', 'old',
    'conversation', 'pretty', 'bad', 'big', 'small', 'work', 'owner',
    'cafe', 'coffee', 'visit', 'visited', 'often', 'favorite', 'think',
    'seems', 'thought', 'expected', 'wanted', 'tried', 'found', 'happened',
    'overall', 'already', 'almost', 'always', 'back', 'next', 'first time',

    # [수치 및 범용 명사 - expanded_json 오염 주범]
    'things', 'people', 'someone', 'today', 'home', 'way', 'amount',
    'item', 'items', 'fact', 'reason', 'point', 'case', 'piece', 'pieces',
    'bit', 'lot', 'variety', 'various', 'such', 'able', 'possible',
    'high', 'quality', 'essential', 'one', 'day', 'days', 'weather',
    'points', 'single', 'basic', 'morning', 'evening', 'afternoon',
    'long', 'new', 'full', 'line', 'door', 'floor', 'times', 'thing',
    'part', 'fact', 'level', 'side', 'background', 'kind', 'sure',
    'certain', 'several', 'multiple', 'everything', 'anything',

    # [가족 및 인칭]
    'mom', 'dad', 'family', 'friends', 'coworkers', 'boyfriend', 'girlfriend',
    'person', 'someone', 'everyone', 'customers', 'guests', 'tourists',

    # [범용 형용사/부사]
    'perfect', 'amazing', 'awesome', 'special', 'different', 'similar',
    'certain', 'particular', 'least', 'most', 'many', 'much', 'interested',
    'typical', 'charming', 'serious', 'disappointing', 'lucky', 'pleasant',

    # [동사 및 상태]
    'feel', 'looks', 'tried', 'found', 'want', 'wanted', 'need', 'needs',
    'going', 'went', 'came', 'coming', 'take', 'took', 'get', 'got',
    'makes', 'made', 'lets', 'done', 'used', 'using'
}

NOISE_WORDS_EN = {
    # [지명 및 특정 상호]
    'seongsu', 'dong', 'seongsu-dong', 'seoul', 'shinnonhyeon', 'station',
    'alley', 'road', 'forest', 'branch', 'branches', 'seongbuk-dong', 'jongno',
    'yeonnnam', 'apgujeong', 'seongbuk', 'station', 'exit', 'minutes',

    # [분석과 상관없는 고유 명사]
    'gutteroite', 'jayeondo', 'london', 'artist', 'bakery', 'tendong',
    'omakase', 'gongcha', 'earl', 'gray', 'spacy', 'kiwi', 'tiger',
    'yuzu', 'jambon', 'spirited', 'ups', 'overlapping', 'gambling',
    'said', 'it is said', 'reputation', 'famous', 'famous for',
    'tiger', 'magpie', 'henry', 'boisok', 'imu', 'sinnonhyeon',

    # [기타 노이즈]
    'reviews', 'picture', 'pictures', 'photo', 'photos', 'shot',
    'instagram', 'naver', 'kakaotalk', 'tag', 'kiosk', 'coupon',
    'description', 'descriptions', 'won', 'price', 'prices'
}


# ============================================================
# 측면 매핑 규칙 (신규 측면 통합용)
# ============================================================

ASPECT_MAPPING = {
    'Coffee': 'Coffee/Drink',
    'Drink': 'Coffee/Drink',
    'Flavor': 'Coffee/Drink',  # 또는 맛의 전반적 평가면 한곳으로 지정
    'Bread': 'Dessert/Bakery',
    'Desserts': 'Dessert/Bakery',
    'Scone': 'Dessert/Bakery',
    'Cake': 'Dessert/Bakery',
    'Atmosphere': 'Vibe',
    'Ambiance': 'Vibe',
    'Ambience': 'Vibe',
    'Seating': 'Vibe',
    'Queue': 'Service',
    'Waiting': 'Service',
    'Opening': 'Service',
    'Menu': 'Price',
    'Goods': 'Event/Goods',
    'Character': 'Event/Goods',
    'Popup': 'Event/Goods',
    'Slush': 'Coffee/Drink',
    'Granita': 'Coffee/Drink'
}


# ============================================================
# BERTopic 설정
# ============================================================

BERTOPIC_CONFIG = {
    'top_n': 5,                    # 발견할 최대 측면 개수
    'min_cluster_size': 30,        # 최소 클러스터 크기
    'groq_model': 'llama-3.3-70b-versatile',
    'groq_delay': 0.5              # API 호출 간 대기 시간
}


# ============================================================
# 감성 분석용 긍정/부정 키워드
# ============================================================

POSITIVE_WORDS_KR = [
    '좋', '맛있', '친절', '깔끔', '최고', '훌륭', '완벽',
    '추천', '만족', '괜찮'
]

NEGATIVE_WORDS_KR = [
    '별로', '불친절', '비싸', '더럽', '최악', '실망',
    '불만', '아쉽', '그냥', '보통'
]

NEGATION_PATTERNS_KR = ['안', '못', '없', '지 않', '지않']


# ============================================================
# 유틸리티 함수
# ============================================================

def load_aspect_config(file_path='aspect_config.json'):
    """JSON에서 측면 설정 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  {file_path} 없음. 기본 사전 사용")
        return BASE_ASPECT_DICT_EN


def merge_aspect_dicts(base_dict, additions_dict):
    """두 사전을 병합 (중복 제거)"""
    merged = {}
    all_aspects = set(list(base_dict.keys()) + list(additions_dict.keys()))

    for aspect in all_aspects:
        base_kws = base_dict.get(aspect, [])
        additional_kws = additions_dict.get(aspect, [])
        merged[aspect] = list(set(base_kws + additional_kws))

    return merged


def get_final_english_dict():
    """최종 영문 키워드 사전 반환"""
    return merge_aspect_dicts(BASE_ASPECT_DICT_EN, MANUAL_ADDITIONS_EN)


def save_dict_to_json(data, filepath):
    """사전을 JSON으로 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 저장 완료: {filepath}")


def load_dict_from_json(filepath):
    """JSON에서 사전 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  {filepath} 파일 없음")
        return None