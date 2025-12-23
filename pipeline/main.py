"""
main.py
ABSA 파이프라인 메인 실행 파일
"""

import sys
import pandas as pd
from config import (
    Config,
    RARE_ASPECT_KEYWORDS,
    STOPWORDS_EN,
    NOISE_WORDS_EN,
    ESSENTIAL_KOREAN,
    POSITIVE_WORDS_KR,
    NEGATIVE_WORDS_KR,
    NEGATION_PATTERNS_KR,
    ASPECT_MAPPING,
    BERTOPIC_CONFIG,
    get_final_english_dict,
    save_dict_to_json
)
from processor import (
    load_and_split_sentences,
    sample_and_translate,
    expand_english_keyword_dict,
    translate_keywords_to_korean,
    extract_korean_stems,
    classify_with_korean_keywords,
    classify_with_multilingual_sbert,
    add_sentiment_labels_rule_based,
    discover_new_aspects_with_bertopic
)


def main():
    """메인 파이프라인 실행"""

    print("=" * 60)
    print("ABSA 파이프라인 v3.0")
    print("=" * 60)

    # ===== STEP 1: 데이터 로드 =====
    print("\n=== STEP 1: 데이터 로드 ===")
    df_all = load_and_split_sentences(
        Config.JSONL_FOLDER_PATH,
        max_reviews=Config.MAX_REVIEWS
    )

    # 테스트용 샘플링
    if len(df_all) > Config.MAX_TEST_SAMPLES:
        print(f"데이터가 많아 {Config.MAX_TEST_SAMPLES:,}건만 샘플링")
        df_all = df_all.sample(n=Config.MAX_TEST_SAMPLES, random_state=42).reset_index(drop=True)

    # ===== STEP 2: 샘플링 및 번역 =====
    print("\n=== STEP 2: 샘플링 및 번역 ===")
    df_sample = sample_and_translate(
        df_all,
        RARE_ASPECT_KEYWORDS,
        sample_size=Config.SAMPLE_SIZE,
        cache_file=Config.CACHE_TRANSLATED
    )

    # ===== STEP 3: 영문 키워드 사전 확정 =====
    print("\n=== STEP 3: 영문 키워드 사전 확장 및 탐사 ===")

    # 3.1: BERTopic으로 새로운 측면 후보 찾기
    print("\n[3.1] BERTopic 기반 신규 측면 탐색")
    new_aspects = discover_new_aspects_with_bertopic(
        df_sample,
        Config.GROQ_API_KEY,
        top_n=BERTOPIC_CONFIG['top_n'],
        min_cluster_size=BERTOPIC_CONFIG['min_cluster_size']
    )

    # 3.2: 신규 측면 처리 방식 선택
    print("\n[3.2] 신규 측면 처리 방식 선택")
    print(f"발견된 신규 측면: {new_aspects}")

    merge_mode = input("\n신규 측면 처리 방식을 선택하세요:\n"
                      "  1: 기존 측면에 통합 (예: Parking → Position)\n"
                      "  2: 독립적인 새 측면으로 추가 (예: Parking 그대로 유지)\n"
                      "선택 (1/2): ").strip()

    base_dict = get_final_english_dict()

    if merge_mode == '1':
        # 기존 방식: 매핑 규칙 사용
        print("\n📦 [모드 1] 기존 측면에 통합")
        for new_aspect in new_aspects:
            if new_aspect in ASPECT_MAPPING:
                target = ASPECT_MAPPING[new_aspect]
                base_dict[target].append(new_aspect.lower())
                base_dict[target] = list(set(base_dict[target]))
                print(f"  ✅ {new_aspect} → {target}로 통합")
            elif new_aspect not in base_dict:
                base_dict[new_aspect] = [new_aspect.lower()]
                print(f"  🆕 새로운 측면 추가: {new_aspect}")

    else:
        # 새로운 방식: 모두 독립 측면으로 추가
        print("\n🆕 [모드 2] 독립적인 새 측면으로 추가")
        for new_aspect in new_aspects:
            if new_aspect not in base_dict:
                base_dict[new_aspect] = [new_aspect.lower()]
                print(f"  ✅ 새 측면 추가: {new_aspect}")
            else:
                print(f"  ⚠️  이미 존재하는 측면: {new_aspect}")

    # 3.3: SBERT로 키워드 확장 (선택)
    use_sbert = input("\nSBERT 키워드 확장을 진행하시겠습니까? (y/n): ").lower() == 'y'

    if use_sbert:
        print("\n[3.3] SBERT 기반 키워드 확장")
        expanded_keywords_en = expand_english_keyword_dict(
            df_sample,
            base_dict,
            STOPWORDS_EN,
            NOISE_WORDS_EN
        )
    else:
        print("\n[3.3] SBERT 확장 스킵 (수동 정제 사전 사용)")
        expanded_keywords_en = base_dict

    print("\n✨ 최종 영문 키워드 사전:")
    for aspect, keywords in expanded_keywords_en.items():
        print(f"  [{aspect}]: {len(keywords)}개 키워드")

    # 저장
    save_dict_to_json(expanded_keywords_en, Config.CACHE_EXPANDED_EN)

    # ===== STEP 4: 한글 키워드 변환 =====
    print("\n=== STEP 4: 한글 키워드 변환 ===")
    raw_korean_keywords = translate_keywords_to_korean(
        expanded_keywords_en,
        ESSENTIAL_KOREAN,
        cache_file=Config.CACHE_KOREAN_KW
    )

    # 어간 추출
    korean_keywords = extract_korean_stems(raw_korean_keywords)
    save_dict_to_json(korean_keywords, Config.CACHE_KOREAN_STEMS)

    # ===== STEP 5: 하이브리드 라벨링 =====
    print("\n=== STEP 5: [Hybrid 1] 키워드 기반 라벨링 ===")
    df_labeled, df_unlabeled = classify_with_korean_keywords(df_all, korean_keywords)

    print("\n=== STEP 5.5: [Hybrid 2] SBERT 의미 기반 라벨링 ===")
    df_sbert_labeled, df_still_unlabeled = classify_with_multilingual_sbert(
        df_unlabeled,
        expanded_keywords_en,
        threshold=Config.SBERT_THRESHOLD
    )

    # 데이터 통합
    df_combined = pd.concat([df_labeled, df_sbert_labeled], ignore_index=True)

    # 저장 (리스트 → 문자열 변환)
    df_save = df_combined.copy()
    df_save['aspect'] = df_save['aspect'].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    df_save.to_csv('labeled_results_step5.csv', index=False, encoding='utf-8-sig')
    df_still_unlabeled.to_csv('still_unlabeled_step5.csv', index=False, encoding='utf-8-sig')

    # ===== 결과 보고 =====
    print("\n" + "=" * 60)
    print("📊 최종 통계")
    print("=" * 60)
    print(f"전체 데이터: {len(df_all):,}건")
    print(f"✅ 키워드 매칭: {len(df_labeled):,}건")
    print(f"✅ SBERT 구제: {len(df_sbert_labeled):,}건")
    print(f"✅ 총 라벨링: {len(df_combined):,}건 ({len(df_combined)/len(df_all)*100:.1f}%)")
    print(f"❌ 미분류: {len(df_still_unlabeled):,}건")

    # 다중 측면 통계
    multi_count = df_combined['aspect'].apply(
        lambda x: len(x) if isinstance(x, list) else 1
    ).gt(1).sum()
    print(f"🔗 복합 측면: {multi_count:,}건")

    print("\n💾 결과 파일:")
    print("  - labeled_results_step5.csv")
    print("  - still_unlabeled_step5.csv")
    print("=" * 60)

    # ===== STEP 6: 감성 분석 (선택) =====
    if input("\n감성 분석을 진행하시겠습니까? (y/n): ").lower() == 'y':
        print("\n=== STEP 6: 감성 분석 ===")
        df_combined = add_sentiment_labels_rule_based(
            df_combined,
            POSITIVE_WORDS_KR,
            NEGATIVE_WORDS_KR,
            NEGATION_PATTERNS_KR
        )

        # 최종 저장
        df_final = df_combined.copy()
        df_final['aspect'] = df_final['aspect'].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        df_final.to_csv('labeled_data_final_with_sentiment.csv', index=False, encoding='utf-8-sig')
        print("\n💾 최종 결과: labeled_data_final_with_sentiment.csv")

    print("\n🎉 파이프라인 완료!")


if __name__ == "__main__":
    try:
        # API 키 확인
        if not Config.GROQ_API_KEY:
            print("❌ .env 파일에 GROQ_API_KEY가 설정되지 않았습니다.")
            sys.exit(1)

        main()

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)