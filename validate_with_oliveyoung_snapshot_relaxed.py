"""
validate_with_oliveyoung_snapshot.py의 "결합형 키워드가 제품명 원형과
매칭되지 않아 과소집계됐을 것"이라는 가설(THESIS_DRAFT.md 4.3/5.2절)을
실제로 검증한다. naver_keywords_final.csv의 risk_note에 이미 기록된
"결합형 → 원형(위험 단독어)" 대응관계를 이용해, 결합형 정확매칭(strict)과
원형까지 포함한 완화매칭(relaxed) 두 가지로 각각 집계해 비교한다.

⚠️ 완화매칭의 한계: 원형 단어는 정의상 동음이의어 위험이 있는 일상어다
(예: "탄력" "진정" "미스트"). 제품명에 이 단어가 등장해도 화장품
맥락이라는 보장이 없다 — 그래서 완화매칭 결과는 "상한선(upper bound)"
추정치로만 해석해야 하며, strict 결과를 대체하는 게 아니라 "과소집계가
실제로 어느 정도 규모였는지"를 가늠하는 보조 지표로만 쓴다.

사용법:
  python3 validate_with_oliveyoung_snapshot_relaxed.py
"""

import argparse

import pandas as pd

from validate_with_oliveyoung_snapshot import (
    load_recent_breakout_keywords,
    match_name_keyword,
)

# naver_keywords_final.csv risk_note에 명시된 "결합형 -> 원형(위험 단독어)" 대응관계.
# 성분 카테고리는 전부 전용어라 결합형 문제가 없으므로 제외(= validate 스크립트의
# 성분 매칭 로직은 그대로 strict만 사용).
COMBINED_TO_CORE = {
    "스킨에센스": "에센스",
    "쿠션파운데이션": "쿠션",
    "페이셜미스트": "미스트",
    "페이스오일": "오일",
    "립밤": "밤",
    "토너패드": "패드",
    "수분젤": "젤",
    "스킨토너": "토너",
    "수분크림": "크림",
    "선스틱": "스틱",
    "피부탄력": "탄력",
    "피부진정": "진정",
    "피부트러블": "트러블",
    "피지조절": "피지",
    "안면홍조": "홍조",
    "피부재생": "재생",
    "수분공급": "수분",
    "친환경 화장품": "친환경",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-input", default="naver_trend_features.csv")
    parser.add_argument("--products-input", default="ranking_products.csv")
    parser.add_argument("--since", default="2026-06-01")
    parser.add_argument("--output", default="oliveyoung_validation_relaxed.csv")
    args = parser.parse_args()

    keywords = load_recent_breakout_keywords(args.features_input, pd.Timestamp(args.since))
    products = pd.read_csv(args.products_input, encoding="utf-8-sig")

    rows = []
    for _, kw_row in keywords.iterrows():
        keyword, category = kw_row["keyword"], kw_row["category"]
        if category == "성분":
            continue  # 성분은 결합형 문제가 없음(전용어) — strict 결과가 그대로 최종치
        strict_ranks = match_name_keyword(keyword, products)
        core = COMBINED_TO_CORE.get(keyword)
        if core:
            relaxed_ranks = sorted(set(strict_ranks) | set(match_name_keyword(core, products)))
        else:
            relaxed_ranks = strict_ranks  # 원형 매핑이 없는 키워드(이미 단독으로 안전한 표현)
        rows.append(
            {
                "category": category,
                "keyword": keyword,
                "core_term": core if core else "(결합형 아님)",
                "strict_matched": len(strict_ranks),
                "relaxed_matched": len(relaxed_ranks),
                "base_n": len(products),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("카테고리별 strict vs relaxed 매칭 비율(1개 이상 매칭된 키워드 비율):")
    for category, group in results.groupby("category"):
        strict_rate = (group["strict_matched"] > 0).mean() * 100
        relaxed_rate = (group["relaxed_matched"] > 0).mean() * 100
        print(f"  {category}: strict {strict_rate:.0f}% -> relaxed {relaxed_rate:.0f}% ({len(group)}개 키워드)")

    print("\n키워드별 상세:")
    print(results.to_string(index=False))
    print(f"\n완료: {args.output}")


if __name__ == "__main__":
    main()
