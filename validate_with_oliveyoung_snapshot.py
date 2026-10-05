"""
RESEARCH_PLAN.md 1) 데이터 소스의 "실무 검증(보조)용 스냅샷"과 목차 Ⅳ.3
"올리브영 스냅샷 기반 실무 사례 검증"을 구현한다.

목적: 모델(Tukey IQR 라벨링)이 최근 breakout으로 식별한 키워드가 실제
올리브영 랭킹 상위 제품(판매랭킹 100위, `ranking_products.csv`/
`ranking_ingredients.csv`에 기수집된 1회성 스냅샷)에 얼마나 반영되고
있는지 사례로 제시한다. RESEARCH_PLAN.md에 이미 명시된 대로 이 결과는
본 모델의 핵심 평가지표(AUC/Precision@K)가 아니라 **보조적인 실무
사례 검증**이다 — 스냅샷이 특정 시점 1회 수집이라 인과관계나 통계적
유의성을 주장할 수 없고, "모델이 찾아낸 키워드가 현실 제품에도 실제로
나타나는가"를 보여주는 정성적 근거로만 사용한다.

방법:
  1. "최근 breakout 키워드" = naver_trend_features.csv에서 최근
     cutoff(기본 2026-06-01) 이후 is_breakout=True로 라벨링된 키워드
     (카테고리별로 집계, label_breakouts.py가 만든 라벨을 그대로 재사용).
  2. 성분 카테고리 키워드: ranking_ingredients.csv의 전성분표(쉼표 분리,
     analyze_ingredient_frequency.py와 동일한 "1,2-헥산다이올" 쉼표 보호
     로직 재사용)에서 각 키워드가 몇 개 제품에 실제로 배합돼 있는지 집계.
  3. 제형/컨셉·클레임/효능 카테고리 키워드: ranking_products.csv의 제품명
     텍스트에 키워드 문자열이 포함돼 있는지 집계.
  4. ⚠️ 알려진 한계: naver_keywords_final.csv의 키워드는 동음이의어 충돌을
     피하려고 "피지조절", "스킨에센스"처럼 결합형으로 바꾼 것들이 많다
     (KEYWORD_CANDIDATES.md 부록3 참고). 네이버 검색 트렌드 조회에는 이
     결합형이 꼭 필요했지만, 실제 제품명 마케팅 문구는 더 짧은 원형("에센스",
     "쿠션")을 쓰는 경우가 많아 완전 일치(substring) 매칭 기준으로는
     과소집계될 수 있다. 이 스크립트는 정확 문자열 포함 매칭만 쓰고, 이
     한계를 결과에 그대로 남긴다(임의로 재해석/완화하지 않음).

사용법:
  python3 validate_with_oliveyoung_snapshot.py
  python3 validate_with_oliveyoung_snapshot.py --since 2026-01-01
"""

import argparse
import re

import pandas as pd

from analyze_ingredient_frequency import protect_commas  # noqa: F401 (재사용 근거 명시용)


def split_ingredients(text):
    text = re.sub(r"(\d),(\d+-)", r"\1<C>\2", text)
    parts = [p.replace("<C>", ",").strip() for p in re.split("[,\n]", text)]
    return [p for p in parts if p]


def load_recent_breakout_keywords(path, since):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    recent = df[(df["period"] >= since) & (df["is_breakout"] == True)]  # noqa: E712
    return (
        recent[["keyword", "category"]]
        .drop_duplicates()
        .sort_values(["category", "keyword"])
        .reset_index(drop=True)
    )


def load_snapshot(products_path, ingredients_path):
    products = pd.read_csv(products_path, encoding="utf-8-sig")
    ingredients = pd.read_csv(ingredients_path, encoding="utf-8-sig")
    return products, ingredients


def match_ingredient_keyword(keyword, ingredients_df):
    matched_ranks = []
    for _, row in ingredients_df.iterrows():
        tokens = split_ingredients(str(row["ingredients"]))
        if any(keyword in token for token in tokens):
            matched_ranks.append(int(row["rank"]))
    return sorted(matched_ranks)


def match_name_keyword(keyword, products_df):
    hit = products_df[products_df["name"].str.contains(keyword, regex=False, na=False)]
    return sorted(hit["rank"].astype(int).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-input", default="naver_trend_features.csv")
    parser.add_argument("--products-input", default="ranking_products.csv")
    parser.add_argument("--ingredients-input", default="ranking_ingredients.csv")
    parser.add_argument("--since", default="2026-06-01", help="YYYY-MM-DD, 이 날짜 이후 breakout만 대상")
    parser.add_argument("--output", default="oliveyoung_validation_results.csv")
    args = parser.parse_args()

    keywords = load_recent_breakout_keywords(args.features_input, pd.Timestamp(args.since))
    products, ingredients = load_snapshot(args.products_input, args.ingredients_input)
    print(f"최근 breakout 키워드({args.since} 이후): {len(keywords)}개")
    print(f"올리브영 스냅샷: 제품 {len(products)}개, 전성분 확보 {len(ingredients)}개")

    rows = []
    for _, kw_row in keywords.iterrows():
        keyword, category = kw_row["keyword"], kw_row["category"]
        if category == "성분":
            ranks = match_ingredient_keyword(keyword, ingredients)
            base_n = len(ingredients)
        else:
            ranks = match_name_keyword(keyword, products)
            base_n = len(products)
        rows.append(
            {
                "category": category,
                "keyword": keyword,
                "matched_products": len(ranks),
                "base_n": base_n,
                "coverage_pct": round(len(ranks) / base_n * 100, 1) if base_n else float("nan"),
                "matched_ranks": ",".join(map(str, ranks[:10])) + ("..." if len(ranks) > 10 else ""),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n카테고리별 결과:")
    for category, group in results.groupby("category"):
        any_match = (group["matched_products"] > 0).mean() * 100
        print(f"  {category}: {len(group)}개 키워드 중 {any_match:.0f}%가 1개 이상 제품에서 매칭")

    print("\n매칭 많은 순 상위 15개:")
    print(
        results.sort_values("matched_products", ascending=False)
        .head(15)[["category", "keyword", "matched_products", "base_n", "coverage_pct"]]
        .to_string(index=False)
    )

    print(f"\n완료: {args.output}")


if __name__ == "__main__":
    main()
