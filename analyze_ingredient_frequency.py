"""
성분 카테고리 키워드 후보 검증용 빈도분석 스크립트.
ranking_ingredients.csv(올리브영 랭킹 100개 중 화장품 97개 전성분표)를 바탕으로
각 성분이 몇 개 제품에 배합되어 있는지(제품 커버리지 기준) 집계한다.

사용법: python3 analyze_ingredient_frequency.py
"""

import csv
import re
from collections import Counter

INPUT_CSV = "ranking_ingredients.csv"


def protect_commas(text):
    # "1,2-헥산다이올", "2,3-부탄다이올"처럼 성분명 내부에 쉼표가 포함된 경우
    # 단순 split 시 잘못 분리되므로, 분리 전에 해당 쉼표를 임시 치환해 보호한다.
    return re.sub(r"(\d),(\d+-)", r"\1<C>\2", text)


def split_ingredients(text):
    text = protect_commas(text)
    parts = [p.replace("<C>", ",").strip() for p in re.split("[,\n]", text)]
    return [p for p in parts if p]


def main():
    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    product_counter = Counter()
    for r in rows:
        parts = split_ingredients(r["ingredients"])
        for p in set(parts):
            product_counter[p] += 1

    print(f"제품 수: {n}, 고유 성분 수: {len(product_counter)}\n")
    print("=== 배합 빈도 상위 100개 (제품 커버리지 기준) ===")
    for name, cnt in product_counter.most_common(100):
        print(f"{cnt:3d}/{n}  {name}")


if __name__ == "__main__":
    main()
