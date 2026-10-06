"""
naver_trend_features.csv에 "계절성 대비 이례적 상승 여부" feature를 추가한다.
RESEARCH_PLAN.md "3) Feature 설계"의 세 번째 항목을 구현한다.

방법론:
  - 문태남(2017, 인하대 VECM 논문)은 거시경제 시계열 모형에 계절성 더미
    (seasonal dummy)를 BIC로 선정해 포함시켜, 특정 시점의 계절적 패턴과
    "실제 이례적 변화"를 분리했다. 본 연구는 월별 거시지표가 아닌 주간
    검색 비율(ratio) 데이터이므로, 동일한 목적(계절 요인과 이례적 변화의
    분리)을 달성하되 구현은 더 단순한 **계절-나이브(seasonal-naive) 기준선
    대비 편차** 방식을 택한다: ISO 주차(1~53주, 예: 매년 "12월 둘째 주")가
    같은 과거 시점들의 ratio 중앙값을 그 주의 "계절적으로 기대되는 수준"으로
    삼고, 실제 ratio가 이 기준선 대비 얼마나 벗어났는지를 feature로 쓴다.
  - 미래 정보 누수 방지: 기준선은 반드시 **해당 연도보다 이전 연도**의 같은
    ISO 주차 값들로만 계산한다(같은 해 다른 주나 미래 연도는 제외).
  - 표본 신뢰성: 과거 동일 ISO 주차 관측치가 **2개 미만**이면(비교할 과거
    연도가 2개 미만이면) 기준선을 NaN으로 남긴다.
    (중간 경과: 최초 설계는 "2개 이상"이었으나, 수집 기간이 2020-12-28~
    현재로 약 5.7년뿐이던 시점(2026-10-05)에는 그 기준으로 학습 구간
    (~2022)에 유효한 계절 feature가 전혀 남지 않아 "1개 이상"으로
    한시적으로 완화했었다. 2026-10-06 네이버 데이터랩 수집 시작일을
    2016-01-01로 앞당겨 재수집한 뒤(약 10년 치 확보), 과거 연도 2개
    이상 조건을 만족하는 행이 전체의 80%(28,147/35,102)로 늘어나
    원래 설계대로 "2개 이상"으로 되돌렸다 — 중앙값이 단일 관측치가
    아니라 최소 2개 관측치 기반이 되어 안정성이 개선됐다.)
  - seasonal_deviation_pct = (ratio_t - 계절 기준선) / 계절 기준선 * 100
    (기준선 대비 몇 % 위/아래인지). 기준선이 0이면(검색 비율 자체가 0) 계산
    불가로 NaN 처리.

사용법:
  python3 compute_seasonality.py
"""

import argparse

import pandas as pd


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    df["iso_year"] = df["period"].dt.isocalendar().year
    df["iso_week"] = df["period"].dt.isocalendar().week
    return df.sort_values(["keyword", "period"])


def seasonal_baseline_for_keyword(group):
    keyword = group.name
    group = group.sort_values("period")
    baselines = []
    for _, row in group.iterrows():
        past = group[
            (group["iso_week"] == row["iso_week"]) & (group["iso_year"] < row["iso_year"])
        ]
        if len(past) >= 2:
            baselines.append(past["ratio"].median())
        else:
            baselines.append(float("nan"))
    group = group.copy()
    group["keyword"] = keyword
    group["seasonal_baseline"] = baselines
    return group


def compute_seasonality(df):
    df = df.groupby("keyword", group_keys=False).apply(seasonal_baseline_for_keyword)
    df["seasonal_deviation_pct"] = (
        (df["ratio"] - df["seasonal_baseline"]) / df["seasonal_baseline"] * 100
    )
    df.loc[df["seasonal_baseline"] == 0, "seasonal_deviation_pct"] = float("nan")
    return df.drop(columns=["iso_year", "iso_week"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_features.csv")
    parser.add_argument("--output", default="naver_trend_features.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    print(f"입력: {len(df)}행, 키워드 {df['keyword'].nunique()}개")

    df = compute_seasonality(df)

    n_available = int(df["seasonal_deviation_pct"].notna().sum())
    print(f"seasonal_deviation_pct 계산 가능 행(과거 동일 ISO주차 2회 이상): {n_available} / {len(df)}")
    print(df.groupby("category")["seasonal_deviation_pct"].agg(["count", "mean", "std"]))

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"완료: {args.output}")


if __name__ == "__main__":
    main()
