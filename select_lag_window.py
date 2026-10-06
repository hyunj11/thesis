"""
naver_trend_labeled.csv를 바탕으로 feature 설계에 쓸 lag window(N주)를
서주연(2018, 이화여대 ARDL 논문)의 교차상관(cross-correlation) 기반 최적
시차 선택 방법으로 데이터 기반 결정한다. RESEARCH_PLAN.md "3) Feature 설계"
절에 정리된 방법론을 그대로 구현한다.

방법론 요약 (근거는 RESEARCH_PLAN.md 참고):
  - "최근 N주 검색량 증가율" feature의 N을 임의로 정하지 않고, 각 후보 시차
    L(1~26주, label_breakouts.py와 동일한 카테고리 그룹화를 적용)에서
    growth_rate를 L주 과거로 이동(shift)시킨 값과 breakout 라벨
    (is_breakout) 간의 상관관계를 계산해, 상관관계 절댓값이 가장 큰 시차를
    "가장 설명력 높은 시차"로 선택한다.
  - 미래 정보 누수 방지: growth_rate_{t-L} (L>=1, 반드시 과거 시점)과
    is_breakout_t (현재 시점의 라벨)를 비교한다. 같은 시점(L=0)은 라벨
    자체가 growth_rate로부터 정의되므로 순환 논리가 되어 제외.
  - 카테고리(성분/컨셉·클레임/제형/효능)별로 따로 계산하는 이유: 라벨링
    단계(Tukey IQR)와 동일하게 RQ2(카테고리 간 차이) 설계를 따름 — "가장
    설명력 높은 시차"도 카테고리마다 다를 수 있다고 봄.
  - 26주 상한: RESEARCH_PLAN.md에서 기존 "1~N개월" 탐색 범위를 주 단위로
    환산한 값(약 6개월)을 그대로 사용.

사용법:
  python3 select_lag_window.py
  python3 select_lag_window.py --max-lag 12

결과:
  - lag_crosscorr.csv  : 카테고리 x 후보 시차(1~max-lag)별 상관계수 전체 표
  - lag_selected.csv   : 카테고리별 최적 시차(|상관계수| 최댓값) 선택 결과
"""

import argparse

import numpy as np
import pandas as pd

MAX_LAG_WEEKS = 26  # RESEARCH_PLAN.md 3) Feature 설계: 1~26주(약 6개월) 탐색 범위


def load_labeled(path="naver_trend_labeled.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    df["is_breakout"] = df["is_breakout"].astype(int)
    return df.sort_values(["category", "keyword", "period"])


def cross_correlation_by_lag(df, max_lag):
    rows = []
    for category, cat_df in df.groupby("category"):
        target = cat_df["is_breakout"]
        for lag in range(1, max_lag + 1):
            lagged = cat_df.groupby("keyword")["growth_rate"].shift(lag)
            valid = lagged.notna() & target.notna()
            n = int(valid.sum())
            corr = lagged[valid].corr(target[valid]) if n >= 2 else np.nan
            rows.append(
                {
                    "category": category,
                    "lag_weeks": lag,
                    "n": n,
                    "correlation": corr,
                }
            )
    return pd.DataFrame(rows)


def select_best_lag(corr_df):
    df = corr_df.dropna(subset=["correlation"]).copy()
    df["abs_correlation"] = df["correlation"].abs()
    idx = df.groupby("category")["abs_correlation"].idxmax()
    best = df.loc[idx].drop(columns="abs_correlation").reset_index(drop=True)
    return best.sort_values("category")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_labeled.csv")
    parser.add_argument("--corr-output", default="lag_crosscorr.csv")
    parser.add_argument("--selected-output", default="lag_selected.csv")
    parser.add_argument("--max-lag", type=int, default=MAX_LAG_WEEKS)
    args = parser.parse_args()

    df = load_labeled(args.input)
    print(f"입력: {len(df)}행, 카테고리 {df['category'].nunique()}개")

    corr_df = cross_correlation_by_lag(df, args.max_lag)
    corr_df.to_csv(args.corr_output, index=False, encoding="utf-8-sig")
    print(f"시차별 상관계수 전체 표: {args.corr_output}")

    best = select_best_lag(corr_df)
    best.to_csv(args.selected_output, index=False, encoding="utf-8-sig")

    print("\n카테고리별 선택된 lag window (|상관계수| 최댓값 기준):")
    print(best.to_string(index=False))
    print(f"\n완료: {args.selected_output}")


if __name__ == "__main__":
    main()
