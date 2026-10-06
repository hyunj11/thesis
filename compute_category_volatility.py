"""
RESEARCH_PLAN.md "5) 검증(백테스팅)"의 "카테고리별(성분/컨셉/제형/효능)
변동성 지표(변동계수, peak 대비 baseline 비율, half-life)도 함께 분석해
RQ2 뒷받침" 항목을 구현한다. 세 지표 모두 RESEARCH_PLAN.md에 이미
결정·명시된 지표이며, 이 스크립트는 그 정의를 데이터에 적용하는 구현이다.

지표 정의:
  1. 변동계수(CV, Coefficient of Variation) = std(ratio) / mean(ratio)
     - 단위에 무관한 상대적 변동성 지표(표준편차를 평균으로 나눔). 절대
       검색량 규모가 다른 키워드끼리도 공정하게 비교 가능(이미 breakout
       라벨링에서 "절대 검색량이 아닌 상대적 증가율" 원칙을 쓴 것과 같은
       이유).
     - 키워드별 전체 수집 기간(ratio 시계열 전체)을 대상으로 계산.
  2. peak 대비 baseline 비율 = max(ratio) / median(ratio)
     - median을 "평상시 수준(baseline)"으로 삼는 이유: compute_seasonality.py
       에서 계절 기준선도 동일하게 median을 사용했음(평균보다 이상치에
       강건) — 같은 원칙을 재사용.
  3. half-life(반감기) = breakout이 발생한 시점의 ratio(peak)가 그 절반
     이하로 떨어지기까지 걸린 주 수.
     - breakout 시점 정의는 label_breakouts.py가 만든 is_breakout 라벨을
       그대로 사용(= Tukey IQR 기준 초과 시점). 그 주의 ratio를 peak으로
       보고, 이후 주들을 순서대로 보며 ratio가 peak/2 이하로 처음
       떨어지는 시점까지 걸린 주 수를 계산.
     - 중도절단(censoring): 수집 데이터가 끝날 때까지 peak/2 이하로
       떨어지지 않으면(트렌드가 아직 안 끝났거나 관측 기간 부족) half-life를
       관측할 수 없으므로 NaN 처리하고 censored=True로 표시한다. 카테고리별
       집계에는 censored가 아닌 사례만 포함하고, censored 비율도 함께
       보고해 해석에 참고하게 한다.

사용법:
  python3 compute_category_volatility.py
"""

import argparse

import pandas as pd


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["keyword", "period"])


def compute_cv_and_peak_ratio(df):
    rows = []
    for (keyword, category), group in df.groupby(["keyword", "category"]):
        ratio = group["ratio"]
        mean = ratio.mean()
        baseline = ratio.median()
        rows.append(
            {
                "keyword": keyword,
                "category": category,
                "cv": ratio.std() / mean if mean else float("nan"),
                "peak_to_baseline": ratio.max() / baseline if baseline else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_halflife_events(df, max_horizon_weeks=52):
    events = []
    for keyword, group in df.groupby("keyword"):
        group = group.sort_values("period").reset_index(drop=True)
        category = group["category"].iloc[0]
        breakout_idx = group.index[group["is_breakout"] == True]  # noqa: E712
        for idx in breakout_idx:
            peak_ratio = group.loc[idx, "ratio"]
            peak_period = group.loc[idx, "period"]
            if peak_ratio <= 0:
                continue
            target = peak_ratio / 2
            future = group.loc[idx + 1 : idx + max_horizon_weeks]
            decayed = future[future["ratio"] <= target]
            if len(decayed) > 0:
                weeks = int(decayed.index[0] - idx)
                censored = False
            else:
                # 관측 가능한 구간(최대 max_horizon_weeks 또는 데이터 끝까지) 안에서
                # peak/2 이하로 떨어지는 시점을 못 찾음 → 중도절단(censored)
                weeks = float("nan")
                censored = True
            events.append(
                {
                    "keyword": keyword,
                    "category": category,
                    "breakout_period": peak_period,
                    "peak_ratio": peak_ratio,
                    "halflife_weeks": weeks,
                    "censored": censored,
                }
            )
    return pd.DataFrame(events)


def summarize_by_category(cv_peak_df, halflife_df):
    cv_summary = cv_peak_df.groupby("category")[["cv", "peak_to_baseline"]].mean()

    hl_summary = []
    for category, group in halflife_df.groupby("category"):
        observed = group.dropna(subset=["halflife_weeks"])
        hl_summary.append(
            {
                "category": category,
                "n_breakout_events": len(group),
                "n_censored": int(group["censored"].sum()),
                "censored_rate": group["censored"].mean(),
                "halflife_weeks_median": observed["halflife_weeks"].median(),
            }
        )
    hl_summary = pd.DataFrame(hl_summary).set_index("category")

    return cv_summary.join(hl_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_features.csv")
    parser.add_argument("--max-horizon-weeks", type=int, default=52)
    parser.add_argument("--keyword-output", default="category_volatility_keyword.csv")
    parser.add_argument("--halflife-output", default="category_volatility_halflife_events.csv")
    parser.add_argument("--summary-output", default="category_volatility_summary.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    print(f"입력: {len(df)}행, 키워드 {df['keyword'].nunique()}개")

    cv_peak_df = compute_cv_and_peak_ratio(df)
    cv_peak_df.to_csv(args.keyword_output, index=False, encoding="utf-8-sig")

    halflife_df = compute_halflife_events(df, args.max_horizon_weeks)
    halflife_df.to_csv(args.halflife_output, index=False, encoding="utf-8-sig")
    print(f"breakout 이벤트 수: {len(halflife_df)}, 그 중 censored: {int(halflife_df['censored'].sum())}")

    summary = summarize_by_category(cv_peak_df, halflife_df)
    summary.to_csv(args.summary_output, encoding="utf-8-sig")

    print("\n카테고리별 변동성 지표 요약:")
    print(summary.round(3).to_string())

    print(f"\n완료: {args.keyword_output}, {args.halflife_output}, {args.summary_output}")


if __name__ == "__main__":
    main()
