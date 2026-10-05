"""
naver_trend_labeled.csv에 검색량 변동성(분산) feature를 추가한다.
RESEARCH_PLAN.md "3) Feature 설계"에서 "검색량 변동성(분산) — 위에서
카테고리별로 선택된 N주 윈도를 그대로 사용해 계산"이라고 정리한 부분을
구현한다. 윈도 크기(N)는 select_lag_window.py가 산출한 lag_selected.csv를
그대로 읽어 카테고리별로 다르게 적용한다(성분 14주/제형 22주/
컨셉·클레임 7주/효능 24주).

정의:
  volatility_t = std(growth_rate_{t-N} .. growth_rate_{t-1})  (키워드별)
  - 현재 시점(t)의 growth_rate는 포함하지 않는다(shift(1)로 한 칸 밀어서
    계산) — is_breakout_t가 growth_rate_t로부터 정의되므로, t 시점 값을
    변동성 feature에 넣으면 라벨링 시점 T 이전 데이터만 쓴다는 원칙(미래
    정보 누수 방지)에 어긋나기 때문. select_lag_window.py에서 growth_rate를
    1주 이상 과거로 shift해서 is_breakout과 비교한 것과 동일한 원칙.
  - min_periods=N으로 둬서, 윈도가 아직 N주만큼 쌓이지 않은 키워드 초반
    구간은 NaN으로 남긴다(불완전한 윈도로 계산된 분산을 쓰지 않기 위함).

사용법:
  python3 compute_volatility.py
"""

import argparse

import pandas as pd


def load_labeled(path="naver_trend_labeled.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def load_lag_windows(path="lag_selected.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(df["category"], df["lag_weeks"]))


def compute_volatility(df, lag_by_category):
    df = df.copy()
    df["volatility_window_weeks"] = df["category"].map(lag_by_category)
    df["volatility"] = float("nan")

    for category, window in lag_by_category.items():
        mask = df["category"] == category
        vol = (
            df.loc[mask]
            .groupby("keyword")["growth_rate"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=window).std())
        )
        df.loc[mask, "volatility"] = vol.droplevel("keyword").values

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_labeled.csv")
    parser.add_argument("--lag-input", default="lag_selected.csv")
    parser.add_argument("--output", default="naver_trend_features.csv")
    args = parser.parse_args()

    df = load_labeled(args.input)
    lag_by_category = load_lag_windows(args.lag_input)
    print(f"입력: {len(df)}행, 카테고리별 윈도: {lag_by_category}")

    df = compute_volatility(df, lag_by_category)

    n_available = int(df["volatility"].notna().sum())
    print(f"volatility 계산 가능 행(윈도 충족): {n_available} / {len(df)}")
    for category in lag_by_category:
        sub = df[df["category"] == category]
        print(
            f"  {category}: 가능 {sub['volatility'].notna().sum()}/{len(sub)}, "
            f"평균 {sub['volatility'].mean():.3f}"
        )

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"완료: {args.output}")


if __name__ == "__main__":
    main()
