"""
naver_trends.csv를 바탕으로 Tukey IQR 이상치 탐지법으로 breakout 라벨을
생성한다. RESEARCH_PLAN.md "2) Breakout 라벨링 기준"에 정리된 방법론을
그대로 구현한다.

방법론 요약 (근거는 RESEARCH_PLAN.md 참고):
  1. 완전히 끝나지 않은 마지막 주(들)는 분석에서 제외한다.
     - 네이버 데이터랩 주간 데이터는 각 period가 그 주의 시작일(월요일)이며,
       period + 6일이 그 주의 종료일이다. 종료일이 아직 안 지났으면 그 주는
       부분 집계된 값이라 분석에서 제외한다("비례 보정" 대신 "완전히 끝난
       기간만 사용"하는 규칙 — 요일별 검색량 패턴이 불균등해 비례 보정은
       편향을 만들 수 있다고 판단).
  2. growth_rate = 키워드별 전주 대비 증가율(pct_change, %)
     - 원 수준값(level) 대신 증가율(차분)을 쓰는 이유: 가성회귀
       (spurious regression, Yule 1926) 위험 회피.
  3. 카테고리(성분/컨셉·클레임/제형/효능)별로 그룹화해 Q1/Q3/IQR을 따로
     계산하고, growth_rate > Q3 + 1.5*IQR 인 시점을 breakout(1)으로 라벨링.

주의: ratio 값은 API를 호출한 "수집 시점" 기준으로 고정되어 저장된다. 즉
  마지막 주가 수집 당시 완전히 끝나지 않았다면 그 값은 이미 부분 집계된 채로
  파일에 박제돼 있고, 스크립트를 나중에 다시 돌린다고 저절로 갱신되지 않는다.
  따라서 "완전히 끝난 주"인지 판단하는 기준(as_of)은 스크립트 실행 시점의
  오늘 날짜가 아니라 데이터 수집 시점이어야 한다. 기본값은 입력 CSV 파일의
  수정 시각(mtime, 즉 collect_naver_trends.py를 마지막으로 실행한 시점)을
  사용한다. 데이터를 재수집(re-fetch)했다면 파일이 새로 쓰여지므로 mtime도
  자동으로 갱신된다.

사용법:
  python3 label_breakouts.py
  python3 label_breakouts.py --as-of 2026-09-29  (수집 시점을 수동으로 지정하고
    싶을 때. 기본값은 입력 CSV의 파일 수정 시각)
"""

import argparse
import os
from datetime import datetime

import pandas as pd


def load_trends(path="naver_trends.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df


def drop_incomplete_periods(df, as_of):
    period_length = pd.Timedelta(days=7)
    period_end = df["period"] + period_length - pd.Timedelta(days=1)
    complete = period_end < as_of
    dropped = df.loc[~complete, "period"].unique()
    if len(dropped) > 0:
        dropped_str = ", ".join(sorted(d.strftime("%Y-%m-%d") for d in dropped))
        print(f"완전히 끝나지 않은 주 제외: {dropped_str}")
    return df.loc[complete].copy()


def compute_growth_rate(df):
    df = df.sort_values(["keyword", "period"])
    df["growth_rate"] = df.groupby("keyword")["ratio"].pct_change() * 100
    return df


def flag_breakouts(df):
    grouped = df.groupby("category")["growth_rate"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df["breakout_threshold"] = q3 + 1.5 * iqr
    df["is_breakout"] = df["growth_rate"] > df["breakout_threshold"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trends.csv")
    parser.add_argument("--output", default="naver_trend_labeled.csv")
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "YYYY-MM-DD. 이 날짜를 기준으로 '완전히 끝난 주'만 사용. "
            "기본값은 입력 CSV 파일의 수정 시각(=데이터 수집 시점)."
        ),
    )
    args = parser.parse_args()

    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    else:
        mtime = os.path.getmtime(args.input)
        as_of = pd.Timestamp(datetime.fromtimestamp(mtime).date())
        print(f"as_of 미지정: 입력 파일 수정 시각({as_of.date()})을 수집 시점으로 사용")

    df = load_trends(args.input)
    print(f"원본: {len(df)}행, 키워드 {df['keyword'].nunique()}개")

    df = drop_incomplete_periods(df, as_of)
    print(f"완전한 주만 남긴 뒤: {len(df)}행")

    df = compute_growth_rate(df)
    df = flag_breakouts(df)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    n_breakout = int(df["is_breakout"].sum())
    print(f"breakout 라벨 1(양성) 개수: {n_breakout} / {df['growth_rate'].notna().sum()}")
    print(f"완료: {args.output}")


if __name__ == "__main__":
    main()
