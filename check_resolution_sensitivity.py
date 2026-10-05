"""
5.2절 "주간 단위 한계(더 세밀한 일간 단위는 아님)"에 대한 간접 검증.

이 세션에는 네이버 API 키가 없어 일간(daily) 데이터를 새로 수집할 수는
없다(config.json 미존재). 대신 **반대 방향**으로 검증한다: 이미 가진 주간
데이터를 월 단위로 "더 거칠게" 집계했을 때 breakout 신호가 얼마나
손실되는지를 보면, "주간보다 더 세밀한 일간으로 가면 반대로 얼마나 더
포착할 수 있을지"를 가늠하는 간접적 근거가 된다 — 주간→월간에서 신호
손실이 크다면, 일간→주간에서도 비슷한 정도의 손실이 있었을 가능성을
시사한다(엄밀한 증명은 아니고 정황 근거).

방법: 키워드별 주간 ratio를 월 단위로 집계(해당 월에 속하는 주들의 평균)
→ 월간 growth_rate 계산 → 동일한 Tukey IQR(카테고리별)로 월간 breakout
라벨링 → 주간 breakout이 발생한 주가 "그 주가 속한 달"에서도 월간
breakout으로 잡히는지(재현율) 확인.

사용법:
  python3 check_resolution_sensitivity.py
"""

import pandas as pd


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def build_monthly(df):
    df = df.copy()
    df["month"] = df["period"].dt.to_period("M")
    monthly = (
        df.groupby(["keyword", "category", "month"], as_index=False)["ratio"].mean()
    )
    monthly = monthly.sort_values(["keyword", "month"])
    monthly["growth_rate"] = monthly.groupby("keyword")["ratio"].pct_change() * 100

    grouped = monthly.groupby("category")["growth_rate"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    monthly["breakout_threshold"] = q3 + 1.5 * (q3 - q1)
    monthly["is_breakout_monthly"] = monthly["growth_rate"] > monthly["breakout_threshold"]
    return monthly


def main():
    weekly = load_features()
    weekly["month"] = weekly["period"].dt.to_period("M")

    monthly = build_monthly(weekly)
    print("=== 월간 집계 breakout 비율 (카테고리별) ===")
    print(monthly.groupby("category")["is_breakout_monthly"].agg(["mean", "sum", "count"]))
    print()
    print("=== 주간 집계 breakout 비율 (비교용, 기존 결과) ===")
    print(weekly.groupby("category")["is_breakout"].agg(["mean", "sum", "count"]))
    print()

    merged = weekly.merge(
        monthly[["keyword", "month", "is_breakout_monthly"]], on=["keyword", "month"], how="left"
    )

    weekly_breakouts = merged[merged["is_breakout"] == True]  # noqa: E712
    print(f"주간 breakout으로 식별된 사건 수: {len(weekly_breakouts)}")
    recall = weekly_breakouts["is_breakout_monthly"].mean()
    print(f"그 중 '같은 달'도 월간 breakout으로 잡히는 비율(재현율): {recall:.1%}")
    print()
    print("카테고리별 재현율:")
    print(weekly_breakouts.groupby("category")["is_breakout_monthly"].mean())

    print()
    print(f"=> 주간 breakout 사건의 {1 - recall:.1%}는 월 단위로 집계하면 '보이지 않게' 된다.")
    print("   (주간보다 더 세밀한 일간으로 갔을 때 반대로 얼마나 더 포착될 수 있는지에 대한 정황적 하한선)")


if __name__ == "__main__":
    main()
