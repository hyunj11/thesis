"""
RQ3 1단계: 미국 Google Trends breakout이 한국 네이버 breakout보다 먼저
발생하는 "해외 선행 신호"가 실제로 존재하는지 검증한다(성분 22개 한정).

방법:
  1. 미국 쪽: google_trends.csv(월간, 2016-01~2026-09)에 Tukey IQR
     breakout 라벨링을 적용한다(기존 label_breakouts.py와 동일한 공식,
     성분 하나뿐이므로 카테고리 그룹화는 불필요).
  2. 한국 쪽: naver_trend_features.csv(주간)을 월별로 집계(해당 월에
     속한 주의 ratio 평균 — check_resolution_sensitivity.py와 동일한
     방식)한 뒤 같은 Tukey IQR로 월간 breakout을 라벨링한다. 기존 주간
     라벨을 그대로 쓰지 않는 이유는 미국 쪽과 동일한 시간 단위(월)로
     맞춰야 시차를 "개월" 단위로 정직하게 비교할 수 있기 때문이다.
  3. 키워드별로 각 한국 breakout 월에 대해, 그 이전 N개월(기본 24개월)
     안에 미국 breakout이 있었는지 찾고, 있다면 가장 가까운 것과의 시차
     (개월)를 기록한다. 반대 방향(한국이 먼저 뜬 경우)도 같은 방식으로
     측정해 비교한다.

해석 기준: "미국이 선행" 사례가 "한국이 선행" 사례보다 뚜렷이 많고, 시차가
몇 개월~1년 내의 일관된 범위에 몰려 있다면 해외 선행 신호 가설을 지지하는
근거가 된다. 사례가 드물거나 양방향이 비슷하면 가설은 기각된다.

사용법:
  python3 check_us_lead_lag.py
"""

import pandas as pd

MAX_LOOKBACK_MONTHS = 24


def label_breakouts_monthly(df, value_col="ratio"):
    df = df.sort_values(["keyword", "month"]).copy()
    df["growth_rate"] = df.groupby("keyword")[value_col].pct_change() * 100
    q1 = df["growth_rate"].quantile(0.25)
    q3 = df["growth_rate"].quantile(0.75)
    threshold = q3 + 1.5 * (q3 - q1)
    df["is_breakout"] = df["growth_rate"] > threshold
    return df, threshold


def load_us():
    df = pd.read_csv("google_trends.csv", encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    df["month"] = df["period"].dt.to_period("M")
    return df


def load_kr_monthly():
    df = pd.read_csv("naver_trend_features.csv", encoding="utf-8-sig")
    df = df[df["category"] == "성분"].copy()
    df["period"] = pd.to_datetime(df["period"])
    df["month"] = df["period"].dt.to_period("M")
    monthly = df.groupby(["keyword", "month"], as_index=False)["ratio"].mean()
    return monthly


def nearest_lag(target_month, candidate_months, max_lookback):
    """target_month 이전(또는 이후) candidate_months 중 가장 가까운 것과의 시차(개월, 양수=선행)."""
    best = None
    for cm in candidate_months:
        diff = (target_month - cm).n  # target - candidate; 양수면 candidate가 먼저
        if 0 < diff <= max_lookback:
            if best is None or diff < best:
                best = diff
    return best


def main():
    us = load_us()
    us, us_threshold = label_breakouts_monthly(us)
    kr = load_kr_monthly()
    kr, kr_threshold = label_breakouts_monthly(kr)

    print(f"US breakout threshold(growth_rate %): {us_threshold:.1f}")
    print(f"KR(월간 집계) breakout threshold(growth_rate %): {kr_threshold:.1f}")
    print(f"US breakout 발생 건수: {int(us['is_breakout'].sum())} / {len(us)}")
    print(f"KR(월간) breakout 발생 건수: {int(kr['is_breakout'].sum())} / {len(kr)}")
    print()

    us_events = us[us["is_breakout"]][["keyword", "month"]]
    kr_events = kr[kr["is_breakout"]][["keyword", "month"]]

    rows = []
    for keyword in sorted(set(us["keyword"]) | set(kr["keyword"])):
        us_months = us_events[us_events["keyword"] == keyword]["month"].tolist()
        kr_months = kr_events[kr_events["keyword"] == keyword]["month"].tolist()

        for km in kr_months:
            lag = nearest_lag(km, us_months, MAX_LOOKBACK_MONTHS)
            rows.append({"keyword": keyword, "direction": "US_leads_KR", "kr_month": str(km), "lag_months": lag})

        for um in us_months:
            lag = nearest_lag(um, kr_months, MAX_LOOKBACK_MONTHS)
            rows.append({"keyword": keyword, "direction": "KR_leads_US", "us_month": str(um), "lag_months": lag})

    result = pd.DataFrame(rows)
    result.to_csv("us_lead_lag_events.csv", index=False, encoding="utf-8-sig")

    us_leads = result[(result["direction"] == "US_leads_KR") & result["lag_months"].notna()]
    kr_leads = result[(result["direction"] == "KR_leads_US") & result["lag_months"].notna()]

    print(f"=== 'US가 먼저 뜨고 KR이 나중에 뜬' 사례 (한국 breakout {len(kr_events)}건 중) ===")
    print(f"  매칭된 사례 수: {len(us_leads)} / {len(kr_events)} ({len(us_leads) / max(len(kr_events), 1):.1%})")
    if len(us_leads) > 0:
        print(f"  시차(개월) 분포: {us_leads['lag_months'].describe()[['min', '25%', '50%', '75%', 'max']].to_dict()}")
        print(us_leads.sort_values("lag_months").to_string(index=False))
    print()

    print(f"=== 'KR이 먼저 뜨고 US가 나중에 뜬' 사례 (미국 breakout {len(us_events)}건 중) ===")
    print(f"  매칭된 사례 수: {len(kr_leads)} / {len(us_events)} ({len(kr_leads) / max(len(us_events), 1):.1%})")
    if len(kr_leads) > 0:
        print(f"  시차(개월) 분포: {kr_leads['lag_months'].describe()[['min', '25%', '50%', '75%', 'max']].to_dict()}")

    print("\n완료: us_lead_lag_events.csv")


if __name__ == "__main__":
    main()
