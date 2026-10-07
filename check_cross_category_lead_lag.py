"""
RQ2를 확장하는 새 분석: "한 카테고리의 breakout이 다른 카테고리의 breakout보다
먼저 일어나는가?"(카테고리 간 확산/전이 순서) — 사용자 피드백("변동성 발견은
화장품에 국한된 차별점이 아니다")에 따라, 이 논문이 유일하게 가진 구조(성분/
제형/컨셉·클레임/효능 4개 의미론적 카테고리를 동시 추적)를 실제로 활용하는
분석으로 신규 설계했다.

가설(업계 직관): 신성분이 먼저 유행하고, 그 성분을 내세운 마케팅 문구(컨셉·
클레임)나 효능 주장이 뒤따라 유행한다 — 즉 성분 → 효능/컨셉·클레임 방향의
확산 지연(diffusion lag)이 있을 수 있다. 반대 방향(마케팅 유행이 성분보다
먼저)도 이론적으로 가능하므로 양방향 모두 검증한다.

방법(기존 검증된 기법 재사용, 새 통계 기법 도입 없음):
  1. 카테고리별 주간 breakout 강도 = 그 주에 breakout한 키워드 수 / 카테고리
     전체 키워드 수(비율). 단순 "이번 주에 하나라도 떴는가"(이진)는 키워드를
     풀링하면 주당 54~69%가 True로 나와 포화(saturated)돼 선후관계를 가릴
     신호가 거의 없음을 사전 확인(한 카테고리 내 여러 키워드를 모아 집계하면
     거의 매주 breakout이 있기 때문) — 그래서 이진 대신 강도(비율)를 쓴다.
  2. select_lag_window.py와 동일한 교차상관(cross-correlation) 기법을 카테고리
     쌍(12개 순서쌍)에 적용해, lag -26~+26주 구간에서 상관계수가 최대인 lag를
     찾는다. 양의 lag는 "A가 B보다 먼저"를 의미(A(t)와 B(t+lag)의 상관).
  3. 유의성 검정: 각 카테고리 시계열 자체의 자기상관 구조를 보존하기 위해
     순환 이동(circular shift, 임의 오프셋 500회)으로 귀무분포를 만들고,
     관측된 최대|상관|이 귀무분포 상위 몇 %에 해당하는지로 p-value를 추정한다.

사용법:
  python3 check_cross_category_lead_lag.py
"""

import numpy as np
import pandas as pd

MAX_LAG_WEEKS = 26
N_PERMUTATIONS = 500
RNG_SEED = 42


def load_category_rate_series():
    df = pd.read_csv("naver_trend_features.csv", encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])

    n_keywords = df.groupby("category")["keyword"].nunique()
    weekly_count = df.groupby(["category", "period"])["is_breakout"].sum().unstack("category").fillna(0)
    weekly_rate = weekly_count.div(n_keywords, axis=1)
    return weekly_rate.sort_index()


def cross_corr_at_lag(series_a, series_b, lag):
    # lag > 0: a가 b보다 "lag주 먼저" -> a(t)와 b(t+lag) 비교 -> a를 lag만큼 앞으로 당겨서 b와 맞춤
    if lag >= 0:
        a = series_a[: len(series_a) - lag] if lag > 0 else series_a
        b = series_b[lag:]
    else:
        a = series_a[-lag:]
        b = series_b[: len(series_b) + lag]
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def best_lag(series_a, series_b, max_lag):
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        corr = cross_corr_at_lag(series_a.values, series_b.values, lag)
        rows.append((lag, corr))
    table = pd.DataFrame(rows, columns=["lag", "corr"]).dropna()
    best = table.loc[table["corr"].abs().idxmax()]
    return best["lag"], best["corr"], table


def permutation_pvalue(series_a, series_b, max_lag, observed_abs_corr, n_perm, rng):
    n = len(series_a)
    null_max = np.empty(n_perm)
    a_vals = series_a.values
    for i in range(n_perm):
        shift = rng.integers(1, n - 1)
        shifted_a = np.roll(a_vals, shift)
        corrs = [cross_corr_at_lag(shifted_a, series_b.values, lag) for lag in range(-max_lag, max_lag + 1)]
        corrs = [c for c in corrs if not np.isnan(c)]
        null_max[i] = max(abs(c) for c in corrs)
    p_value = (null_max >= observed_abs_corr).mean()
    return p_value, null_max


def main():
    rate = load_category_rate_series()
    categories = list(rate.columns)
    print(f"카테고리별 주간 breakout 비율 시계열 로드 완료: {rate.shape[0]}주 x {len(categories)}개 카테고리")
    print(rate.describe().round(4))

    rng = np.random.default_rng(RNG_SEED)
    results = []
    tested_pairs = set()
    for a in categories:
        for b in categories:
            if a == b:
                continue
            lag, corr, table = best_lag(rate[a], rate[b], MAX_LAG_WEEKS)
            pair_key = tuple(sorted([a, b]))
            # 양방향(A->B, B->A)을 각각 독립적으로 검정 — 대칭 쌍이라도 순열검정은 매번 새로 함
            p_value, _ = permutation_pvalue(rate[a], rate[b], MAX_LAG_WEEKS, abs(corr), N_PERMUTATIONS, rng)
            results.append(
                {
                    "leader": a,
                    "follower": b,
                    "best_lag_weeks": lag,
                    "correlation": round(corr, 4),
                    "p_value": round(p_value, 4),
                }
            )
            tested_pairs.add(pair_key)

    result_df = pd.DataFrame(results).sort_values("p_value")
    print("\n=== 카테고리 간 선행-후행 교차상관 결과 (12개 순서쌍, p-value 오름차순) ===")
    print(result_df.to_string(index=False))

    result_df.to_csv("cross_category_lead_lag.csv", index=False, encoding="utf-8-sig")

    # 모든 쌍에서 최적 lag가 0으로 쏠린 것을 확인한 뒤, "동시 급등"이 특정
    # 계절(연초/추석 등)에 몰리는지 사후 검증 — 카테고리 간 확산이 아니라
    # 공통 외부 요인(계절성)에 의한 동시 breakout일 가능성을 점검.
    rate["total"] = rate.sum(axis=1)
    threshold = rate["total"].quantile(0.9)
    high = rate[rate["total"] >= threshold]
    target_months = {1, 2, 9, 10}  # 연초(설 포함)·추석/환절기
    in_target_high = high.index.month.isin(target_months).mean()
    in_target_base = rate.index.month.isin(target_months).mean()
    print(f"\n=== 동시다발 급등 주(상위 10%, n={len(high)})의 월별 쏠림 ===")
    print(f"  1/2/9/10월(연초·설날·추석/환절기) 비율: {in_target_high:.1%} (전체 561주 기준선: {in_target_base:.1%})")
    print(high["total"].describe())

    print("\n완료: cross_category_lead_lag.csv")


if __name__ == "__main__":
    main()
