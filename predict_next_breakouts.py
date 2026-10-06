"""
현재 시점 기준으로 "다음에 뜰 것 같은" 키워드를 순위화하는 탐색적 스크립트.

RESEARCH_PLAN.md/THESIS_DRAFT.md의 백테스팅(학습 ~2022, 테스트 2023~2025)과는
다른 목적: 논문 본문에 넣는 "검증된 성능 수치"가 아니라, 현재 가용한 모든
라벨 데이터로 `full` 로지스틱회귀를 재학습한 뒤, 가장 최근 완전한 주
(2026-09-21)의 feature로 각 키워드의 breakout 확률을 예측해 순위를 매긴다.

⚠️ 해석 주의:
  - signal(지연된 growth_rate)·volatility·seasonal_deviation_pct_lag1 세
    feature의 조합으로 나온 확률이라, "이번 주 증가율이 높은 키워드"와는
    다르다 — 특히 volatility(최근 N주간 증가율의 표준편차)의 회귀계수가
    가장 크기 때문에, 최근 변동폭이 큰 키워드가 상위로 올라오는 경향이
    있다(상승이든 하락이든 변동 자체가 큰 경우를 포함).
  - 본 모델의 실제 백테스팅 성능(Precision@1)은 카테고리별로 0.10~0.31
    수준에 불과하다(logistic_regression_results.csv 참고) — 이 순위를
    확정적 예측이 아니라 "스크리닝 후보 목록" 정도로만 활용할 것.

사용법:
  python3 predict_next_breakouts.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baseline_model import load_features, load_lag_windows, add_signal
from logistic_regression_model import add_lagged_seasonal, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]


def main():
    df = load_features("naver_trend_features.csv")
    lag_by_category = load_lag_windows("lag_selected.csv")
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)

    train = df.dropna(subset=FEATURE_COLS + ["is_breakout"])
    X_train = build_design_matrix(train, FEATURE_COLS)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, train["is_breakout"])

    latest_period = df["period"].max()
    latest = df[df["period"] == latest_period].dropna(subset=FEATURE_COLS).copy()
    X_latest = build_design_matrix(latest, FEATURE_COLS).reindex(
        columns=X_train.columns, fill_value=0
    )
    latest["pred_proba"] = model.predict_proba(scaler.transform(X_latest))[:, 1]

    result = latest[
        ["keyword", "category", "ratio", "growth_rate", "is_breakout", "pred_proba"]
    ].sort_values("pred_proba", ascending=False)

    print(f"기준 시점(가장 최근 완전한 주): {latest_period.date()}")
    print(f"예측 대상 키워드 수: {len(result)}")
    print()
    print("전체 상위 15개:")
    print(result.head(15).to_string(index=False))
    print()
    print("카테고리별 상위 3개:")
    for category, group in result.groupby("category"):
        print(f"\n[{category}]")
        print(group.head(3).to_string(index=False))

    result.to_csv("next_breakout_candidates.csv", index=False, encoding="utf-8-sig")
    print("\n완료: next_breakout_candidates.csv")


if __name__ == "__main__":
    main()
