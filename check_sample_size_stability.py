"""
5.2절 "표본 규모의 한계"(카테고리당 키워드 11~22개)에 대한 검증.

가설: 카테고리당 키워드 수가 적어서(11~22개) 모델 성능(AUC)이 어떤 키워드가
표본에 포함되는지에 따라 크게 흔들릴 것이다 — 즉 "표본을 늘리면 더 안정적인
결과를 얻을 수 있다"는 5.2절 주장의 근거가 되는지 확인.

방법: 카테고리별로 키워드의 80%를 무작위로 뽑아(20회 반복, seed 고정) 같은
breakout 라벨링·full 모델 파이프라인을 재학습시키고, 20회 AUC의 표준편차를
측정한다. 표준편차가 크면(=뽑히는 키워드에 따라 성능이 크게 출렁이면) 표본
규모 한계 주장이 실증적으로 뒷받침되고, 작으면 그 한계가 과장된 것일 수
있다는 뜻이다.

2026-10-06: 주 모델을 로지스틱회귀에서 XGBoost로 전환(§4.1)한 데 맞춰, 이
스크립트도 XGBoost로 안정성을 검증하도록 변경했었다. 2026-10-08: 주 모델이
다시 CatBoost+모멘텀+카테고리 내 순위(§4.1)로 바뀜에 따라 동일하게 갱신.

사용법:
  python3 check_sample_size_stability.py
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from baseline_model import add_signal, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix, drop_missing
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]
N_DRAWS = 20
SAMPLE_FRACTION = 0.8
SEED = 42


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def relabel_breakouts(df):
    df = df.sort_values(["keyword", "period"]).copy()
    df["growth_rate"] = df.groupby("keyword")["ratio"].pct_change() * 100
    grouped = df.groupby("category")["growth_rate"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    df["breakout_threshold"] = q3 + 1.5 * (q3 - q1)
    df["is_breakout"] = df["growth_rate"] > df["breakout_threshold"]
    return df


def recompute_volatility(df, lag_by_category):
    df = df.copy()
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


def fit_auc(df, lag_by_category):
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)
    train, test = split_backtest(df)
    train = drop_missing(train, FEATURE_COLS).dropna(subset=["momentum", "rank_in_category"])
    test = drop_missing(test, FEATURE_COLS).dropna(subset=["momentum", "rank_in_category"])
    if train["is_breakout"].nunique() < 2 or len(test) == 0:
        return None

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)

    y_train = train["is_breakout"].astype(int)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    model = CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)

    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return auc_by_category(test)


def main():
    raw = load_features()
    lag_by_category = load_lag_windows()
    rng = np.random.default_rng(SEED)

    keywords_by_category = raw.groupby("category")["keyword"].unique().to_dict()

    draws = []
    for i in range(N_DRAWS):
        sampled_keywords = []
        for category, kws in keywords_by_category.items():
            n_sample = max(2, round(len(kws) * SAMPLE_FRACTION))
            sampled_keywords.extend(rng.choice(kws, size=n_sample, replace=False))

        subset = raw[raw["keyword"].isin(sampled_keywords)].copy()
        df = relabel_breakouts(subset)
        df = recompute_volatility(df, lag_by_category)
        auc = fit_auc(df, lag_by_category)
        if auc is not None:
            draws.append(auc)

    result = pd.DataFrame(draws)
    print(f"{N_DRAWS}회 중 평가 가능했던 횟수: {len(draws)}")
    print(f"매 회 카테고리별 키워드의 {SAMPLE_FRACTION:.0%}를 무작위로 뽑아 재학습 (전체 키워드 사용 시 AUC와 비교)")
    print()
    summary = result.agg(["mean", "std", "min", "max"]).T
    summary["full_sample_auc"] = pd.Series(
        {"성분": 0.716078, "제형": 0.763684, "컨셉·클레임": 0.825339, "효능": 0.869127}
    )
    print(summary.round(4))

    result.to_csv("sample_size_stability_draws.csv", index=False, encoding="utf-8-sig")
    print("\n완료: sample_size_stability_draws.csv")


if __name__ == "__main__":
    main()
