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

사용법:
  python3 check_sample_size_stability.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baseline_model import add_signal, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
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
    train, test = split_backtest(df)
    train = train.dropna(subset=FEATURE_COLS + ["is_breakout"])
    test = test.dropna(subset=FEATURE_COLS + ["is_breakout"])
    if train["is_breakout"].nunique() < 2 or len(test) == 0:
        return None

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(scaler.fit_transform(X_train), train["is_breakout"])

    test = test.copy()
    test["pred_proba"] = model.predict_proba(scaler.transform(X_test))[:, 1]
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
        {"성분": 0.659156, "제형": 0.745129, "컨셉·클레임": 0.734491, "효능": 0.825055}
    )
    print(summary.round(4))

    result.to_csv("sample_size_stability_draws.csv", index=False, encoding="utf-8-sig")
    print("\n완료: sample_size_stability_draws.csv")


if __name__ == "__main__":
    main()
