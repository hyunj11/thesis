"""
RQ1 예측 성능을 더 끌어올릴 방법 탐색 (5) — "동료 대비 초과분" feature.

§4.6에서 발견한 "카테고리 전체가 동시에 들썩인다"는 패턴을 feature로 직접
쓰려는 시도. days_to_holiday는 같은 주·같은 카테고리의 모든 키워드가
동일한 값을 가져 실패했으므로(§4.1), 이번에는 "카테고리 전체"가 아니라
"자기 자신을 제외한 동료들의 평균"과 "자신의 값"의 차이를 feature로 써서
키워드마다 다른 값이 나오도록 설계한다.

새 feature:
  - excess_signal   = signal - (같은 주·같은 카테고리에서 자신을 제외한 나머지의 signal 평균)
  - excess_momentum = momentum - (같은 주·같은 카테고리에서 자신을 제외한 나머지의 momentum 평균)

이미 있는 rank_in_category(동료 대비 백분위 "순위")와 유사한 정보를 담을
가능성이 있어(둘 다 "동료 대비 상대적 위치"를 본다는 점에서), 추가로 얻는
정보가 적을 수 있다는 점을 감안하고 실제로 AUC/Precision@K가 개선되는지
검증한다.

사용법:
  python3 check_excess_signal_performance.py
"""

import pandas as pd
from catboost import CatBoostClassifier

from baseline_model import add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix, drop_missing, precision_recall_at_k
from model_comparison_experiment import add_extra_features

BASE_FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]
NEW_FEATURE_COLS = BASE_FEATURE_COLS + ["excess_signal", "excess_momentum"]


def add_excess_features(df):
    df = df.copy()
    grp = df.groupby(["category", "period"])
    n = grp["signal"].transform("count")
    sum_signal = grp["signal"].transform("sum")
    sum_momentum = grp["momentum"].transform("sum")
    # 자신을 제외한 나머지의 평균 = (전체 합 - 자신의 값) / (개수 - 1)
    peer_mean_signal = (sum_signal - df["signal"]) / (n - 1)
    peer_mean_momentum = (sum_momentum - df["momentum"]) / (n - 1)
    df["excess_signal"] = df["signal"] - peer_mean_signal
    df["excess_momentum"] = df["momentum"] - peer_mean_momentum
    df.loc[n <= 1, ["excess_signal", "excess_momentum"]] = float("nan")
    return df


def fit_eval(train, test, feature_cols, extra_dropna_cols):
    train = drop_missing(train, [c for c in feature_cols if c in BASE_FEATURE_COLS]).dropna(subset=extra_dropna_cols)
    test = drop_missing(test, [c for c in feature_cols if c in BASE_FEATURE_COLS]).dropna(subset=extra_dropna_cols)

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test, feature_cols).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05,
                                scale_pos_weight=spw, random_state=42, verbose=False)
    model.fit(X_train, y_train)
    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return test, len(train)


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)
    df = add_excess_features(df)

    train, test = split_backtest(df)

    print("=== (A) 기존 최종 모델(signal/volatility/seasonal/momentum/rank_in_category) ===")
    test_a, n_tr_a = fit_eval(train, test, BASE_FEATURE_COLS, ["momentum", "rank_in_category"])
    auc_a = auc_by_category(test_a)
    pr_a = precision_recall_at_k(test_a, 1)
    print(auc_a.round(4))
    print(f"평균 AUC: {auc_a.mean():.4f}, 평균 Precision@1: {pr_a['precision_at_k'].mean():.4f}")

    print("\n=== (B) + excess_signal, excess_momentum 추가 ===")
    test_b, n_tr_b = fit_eval(train, test, NEW_FEATURE_COLS,
                               ["momentum", "rank_in_category", "excess_signal", "excess_momentum"])
    auc_b = auc_by_category(test_b)
    pr_b = precision_recall_at_k(test_b, 1)
    print(auc_b.round(4))
    print(f"평균 AUC: {auc_b.mean():.4f}, 평균 Precision@1: {pr_b['precision_at_k'].mean():.4f}")

    print("\n=== 비교 (B - A) ===")
    comparison = pd.DataFrame({"base(A)": auc_a, "excess_added(B)": auc_b})
    comparison["diff"] = comparison["excess_added(B)"] - comparison["base(A)"]
    print(comparison.round(4))
    print(f"\nPrecision@1 변화: {pr_a['precision_at_k'].mean():.4f} -> {pr_b['precision_at_k'].mean():.4f}")
    print(f"(참고: (B)는 학습행 {n_tr_b}개로 (A)의 {n_tr_a}개보다 적을 수 있음 — 카테고리 키워드 수 1개뿐인 주 제외 때문)")


if __name__ == "__main__":
    main()
