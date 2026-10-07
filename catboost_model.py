"""
`model_comparison_experiment.py`에서 알고리즘 5종 x feature 조합 5종(25가지)을
전부 비교했을 때는 CatBoost + 확장 feature셋(momentum·rank_in_category·
days_to_holiday) 3개 전부 추가가 평균 AUC 0.8075로 가장 높았다. 그러나 추가
검증 결과 **days_to_holiday는 같은 주·같은 카테고리 안에서 모든 키워드가
동일한 값을 가지는 feature라 "이번 주 여러 키워드 중 무엇을 고를지" 고르는
문제(Precision@K)에는 전혀 기여하지 못하고, 오히려 Precision@1을 악화시켰다**
(성분 17.2%→15.9%) — AUC 개선은 "명절 근처 주는 전반적으로 breakout이 많다"는
시간대별 기준선 이동을 테스트 전체 기간에 걸쳐 포착한 것일 뿐, 키워드 간
상대적 순위를 바꾸는 효과가 아니었다(§4.1·5.2 참고).

따라서 이 스크립트는 **days_to_holiday를 제외**하고 within-week 판별력이
실제로 있는 momentum·rank_in_category만 추가한 **CatBoost(평균 AUC 0.7936,
Precision@1 평균 25.0%로 holiday 포함 모델의 24.7%보다도 높음)**를 §4.1의
최종 공식 주 모델로 정식화한다(xgboost_model.py와 동일한 구조·평가 방식 사용).
days_to_holiday가 보여준 "명절 동시성"(§4.6)은 폐기하지 않고, "어떤 키워드를
고를지"가 아니라 "언제 모니터링 강도를 올릴지"를 결정하는 별도의 운영 전략
feature로 역할을 재배치한다.

신규 feature 2개의 의미:
  - momentum: signal(지연된 growth_rate)의 1주 전 대비 변화량 — "오르는 속도
    자체가 더 빨라지고 있는가"(가속도). 키워드마다 다른 값을 가져 같은 주
    안에서도 순위를 바꿀 수 있다.
  - rank_in_category: 같은 주·같은 카테고리 내에서 이 키워드의 signal이 상위
    몇 %인지(0~1) — 절대적 증가율이 아니라 "동료 키워드 대비 상대적 두각".
    정의상 같은 주 안에서 키워드 간 순위를 만들어내는 feature다.

사용법:
  python3 catboost_model.py
"""

import argparse

import pandas as pd
from catboost import CatBoostClassifier

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix, drop_missing, precision_recall_at_k
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]


def fit_and_predict(train, test, feature_cols):
    train = drop_missing(train, [c for c in feature_cols if c in
                                  ("signal", "volatility", "seasonal_deviation_pct_lag1")])
    train = train.dropna(subset=[c for c in feature_cols if c not in
                                  ("signal", "volatility", "seasonal_deviation_pct_lag1")])
    test_valid = drop_missing(test, [c for c in feature_cols if c in
                                      ("signal", "volatility", "seasonal_deviation_pct_lag1")])
    test_valid = test_valid.dropna(subset=[c for c in feature_cols if c not in
                                            ("signal", "volatility", "seasonal_deviation_pct_lag1")])

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test_valid, feature_cols).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)

    test_valid = test_valid.copy()
    test_valid["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return model, X_train.columns.tolist(), test_valid


def evaluate(train, test, feature_cols, model_name, ks):
    model, columns, test_pred = fit_and_predict(train, test, feature_cols)
    importances = dict(zip(columns, model.get_feature_importance().round(3)))
    print(f"\n[{model_name}] 사용 feature: {feature_cols} (+ category one-hot)")
    print(f"  feature importance: {importances}")

    auc = auc_by_category(test_pred)
    all_k = []
    for k in ks:
        pr = precision_recall_at_k(test_pred, k)
        pr["k"] = k
        all_k.append(pr.reset_index())
    results = pd.concat(all_k, ignore_index=True)
    results = results.merge(auc.reset_index().rename(columns={"index": "category"}), on="category")
    results["model"] = model_name
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_features.csv")
    parser.add_argument("--lag-input", default="lag_selected.csv")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", default="catboost_results.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    lag_by_category = load_lag_windows(args.lag_input)
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    train, test = split_backtest(df)
    print(f"학습 구간(~{TRAIN_END}): {len(train)}행, 테스트 구간({TEST_START}~{TEST_END}): {len(test)}행")

    full = evaluate(train, test, FEATURE_COLS, "catboost_full_extended", args.k)

    cols = ["model", "category", "k", "tp", "fp", "fn", "precision_at_k", "recall_at_k", "auc"]
    print("\nCatBoost(확장 feature) 결과:")
    print(full[cols].sort_values(["category", "k"]).to_string(index=False))

    full.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n완료: {args.output}")


if __name__ == "__main__":
    main()
