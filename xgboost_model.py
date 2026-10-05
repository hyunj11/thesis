"""
RESEARCH_PLAN.md "4) 모델"의 "본 모델: 로지스틱회귀 → XGBoost/LightGBM 비교
(해석가능성 vs 성능 트레이드오프)" 중 XGBoost를 구현한다.

logistic_regression_model.py와 동일한 feature/백테스팅 split/ablation
구조(growth_only vs full)·평가 방식(Precision@K/Recall@K/AUC)을 그대로
재사용해서, 알고리즘만 바꿨을 때의 성능 차이를 공정하게 비교할 수 있게
한다. 로지스틱회귀와 달리 트리 기반 모델은 feature 스케일링이 필요 없어
StandardScaler는 생략한다(비교 대상은 "feature 구성"이 아니라
"알고리즘"이므로, feature 자체는 로지스틱회귀와 동일하게 유지).

클래스 불균형 대응: 로지스틱회귀의 class_weight="balanced"와 동일한 목적으로
scale_pos_weight = (음성 수 / 양성 수)를 학습 데이터 기준으로 설정(XGBoost
공식 문서가 권장하는 불균형 이진분류 표준 설정).

과적합 방지: 학습 표본이 카테고리당 수천 행 수준으로 크지 않으므로
max_depth=3(얕은 트리), subsample/colsample_bytree=0.8(행·열 서브샘플링)로
트리 앙상블의 전형적인 정칙화(regularization) 설정을 사용.

사용법:
  python3 xgboost_model.py
"""

import argparse

import pandas as pd
from xgboost import XGBClassifier

from baseline_model import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    add_signal,
    load_features,
    load_lag_windows,
    split_backtest,
)
from logistic_regression_model import (
    add_lagged_seasonal,
    auc_by_category,
    build_design_matrix,
    precision_recall_at_k,
)


def fit_and_predict(train, test, feature_cols):
    train = train.dropna(subset=feature_cols + ["is_breakout"])
    test_valid = test.dropna(subset=feature_cols + ["is_breakout"])

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test_valid, feature_cols)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train["is_breakout"].astype(int)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_valid = test_valid.copy()
    test_valid["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return model, X_train.columns.tolist(), test_valid


def evaluate(train, test, feature_cols, model_name, ks):
    model, columns, test_pred = fit_and_predict(train, test, feature_cols)
    importances = dict(zip(columns, model.feature_importances_.round(3)))
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
    parser.add_argument("--output", default="xgboost_results.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    lag_by_category = load_lag_windows(args.lag_input)
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)

    train, test = split_backtest(df)
    print(f"학습 구간(~{TRAIN_END}): {len(train)}행, 테스트 구간({TEST_START}~{TEST_END}): {len(test)}행")

    growth_only = evaluate(train, test, ["signal"], "xgb_growth_only", args.k)
    full = evaluate(
        train, test, ["signal", "volatility", "seasonal_deviation_pct_lag1"], "xgb_full", args.k
    )

    results = pd.concat([growth_only, full], ignore_index=True)
    cols = ["model", "category", "k", "tp", "fp", "fn", "precision_at_k", "recall_at_k", "auc"]
    print("\nAblation 비교 (xgb_growth_only vs xgb_full):")
    print(results[cols].sort_values(["category", "model", "k"]).to_string(index=False))

    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n완료: {args.output}")


if __name__ == "__main__":
    main()
