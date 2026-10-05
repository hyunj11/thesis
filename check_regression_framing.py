"""
5.2절 "이진분류라는 단순한 문제 설정의 한계"에 대한 검증.

BuzzProphet[23] 등 최신 해외 연구는 breakout 여부(이진)가 아니라 인기도
자체를 회귀로 예측하는 방향으로 가고 있다(Ⅱ장 2.2절). 본 연구가 이진분류를
택한 것이 실제로 정보 손실로 이어지는지, 즉 growth_rate를 직접 회귀로
예측하는 편이 breakout 여부 판별에도 더(또는 비슷하게) 유용한지를 검증한다.

방법: 분류 모델(logistic_regression_model.py)과 완전히 동일한 feature·
백테스팅 split으로 **선형회귀**를 학습시켜 growth_rate(연속값)를 직접
예측한다. 이 연속 예측값을 그대로 "breakout일 가능성 점수"로 사용해
AUC를 계산하고, 분류 모델의 AUC와 비교한다. 회귀의 R²(결정계수)도 함께
보고해 "증가율 자체를 얼마나 잘 맞히는지"도 확인한다.

해석 기준: 회귀 기반 AUC가 분류 기반 AUC와 비슷하거나 더 높으면, 이진분류로
단순화한 것이 정보 손실로 이어지지 않았다는 뜻이고(현재 설계가 적절), 회귀
기반 AUC가 뚜렷이 낮으면 이진 라벨링 자체가 분류에 유리한 정보를 추가로
제공하고 있다는 뜻이다(이진분류 선택이 실제로 도움이 됐다는 근거).

사용법:
  python3 check_regression_framing.py
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from baseline_model import add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]


def auc_and_r2_by_category(test_df, score_col, label_col="is_breakout", value_col="growth_rate"):
    rows = []
    for category, group in test_df.groupby("category"):
        y_bin = group[label_col].astype(int)
        auc = roc_auc_score(y_bin, group[score_col]) if y_bin.nunique() > 1 else float("nan")
        r2 = r2_score(group[value_col], group[score_col])
        rows.append({"category": category, "auc_vs_breakout_label": auc, "r2_vs_growth_rate": r2})
    return pd.DataFrame(rows).set_index("category")


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)

    train, test = split_backtest(df)
    train = train.dropna(subset=FEATURE_COLS + ["is_breakout", "growth_rate"])
    test = test.dropna(subset=FEATURE_COLS + ["is_breakout", "growth_rate"])

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # (A) 회귀: growth_rate를 직접 예측
    reg = LinearRegression()
    reg.fit(X_train_s, train["growth_rate"])
    test = test.copy()
    test["pred_growth_rate"] = reg.predict(X_test_s)

    # (B) 분류: is_breakout을 직접 예측 (logistic_regression_model.py의 full과 동일)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train_s, train["is_breakout"])
    test["pred_proba"] = clf.predict_proba(X_test_s)[:, 1]

    print("=== (A) 회귀(growth_rate 직접 예측) 결과 ===")
    reg_result = auc_and_r2_by_category(test, "pred_growth_rate")
    print(reg_result.round(4))
    print()

    print("=== (B) 분류(is_breakout 직접 예측, 기존 full 모델과 동일) AUC ===")
    clf_auc = {}
    for category, group in test.groupby("category"):
        clf_auc[category] = roc_auc_score(group["is_breakout"].astype(int), group["pred_proba"])
    print(pd.Series(clf_auc).round(4))
    print()

    print("=== 비교: 분류 AUC vs 회귀 점수의 AUC ===")
    comparison = pd.DataFrame(
        {"classification_auc": pd.Series(clf_auc), "regression_score_auc": reg_result["auc_vs_breakout_label"]}
    )
    comparison["diff"] = comparison["classification_auc"] - comparison["regression_score_auc"]
    print(comparison.round(4))


if __name__ == "__main__":
    main()
