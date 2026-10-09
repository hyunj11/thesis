"""
사용자 제안 "3. 신규/미등록 키워드 온보딩 스코어": §5.2 "고정 후보군의 한계"
에서 실제로 추가한 9개 신규 키워드(스쿠알렌·피부광채·잡티·올인원 스킨케어·
알러지테스트·아미노산 성분·타우린 화장품·쿨링 효과·시카케어)를 이용해,
"처음 보는 키워드를 후보군에 등록하자마자 재학습 없이 바로 스코어링할 수
있는가"를 실제 사례로 검증한다.

`check_keyword_holdout_generalization.py`가 무작위로 20%를 홀드아웃했던
것과 달리, 이 스크립트는 **실제로 나중에 발굴된, 인위적이지 않은 신규
키워드 9개**를 완전히 학습에서 제외한 모델로 평가한다 — 가장 현실적인
"온보딩" 시나리오다.

사용법:
  python3 check_new_keyword_onboarding.py
"""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]

NEW_KEYWORDS = [
    "스쿠알렌", "피부광채", "잡티", "올인원 스킨케어", "알러지테스트",
    "아미노산 성분", "타우린 화장품", "쿨링 효과", "시카케어",
]


def fit_predict(train, test, feature_cols):
    train = drop_missing(train, feature_cols)
    test_valid = drop_missing(test, feature_cols)

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test_valid, feature_cols).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    model = CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        scale_pos_weight=(n_neg / n_pos if n_pos > 0 else 1.0),
        random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)

    test_valid = test_valid.copy()
    test_valid["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return test_valid


def auc_by_category(test_pred):
    results = {}
    for category, group in test_pred.groupby("category"):
        y = group["is_breakout"].astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    return pd.Series(results)


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    train, test = split_backtest(df)

    # (A) 기존 방식: 9개 신규 키워드도 포함해서 학습(참고용, 현재 §4.1의 공식 모델과 동일)
    pred_all = fit_predict(train, test, FEATURE_COLS)
    auc_all = auc_by_category(pred_all)
    new_kw_mask = pred_all["keyword"].isin(NEW_KEYWORDS)
    print("=== (A) 9개 신규 키워드 포함 학습 (참고, §4.1과 동일) ===")
    print("전체 카테고리 AUC:", auc_all.round(4).to_dict())
    if new_kw_mask.sum() > 0:
        y = pred_all.loc[new_kw_mask, "is_breakout"].astype(int)
        if y.nunique() > 1:
            print(f"  신규 키워드 9개만 따로 AUC: {roc_auc_score(y, pred_all.loc[new_kw_mask, 'pred_proba']):.4f} (n={len(y)}, 양성 {y.sum()}건)")

    # (B) 온보딩 시나리오: 9개 신규 키워드를 학습에서 완전히 제외하고, 테스트에서만 평가
    train_without_new = train[~train["keyword"].isin(NEW_KEYWORDS)]
    test_new_only = test[test["keyword"].isin(NEW_KEYWORDS)]
    pred_onboarding = fit_predict(train_without_new, test_new_only, FEATURE_COLS)

    print(f"\n=== (B) 9개 신규 키워드를 학습에서 완전히 제외한 모델로, 그 신규 키워드만 테스트 ===")
    print(f"학습 {len(drop_missing(train_without_new, FEATURE_COLS))}행(9개 신규 키워드 제외) / 테스트 {len(pred_onboarding)}행(신규 키워드만)")
    for category, group in pred_onboarding.groupby("category"):
        y = group["is_breakout"].astype(int)
        if y.nunique() > 1:
            auc = roc_auc_score(y, group["pred_proba"])
            print(f"  {category}: AUC={auc:.4f} (키워드 {group['keyword'].nunique()}개, 양성 {y.sum()}/{len(y)}건)")
        else:
            print(f"  {category}: 평가 불가(양성 또는 음성 사례 없음, 키워드 {group['keyword'].nunique()}개)")

    overall_y = pred_onboarding["is_breakout"].astype(int)
    if overall_y.nunique() > 1:
        print(f"\n전체(4개 카테고리 풀링) AUC: {roc_auc_score(overall_y, pred_onboarding['pred_proba']):.4f}")

    pred_onboarding[["keyword", "category", "period", "is_breakout", "pred_proba"]].to_csv(
        "new_keyword_onboarding_results.csv", index=False, encoding="utf-8-sig"
    )
    print("\n완료: new_keyword_onboarding_results.csv")


if __name__ == "__main__":
    main()
