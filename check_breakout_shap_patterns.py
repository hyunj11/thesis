"""
RQ1 모델이 "왜" breakout을 맞히는지, breakout 직전 키워드들에 어떤 공통된
feature 패턴이 나타나는지를 분석한다. 해외 선행 신호(구글트렌드, RQ3)가
접근성 문제로 막힌 뒤 사용자가 제안한 대안: 새 외부 데이터 없이, 기존
XGBoost `full` 모델과 기존 10년 데이터 자체에서 "공통점"을 뽑아낸다.

두 가지 분석:
  1) SHAP 전역 분석: 학습된 XGBoost full 모델(§4.1과 동일 설정)에 SHAP을
     적용해 "어떤 feature 값대가 breakout 예측에 기여하는가"를 카테고리
     전체 공통(그리고 카테고리별)로 정량화한다. 단순 feature_importances_
     (분할 횟수 기반)보다, SHAP은 "해당 feature 값이 높을 때/낮을 때 예측을
     올리는지 내리는지"까지 알려줘 실무 규칙("signal이 X 이상이면 위험")으로
     해석 가능하다.
  2) breakout 직전 궤적 비교: 실제 is_breakout=True가 된 시점 t 기준으로
     t-4주~t-1주의 signal/volatility/계절편차 평균을, 같은 키워드·카테고리의
     비breakout 시점들과 비교한다. "서서히 가속" vs "급등 직전형" 같은 공통
     전조 패턴이 있는지 확인한다.

사용법:
  python3 check_breakout_shap_patterns.py
"""

import pandas as pd
import shap
from xgboost import XGBClassifier

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]


def fit_full_model(train, test):
    train = drop_missing(train, FEATURE_COLS)
    test = drop_missing(test, FEATURE_COLS)

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, train, test


def shap_global_summary(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame(shap_values, columns=X_train.columns, index=X_train.index)

    rows = []
    for col in FEATURE_COLS:
        mean_abs = shap_df[col].abs().mean()
        # feature 값이 상위 25% / 하위 25%일 때 SHAP 기여도 평균 — 방향성 확인용
        high_mask = X_train[col] >= X_train[col].quantile(0.75)
        low_mask = X_train[col] <= X_train[col].quantile(0.25)
        rows.append({
            "feature": col,
            "mean_abs_shap": mean_abs,
            "shap_at_high_value(top25%)": shap_df.loc[high_mask, col].mean(),
            "shap_at_low_value(bottom25%)": shap_df.loc[low_mask, col].mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False), shap_df


def shap_by_category(model, X_train, train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame(shap_values, columns=X_train.columns, index=X_train.index)

    rows = []
    for category in sorted(train["category"].unique()):
        idx = train.index[train["category"] == category]
        idx = idx.intersection(shap_df.index)
        for col in FEATURE_COLS:
            rows.append({
                "category": category,
                "feature": col,
                "mean_abs_shap": shap_df.loc[idx, col].abs().mean(),
            })
    pivot = pd.DataFrame(rows).pivot(index="feature", columns="category", values="mean_abs_shap")
    return pivot


def pre_breakout_trajectory(df, lookback_weeks=4):
    """breakout 시점 t 직전(t-lookback_weeks..t-1)의 feature 평균을
    같은 키워드의 비breakout 시점들과 비교한다."""
    df = df.sort_values(["keyword", "period"]).copy()
    for col in FEATURE_COLS:
        df[f"{col}_pretrend_mean"] = (
            df.groupby("keyword")[col]
            .apply(lambda s: s.shift(1).rolling(lookback_weeks, min_periods=lookback_weeks).mean())
            .reset_index(level=0, drop=True)
        )

    pretrend_cols = [f"{c}_pretrend_mean" for c in FEATURE_COLS]
    valid = df.dropna(subset=pretrend_cols + ["is_breakout"])

    rows = []
    for category, group in valid.groupby("category"):
        breakout = group[group["is_breakout"]]
        non_breakout = group[~group["is_breakout"]]
        for col in pretrend_cols:
            rows.append({
                "category": category,
                "feature": col,
                "breakout_mean": breakout[col].mean(),
                "non_breakout_mean": non_breakout[col].mean(),
                "n_breakout": len(breakout),
                "n_non_breakout": len(non_breakout),
            })
    return pd.DataFrame(rows)


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)

    train, test = split_backtest(df)
    model, X_train, X_test, train_valid, test_valid = fit_full_model(train, test)

    print("=== 1) SHAP 전역 요약 (학습 데이터 기준) ===")
    summary, shap_df = shap_global_summary(model, X_train)
    print(summary.round(4).to_string(index=False))
    summary.to_csv("shap_global_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== 2) 카테고리별 |SHAP| 평균 ===")
    by_cat = shap_by_category(model, X_train, train_valid)
    print(by_cat.round(4))
    by_cat.to_csv("shap_by_category.csv", encoding="utf-8-sig")

    print("\n=== 3) breakout 직전 4주 평균 feature vs 비breakout 시점 ===")
    trajectory = pre_breakout_trajectory(df, lookback_weeks=4)
    print(trajectory.round(3).to_string(index=False))
    trajectory.to_csv("pre_breakout_trajectory.csv", index=False, encoding="utf-8-sig")

    print("\n완료: shap_global_summary.csv, shap_by_category.csv, pre_breakout_trajectory.csv")


if __name__ == "__main__":
    main()
