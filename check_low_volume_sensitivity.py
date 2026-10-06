"""
5.2절 "저검색량 키워드의 노이즈 문제" 한계에 대한 민감도 분석.

가설: ratio(절대 검색 비율) 평균이 낮은 키워드는 growth_rate가 구조적으로
노이즈에 가까워서(작은 절대 변화가 큰 %로 뻥튀기됨), 이들을 제외하면
breakout 라벨링·모델 성능이 더 안정적으로/좋게 나올 것이다.

방법: 키워드별 전체 기간 평균 ratio를 계산해, 자연스러운 단절점(threshold)
미만인 키워드를 "저검색량"으로 분류 → 라벨링·feature·full 로지스틱회귀
모델을 저검색량 키워드 포함 버전과 제외 버전 두 가지로 각각 돌려서 카테고리별
AUC를 비교한다. RESEARCH_PLAN.md/THESIS_DRAFT.md의 다른 스크립트와 동일한
파이프라인(label_breakouts.py → select_lag_window.py 로직 → compute_volatility/
seasonality → logistic_regression_model.py)을 그대로 재사용해, 새로운 방법론을
도입하지 않고 "입력 키워드 집합만 바꿨을 때" 결과가 어떻게 달라지는지만 본다.

사용법:
  python3 check_low_volume_sensitivity.py
  python3 check_low_volume_sensitivity.py --threshold 1.0
"""

import argparse

import pandas as pd
from xgboost import XGBClassifier

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix, drop_missing

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def relabel_breakouts(df):
    """label_breakouts.py와 동일한 Tukey IQR 라벨링을 키워드 부분집합에 재적용."""
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


def evaluate_auc(df, lag_by_category):
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    train, test = split_backtest(df)
    train = drop_missing(train, FEATURE_COLS)
    test = drop_missing(test, FEATURE_COLS)
    if train["is_breakout"].nunique() < 2 or len(test) == 0:
        return None

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

    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return auc_by_category(test), len(train), len(test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_features.csv")
    parser.add_argument("--lag-input", default="lag_selected.csv")
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    raw = load_features(args.input)
    mean_ratio = raw.groupby("keyword")["ratio"].mean()
    low_volume_keywords = mean_ratio[mean_ratio < args.threshold].index.tolist()
    print(f"평균 ratio < {args.threshold} 인 저검색량 키워드 {len(low_volume_keywords)}개:")
    print(sorted(low_volume_keywords))
    print()

    lag_by_category = load_lag_windows(args.lag_input)

    print("=== (A) 전체 키워드 포함 (기존과 동일) ===")
    df_all = relabel_breakouts(raw.copy())
    df_all = recompute_volatility(df_all, lag_by_category)
    auc_all, n_train_all, n_test_all = evaluate_auc(df_all, lag_by_category)
    print(f"학습 {n_train_all}행 / 테스트 {n_test_all}행")
    print(auc_all)

    print()
    print("=== (B) 저검색량 키워드 제외 후 라벨링·모델 재학습 ===")
    filtered = raw[~raw["keyword"].isin(low_volume_keywords)].copy()
    df_f = relabel_breakouts(filtered)
    df_f = recompute_volatility(df_f, lag_by_category)
    result_f = evaluate_auc(df_f, lag_by_category)
    if result_f is None:
        print("저검색량 제외 시 평가 불가(클래스 수 부족 또는 테스트 데이터 없음)")
        return
    auc_f, n_train_f, n_test_f = result_f
    print(f"학습 {n_train_f}행 / 테스트 {n_test_f}행")
    print(auc_f)

    print()
    print("=== 비교 (A: 전체 vs B: 저검색량 제외) ===")
    comparison = pd.DataFrame({"all_keywords": auc_all, "low_volume_excluded": auc_f})
    comparison["diff"] = comparison["low_volume_excluded"] - comparison["all_keywords"]
    print(comparison.round(4))


if __name__ == "__main__":
    main()
