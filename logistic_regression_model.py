"""
RESEARCH_PLAN.md "4) 모델"의 본 모델 중 로지스틱회귀를 구현한다.
베이스라인(baseline_model.py)과 동일한 백테스팅 split, 동일한 Precision@K/
Recall@K/AUC 평가 방식을 재사용해서 직접 비교 가능하게 만든다.

⚠️ 구현 중 발견한 누수(leakage) 수정:
  compute_seasonality.py가 만든 seasonal_deviation_pct는 **현재 시점 t의
  ratio_t**로 계산된 값이다. 그런데 is_breakout_t도 ratio_t로부터 정의되는
  growth_rate_t(= pct_change)의 함수이므로, seasonal_deviation_pct를 그대로
  t 시점 라벨 예측에 쓰면 "라벨을 만드는 데 쓰인 값으로 라벨을 예측"하는
  셈이 되어 RESEARCH_PLAN.md가 명시한 "라벨링 시점 T 이전 데이터만 사용"
  원칙을 어기게 된다. 따라서 모델 입력으로 쓸 때는 **1주 과거로 shift**해서
  "T-1 시점까지 관측된 계절 편차"로 사용한다. (volatility는 이미
  compute_volatility.py에서 shift(1)로 계산돼 있어 추가 조치 불필요; signal
  = growth_rate도 카테고리별 선택 lag(N>=1주)만큼 이미 과거로 밀려 있어
  안전함.)

Ablation 구조 (RESEARCH_PLAN.md 4) 참고, 김병완 2021 / 서주연 2018 선례):
  두 논문 모두 "검색량(외부변수) 없는 모형 vs 있는 모형"을 비교했다. 본
  연구는 데이터 자체가 전부 검색 트렌드에서 파생되므로 "검색 트렌드 유무"를
  그대로 재현할 수 없다 — 대신 **feature 유무**로 동일한 비교 구조를
  재현한다:
    - growth_only 모델: signal(지연된 growth_rate)과 category만 사용
      (= baseline_model.py의 Top-K와 거의 동일한 정보량을 쓰는 로지스틱
      회귀 버전)
    - full 모델: signal + volatility + 지연된 seasonal_deviation_pct +
      category 모두 사용
  두 모델의 성능 차이가 "변동성·계절성 feature를 추가하는 것이 실제로
  예측력을 높이는가"에 대한 답이 된다.

클래스 불균형: is_breakout 양성 비율이 약 5~9%로 낮아 LogisticRegression에
class_weight="balanced"를 적용(표준적인 불균형 분류 처리 방식).

사용법:
  python3 logistic_regression_model.py
"""

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from baseline_model import (
    SEASONAL_COL,
    TEST_END,
    TEST_START,
    TRAIN_END,
    add_signal,
    load_features,
    load_lag_windows,
    prepare_full_feature_rows,
    split_backtest,
)


def add_lagged_seasonal(df):
    df = df.copy()
    df["seasonal_deviation_pct_lag1"] = df.groupby("keyword")["seasonal_deviation_pct"].shift(1)
    return df


def build_design_matrix(df, feature_cols):
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    X = pd.concat([df[feature_cols], cat_dummies], axis=1)
    return X


def drop_missing(df, feature_cols):
    # full 모델(seasonal 포함)은 baseline_model.prepare_full_feature_rows의
    # 결측 완화 규칙(계절편차 NaN -> 0)을 적용하고, growth_only처럼 seasonal을
    # 쓰지 않는 모델은 그대로 전체 feature에 대해 dropna한다.
    if SEASONAL_COL in feature_cols:
        return prepare_full_feature_rows(df)
    return df.dropna(subset=feature_cols + ["is_breakout"]).copy()


def fit_and_predict(train, test, feature_cols):
    train = drop_missing(train, feature_cols)
    test_valid = drop_missing(test, feature_cols)

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test_valid, feature_cols)
    # 테스트에만 등장하는/학습에만 등장하는 카테고리 더미 컬럼 정합
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, train["is_breakout"])

    test_valid = test_valid.copy()
    test_valid["pred_proba"] = model.predict_proba(X_test_scaled)[:, 1]
    return model, X_train.columns.tolist(), test_valid


def precision_recall_at_k(test_df, k, score_col="pred_proba"):
    rows = []
    for (category, period), group in test_df.groupby(["category", "period"]):
        if len(group) == 0:
            continue
        top_k = group.sort_values(score_col, ascending=False).head(k)
        tp = int(top_k["is_breakout"].sum())
        fp = len(top_k) - tp
        fn = int(group["is_breakout"].sum()) - tp
        rows.append({"category": category, "period": period, "tp": tp, "fp": fp, "fn": fn})

    agg = pd.DataFrame(rows).groupby("category")[["tp", "fp", "fn"]].sum()
    agg["precision_at_k"] = agg["tp"] / (agg["tp"] + agg["fp"]).replace(0, pd.NA)
    agg["recall_at_k"] = agg["tp"] / (agg["tp"] + agg["fn"]).replace(0, pd.NA)
    return agg


def auc_by_category(test_df, score_col="pred_proba"):
    results = {}
    for category, group in test_df.groupby("category"):
        y = group["is_breakout"].astype(int)
        if y.nunique() < 2:
            results[category] = float("nan")
            continue
        results[category] = roc_auc_score(y, group[score_col])
    return pd.Series(results, name="auc")


def evaluate(train, test, feature_cols, model_name, ks):
    model, columns, test_pred = fit_and_predict(train, test, feature_cols)
    print(f"\n[{model_name}] 사용 feature: {feature_cols} (+ category one-hot)")
    print(f"  회귀계수: {dict(zip(columns, model.coef_[0].round(3)))}")

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
    parser.add_argument("--output", default="logistic_regression_results.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    lag_by_category = load_lag_windows(args.lag_input)
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)

    train, test = split_backtest(df)
    print(f"학습 구간(~{TRAIN_END}): {len(train)}행, 테스트 구간({TEST_START}~{TEST_END}): {len(test)}행")

    growth_only = evaluate(train, test, ["signal"], "growth_only", args.k)
    full = evaluate(
        train, test, ["signal", "volatility", "seasonal_deviation_pct_lag1"], "full", args.k
    )

    results = pd.concat([growth_only, full], ignore_index=True)
    cols = ["model", "category", "k", "tp", "fp", "fn", "precision_at_k", "recall_at_k", "auc"]
    print("\nAblation 비교 (growth_only vs full):")
    print(results[cols].sort_values(["category", "model", "k"]).to_string(index=False))

    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n완료: {args.output}")


if __name__ == "__main__":
    main()
