"""
3.2절 Tukey IQR breakout 라벨링의 핵심 설계 선택인 "1.5×IQR" 배수 자체의
민감도 검증.

본 연구는 "Q3 + 1.5×IQR"이라는 Tukey(1977)의 표준 이상치 배수를 그대로
채택했다(3.2절 "통계학에서 널리 검증된 표준 기법"). 그런데 이 1.5라는
숫자 자체도 결국 하나의 임의적 관례(convention)이며, 다른 배수를 쓰면
breakout 비율·모델 성능이 어떻게 달라지는지는 지금까지 검증한 적이 없었다.

방법: 배수를 1.0/1.5(현재 채택값)/2.0/2.5/3.0으로 바꿔 라벨링을 다시
계산하고, (1) 카테고리별 breakout 비율이 어떻게 달라지는지, (2) 각
배수로 다시 라벨링한 뒤 동일한 feature·백테스팅 split으로 재학습한 AUC가
얼마나 달라지는지를 확인한다. lag window(N)는 기존 lag_selected.csv 값을
그대로 쓴다(배수 변화만 격리해서 보기 위함 — lag 재선택까지 같이 바꾸면
두 요인이 섞인다).

사용법:
  python3 check_iqr_multiplier_sensitivity.py
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from baseline_model import load_lag_windows, split_backtest
from check_sample_size_stability import recompute_volatility
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_raw(path="naver_trends.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def load_labeled_asof(path="naver_trend_labeled.csv"):
    # naver_trend_labeled.csv에는 이미 "완전히 끝난 주"만 걸러지고
    # growth_rate가 계산돼 있으므로, 이를 그대로 불러와 라벨만 다시 매긴다.
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def relabel_with_multiplier(df, multiplier):
    df = df.copy()
    grouped = df.groupby("category")["growth_rate"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    df["breakout_threshold"] = q3 + multiplier * (q3 - q1)
    df["is_breakout"] = df["growth_rate"] > df["breakout_threshold"]
    return df


def add_signal(df, lag_by_category):
    df = df.copy()
    df["signal"] = float("nan")
    for category, lag in lag_by_category.items():
        mask = df["category"] == category
        df.loc[mask, "signal"] = df.loc[mask].groupby("keyword")["growth_rate"].shift(lag)
    return df


def fit_auc(df, lag_by_category, seasonal_lookup):
    df = add_signal(df, lag_by_category)
    df = recompute_volatility(df, lag_by_category)
    df = df.merge(seasonal_lookup, on=["keyword", "period"], how="left")

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
    results = {}
    for category, group in test.groupby("category"):
        y = group["is_breakout"].astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    return results


def main():
    df = load_labeled_asof()
    lag_by_category = load_lag_windows()

    existing = pd.read_csv("naver_trend_features.csv", encoding="utf-8-sig")
    existing["period"] = pd.to_datetime(existing["period"])
    existing = existing.sort_values(["keyword", "period"])
    existing["seasonal_deviation_pct_lag1"] = existing.groupby("keyword")["seasonal_deviation_pct"].shift(1)
    seasonal_lookup = existing[["keyword", "period", "seasonal_deviation_pct_lag1"]]

    rate_rows = []
    auc_rows = []
    for m in MULTIPLIERS:
        relabeled = relabel_with_multiplier(df, m)
        rates = relabeled.groupby("category")["is_breakout"].mean()
        counts = relabeled.groupby("category")["is_breakout"].sum()
        for cat in rates.index:
            rate_rows.append({"multiplier": m, "category": cat, "breakout_rate": rates[cat], "n_events": int(counts[cat])})

        auc = fit_auc(relabeled, lag_by_category, seasonal_lookup)
        if auc:
            for cat, v in auc.items():
                auc_rows.append({"multiplier": m, "category": cat, "auc": v})

    rate_df = pd.DataFrame(rate_rows)
    auc_df = pd.DataFrame(auc_rows)

    print("=== 배수별 카테고리별 breakout 비율(%) / 사건 수 ===")
    print(rate_df.pivot(index="category", columns="multiplier", values="breakout_rate").mul(100).round(2))
    print()
    print(rate_df.pivot(index="category", columns="multiplier", values="n_events"))
    print()
    print("=== 배수별 카테고리별 AUC ===")
    pivot_auc = auc_df.pivot(index="category", columns="multiplier", values="auc")
    pivot_auc["std_across_multipliers"] = pivot_auc[MULTIPLIERS].std(axis=1)
    print(pivot_auc.round(4))

    rate_df.to_csv("iqr_multiplier_rates.csv", index=False, encoding="utf-8-sig")
    auc_df.to_csv("iqr_multiplier_auc.csv", index=False, encoding="utf-8-sig")
    print("\n완료: iqr_multiplier_rates.csv, iqr_multiplier_auc.csv")


if __name__ == "__main__":
    main()
