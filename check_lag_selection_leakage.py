"""
3.3절 lag window(N주) 선택 절차의 데이터 누수(leakage) 여부 검증.

select_lag_window.py는 naver_trend_labeled.csv(2020-12-28~2026-09-28 전체
구간, 즉 학습 구간(~2022-12-31)과 테스트 구간(2023~2025)을 모두 포함)에서
growth_rate(lag L)와 is_breakout 간 상관관계가 가장 큰 L을 "카테고리별 최적
lag window"로 선택했다. 이 N(성분14/제형22/컨셉·클레임7/효능24)은 이후
signal·volatility feature를 만드는 데 그대로 쓰이고, 그 feature로 학습한
모델을 테스트 구간(2023~2025)에서 백테스팅해 AUC를 보고했다.

문제: N 자체가 테스트 구간의 is_breakout 라벨까지 포함해 선택됐다면, 이는
"미래 정보를 보고 고른 하이퍼파라미터로 미래를 맞히는" leakage다 — 각 행의
feature 값 자체는 전부 과거 데이터만 쓰지만(개별 row 수준 누수는 없음),
N이라는 하이퍼파라미터 선택 자체에 테스트 구간 정보가 섞여 들어간 것이
문제(하이퍼파라미터 선택 수준의 누수).

검증 방법: lag 선택을 학습 구간(~2022-12-31) 데이터만으로 다시 수행해(진짜
prospective 상황을 재현), 그 결과로 고른 N이 기존 N과 얼마나 다른지, 그리고
그 N으로 다시 학습·백테스팅한 AUC가 기존(전체 구간 기준 N) AUC와 얼마나
다른지 비교한다.

사용법:
  python3 check_lag_selection_leakage.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from logistic_regression_model import drop_missing
from select_lag_window import cross_correlation_by_lag, select_best_lag

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
MAX_LAG = 26

FULL_DATA_LAGS = {"성분": 14, "제형": 22, "컨셉·클레임": 7, "효능": 24}
REPORTED_AUC = {"성분": 0.600260, "제형": 0.678109, "컨셉·클레임": 0.713321, "효능": 0.699517}


def load_labeled(path="naver_trend_labeled.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    df["is_breakout"] = df["is_breakout"].astype(int)
    return df.sort_values(["category", "keyword", "period"])


def compute_volatility(df, lag_by_category):
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


def add_signal(df, lag_by_category):
    df = df.copy()
    df["signal"] = float("nan")
    for category, lag in lag_by_category.items():
        mask = df["category"] == category
        df.loc[mask, "signal"] = df.loc[mask].groupby("keyword")["growth_rate"].shift(lag)
    return df


def build_design_matrix(df, feature_cols):
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    return pd.concat([df[feature_cols], cat_dummies], axis=1)


def fit_auc(df, lag_by_category, seasonal_lookup):
    df = add_signal(df, lag_by_category)
    df = compute_volatility(df, lag_by_category)
    df = df.merge(seasonal_lookup, on=["keyword", "period"], how="left")

    feature_cols = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
    train = drop_missing(df[df["period"] <= TRAIN_END], feature_cols)
    test = drop_missing(df[df["period"] >= TEST_START], feature_cols)

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test, feature_cols).reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(scaler.fit_transform(X_train), train["is_breakout"])

    test = test.copy()
    test["pred_proba"] = model.predict_proba(scaler.transform(X_test))[:, 1]

    results = {}
    for category, group in test.groupby("category"):
        y = group["is_breakout"].astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    return results


def main():
    df = load_labeled()

    # (A) 기존 방식 재현: lag 선택에 전체 구간(학습+테스트) 사용
    corr_full = cross_correlation_by_lag(df, MAX_LAG)
    best_full = select_best_lag(corr_full).set_index("category")["lag_weeks"].to_dict()

    # (B) 올바른 prospective 방식: lag 선택에 학습 구간(~2022-12-31)만 사용
    train_only = df[df["period"] <= TRAIN_END]
    corr_train = cross_correlation_by_lag(train_only, MAX_LAG)
    best_train = select_best_lag(corr_train).set_index("category")["lag_weeks"].to_dict()

    print("=== (A) 전체 구간(학습+테스트) 기준 선택된 lag (기존 논문 수치) ===")
    print(best_full)
    print("\n=== (B) 학습 구간(~2022-12-31)만으로 선택한 lag (올바른 prospective 방식) ===")
    print(best_train)

    # seasonal_deviation_pct_lag1은 lag window와 무관하므로 기존 features 파일에서 그대로 가져옴
    existing = pd.read_csv("naver_trend_features.csv", encoding="utf-8-sig")
    existing["period"] = pd.to_datetime(existing["period"])
    existing = existing.sort_values(["keyword", "period"])
    existing["seasonal_deviation_pct_lag1"] = existing.groupby("keyword")["seasonal_deviation_pct"].shift(1)
    seasonal_lookup = existing[["keyword", "period", "seasonal_deviation_pct_lag1"]]

    auc_full = fit_auc(df, best_full, seasonal_lookup)
    auc_train = fit_auc(df, best_train, seasonal_lookup)

    comparison = pd.DataFrame(
        {
            "lag_full": pd.Series(best_full),
            "lag_trainonly": pd.Series(best_train),
            "auc_reported(full-lag)": pd.Series(REPORTED_AUC),
            "auc_recomputed(full-lag)": pd.Series(auc_full),
            "auc_trainonly-lag": pd.Series(auc_train),
        }
    )
    comparison["auc_diff(trainonly - full)"] = (
        comparison["auc_trainonly-lag"] - comparison["auc_recomputed(full-lag)"]
    )
    print("\n=== 비교: lag window 선택 방식에 따른 AUC 차이 ===")
    print(comparison.round(4))

    comparison.to_csv("lag_leakage_check.csv", encoding="utf-8-sig")
    print("\n완료: lag_leakage_check.csv")


if __name__ == "__main__":
    main()
