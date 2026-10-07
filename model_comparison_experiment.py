"""
사용자 요청: "기본 베이스는 예측 연구가 되어야 한다 — 모든 방법을 다 이용해서
예측 성능 자체를 최대한 끌어올려보자."

지금까지 §4.1의 주 모델은 XGBoost(고정 하이퍼파라미터) + feature 3개(signal/
volatility/seasonal_deviation)였다. 이 스크립트는 RQ1(예측 성능)을 더 끌어올릴
수 있는 모든 레버를 체계적으로 테스트한다:

  (A) 알고리즘: 로지스틱회귀 / XGBoost / LightGBM / CatBoost / RandomForest
  (B) 추가 feature:
      - momentum: signal의 1차 변화량(가속도) — "오르는 속도 자체가 변하는가"
      - rank_in_category: 같은 주·같은 카테고리 내에서 signal의 상대적 순위(0~1)
      - days_to_holiday: §4.6에서 발견한 "명절(설날·추석·양력 새해) 동시 급증"을
        모델이 사후 설명이 아니라 사전 예측에 직접 쓸 수 있도록, 가장 가까운
        명절까지의 거리(일)를 feature로 추가
  (C) 앙상블: XGBoost·LightGBM 예측확률 단순 평균

전부 동일한 백테스팅 split(~2022-12-31 학습 / 2023~2025 테스트)과 동일한
평가(카테고리별 AUC)로 비교해 "어떤 조합이 실제로 성능을 올리는가"만 본다.

사용법:
  python3 model_comparison_experiment.py
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, drop_missing

BASE_FEATURES = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
EXTRA_FEATURES = ["momentum", "rank_in_category", "days_to_holiday"]

# 설날/추석(음력, 매년 날짜 이동)·양력 새해 — §4.6 발견을 feature로 직접 사용
LUNAR_NEW_YEAR = {
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05",
    2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17",
}
CHUSEOK = {
    2016: "2016-09-15", 2017: "2017-10-04", 2018: "2018-09-24", 2019: "2019-09-13",
    2020: "2020-10-01", 2021: "2021-09-21", 2022: "2022-09-10", 2023: "2023-09-29",
    2024: "2024-09-17", 2025: "2025-10-06",
}


def build_holiday_dates():
    dates = []
    for y in range(2015, 2027):
        dates.append(pd.Timestamp(f"{y}-01-01"))
    for d in LUNAR_NEW_YEAR.values():
        dates.append(pd.Timestamp(d))
    for d in CHUSEOK.values():
        dates.append(pd.Timestamp(d))
    return pd.DatetimeIndex(sorted(dates))


HOLIDAY_DATES = build_holiday_dates()


def days_to_nearest_holiday(period_series):
    diffs = np.abs(period_series.values[:, None] - HOLIDAY_DATES.values[None, :])
    min_days = diffs.min(axis=1) / np.timedelta64(1, "D")
    return min_days


def add_extra_features(df):
    df = df.sort_values(["category", "keyword", "period"]).copy()
    df["momentum"] = df.groupby("keyword")["signal"].diff()
    df["rank_in_category"] = df.groupby(["category", "period"])["signal"].rank(pct=True)
    df["days_to_holiday"] = days_to_nearest_holiday(df["period"])
    return df


def build_design_matrix(df, feature_cols):
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    return pd.concat([df[feature_cols], cat_dummies], axis=1)


def fit_predict(model_name, train, test, feature_cols):
    train = drop_missing(train, [c for c in feature_cols if c in BASE_FEATURES] or BASE_FEATURES)
    train = train.dropna(subset=[c for c in feature_cols if c not in BASE_FEATURES])
    test_valid = drop_missing(test, [c for c in feature_cols if c in BASE_FEATURES] or BASE_FEATURES)
    test_valid = test_valid.dropna(subset=[c for c in feature_cols if c not in BASE_FEATURES])

    X_train = build_design_matrix(train, feature_cols)
    X_test = build_design_matrix(test_valid, feature_cols).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    if model_name == "logistic_regression":
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(X_train), scaler.transform(X_test)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
    elif model_name == "xgboost":
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    elif model_name == "lightgbm":
        model = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    elif model_name == "catboost":
        model = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, scale_pos_weight=spw,
                                    random_state=42, verbose=False)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(model_name)

    test_valid = test_valid.copy()
    test_valid["pred_proba"] = proba
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

    feature_variants = {
        "base(signal+vol+seasonal)": BASE_FEATURES,
        "base+momentum": BASE_FEATURES + ["momentum"],
        "base+rank": BASE_FEATURES + ["rank_in_category"],
        "base+holiday": BASE_FEATURES + ["days_to_holiday"],
        "base+all_extra": BASE_FEATURES + EXTRA_FEATURES,
    }
    algorithms = ["logistic_regression", "xgboost", "lightgbm", "catboost", "random_forest"]

    all_results = []
    for feat_name, feat_cols in feature_variants.items():
        for algo in algorithms:
            pred = fit_predict(algo, train, test, feat_cols)
            auc = auc_by_category(pred)
            row = {"feature_set": feat_name, "model": algo, **auc.to_dict()}
            row["mean_auc"] = auc.mean()
            all_results.append(row)
            print(f"[{feat_name} | {algo}] mean AUC={auc.mean():.4f}  {auc.round(4).to_dict()}")

    result_df = pd.DataFrame(all_results).sort_values("mean_auc", ascending=False)
    print("\n=== 전체 결과 (평균 AUC 내림차순) ===")
    print(result_df.to_string(index=False))
    result_df.to_csv("model_comparison_results.csv", index=False, encoding="utf-8-sig")

    # 앙상블: 최고 성능 feature set 기준 XGBoost + LightGBM 확률 평균
    best_feat_name = result_df.iloc[0]["feature_set"]
    best_feat_cols = feature_variants[best_feat_name]
    pred_xgb = fit_predict("xgboost", train, test, best_feat_cols)
    pred_lgbm = fit_predict("lightgbm", train, test, best_feat_cols)
    merged = pred_xgb[["category", "period", "keyword", "is_breakout", "pred_proba"]].rename(
        columns={"pred_proba": "proba_xgb"}
    )
    merged = merged.merge(
        pred_lgbm[["category", "period", "keyword", "pred_proba"]].rename(columns={"pred_proba": "proba_lgbm"}),
        on=["category", "period", "keyword"],
    )
    merged["pred_proba"] = (merged["proba_xgb"] + merged["proba_lgbm"]) / 2
    ensemble_auc = auc_by_category(merged)
    print(f"\n=== 앙상블(XGB+LightGBM 평균, feature={best_feat_name}) ===")
    print(ensemble_auc.round(4))
    print(f"평균 AUC: {ensemble_auc.mean():.4f}")

    print("\n완료: model_comparison_results.csv")


if __name__ == "__main__":
    main()
