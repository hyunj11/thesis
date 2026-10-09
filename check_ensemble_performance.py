"""
RQ1 예측 성능을 더 끌어올릴 방법 탐색 (1) — 앙상블.

§4.1에서 CatBoost+모멘텀+카테고리 내 순위를 최종 모델로 채택했지만,
XGBoost(0.786)·LightGBM(0.787)·CatBoost(0.791)가 서로 다른 오류를
범할 수 있으므로, 세 트리 기반 모델의 예측확률을 단순 평균(soft voting)
하면 개별 모델보다 더 나은 AUC/Precision@K를 얻을 수 있는지 검증한다.

방법: 동일한 feature셋(§4.1 최종, signal/volatility/seasonal/momentum/
rank_in_category)으로 XGBoost·LightGBM·CatBoost를 각각 학습시키고, 테스트
구간 예측확률을 (1) 단순 평균, (2) CatBoost 단독과 비교한다.

사용법:
  python3 check_ensemble_performance.py
"""

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from baseline_model import add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix, drop_missing, precision_recall_at_k
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]


def fit_one(model_name, X_train, y_train, X_test, scale_pos_weight):
    if model_name == "xgboost":
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                               eval_metric="logloss", random_state=42)
    elif model_name == "lightgbm":
        model = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                                random_state=42, verbosity=-1)
    elif model_name == "catboost":
        model = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05,
                                    scale_pos_weight=scale_pos_weight, random_state=42, verbose=False)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    train, test = split_backtest(df)
    train = drop_missing(train, FEATURE_COLS).dropna(subset=["momentum", "rank_in_category"])
    test = drop_missing(test, FEATURE_COLS).dropna(subset=["momentum", "rank_in_category"])

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    proba = {}
    for name in ["xgboost", "lightgbm", "catboost"]:
        proba[name] = fit_one(name, X_train, y_train, X_test, spw)

    test = test.copy()
    test["pred_proba_cat"] = proba["catboost"]
    test["pred_proba_ens3"] = (proba["xgboost"] + proba["lightgbm"] + proba["catboost"]) / 3
    test["pred_proba_ens2"] = (proba["lightgbm"] + proba["catboost"]) / 2  # 상위 2개(§4.1 표 기준)만 평균

    print("=== 카테고리별 AUC: CatBoost 단독 vs 3-모델 앙상블 vs 2-모델(LGBM+CatBoost) 앙상블 ===")
    comparison = pd.DataFrame({
        "catboost_only": auc_by_category(test, "pred_proba_cat"),
        "ensemble_3": auc_by_category(test, "pred_proba_ens3"),
        "ensemble_2(lgbm+cat)": auc_by_category(test, "pred_proba_ens2"),
    })
    comparison["diff(ens3-cat)"] = comparison["ensemble_3"] - comparison["catboost_only"]
    print(comparison.round(4))
    print(f"\n평균 AUC: catboost={comparison['catboost_only'].mean():.4f}, "
          f"ensemble_3={comparison['ensemble_3'].mean():.4f}, "
          f"ensemble_2={comparison['ensemble_2(lgbm+cat)'].mean():.4f}")

    print("\n=== Precision@1 비교 ===")
    for col, label in [("pred_proba_cat", "catboost_only"), ("pred_proba_ens3", "ensemble_3"),
                        ("pred_proba_ens2", "ensemble_2")]:
        pr = precision_recall_at_k(test, 1, score_col=col)
        print(f"{label}: 평균 Precision@1={pr['precision_at_k'].mean():.4f}")
        print(pr["precision_at_k"].round(4).to_dict())

    comparison.to_csv("ensemble_performance_comparison.csv", encoding="utf-8-sig")
    print("\n완료: ensemble_performance_comparison.csv")


if __name__ == "__main__":
    main()
