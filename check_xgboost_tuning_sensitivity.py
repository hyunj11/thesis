"""
4.1절 "로지스틱회귀와 XGBoost의 full 단계 성능이 사실상 동등했다"는 결론이
XGBoost의 하이퍼파라미터를 튜닝하지 않은 데서 온 인공물(artifact)은 아닌지
검증.

xgboost_model.py는 max_depth=3/learning_rate=0.05/n_estimators=200/
subsample=colsample_bytree=0.8을 "학습 표본이 크지 않다"는 판단으로
고정값으로 썼을 뿐, 그리드서치나 교차검증으로 튜닝한 적은 없었다(김병완[1]은
GridSearchCV+5-fold로 여러 알고리즘을 튜닝했는데, 본 연구는 로지스틱회귀만
StandardScaler를 적용했을 뿐 하이퍼파라미터 튜닝 자체는 둘 다 안 함 —
공정한 비교이긴 하나 "XGBoost가 로지스틱회귀보다 안 낫다"는 결론이 혹시
튜닝 부족 때문은 아닌지 점검 필요).

방법: 학습 구간(~2022-12-31)을 다시 내부 train(~2021-12-31)/validation
(2022-01-01~2022-12-31)으로 쪼개, 소규모 그리드(max_depth x
learning_rate x n_estimators)에서 validation 평균 AUC가 가장 높은 조합을
고른다. 그 조합으로 전체 학습 구간(~2022-12-31)에 재학습해 테스트 구간
(2023~2025) AUC를 계산하고, 기존 고정 하이퍼파라미터 결과와 비교한다.

사용법:
  python3 check_xgboost_tuning_sensitivity.py
"""

import itertools

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from baseline_model import add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
INNER_TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"

GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.1],
    "n_estimators": [100, 200, 400],
}

REPORTED_AUC = {"성분": 0.704878, "제형": 0.756398, "컨셉·클레임": 0.811117, "효능": 0.820583}
# xgboost_model.py full 모델 보고값(기존 결과 CSV 기준)


def fit_predict(train, test, params):
    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return test


def mean_auc(test_pred):
    aucs = []
    for category, group in test_pred.groupby("category"):
        y = group["is_breakout"].astype(int)
        if y.nunique() > 1:
            aucs.append(roc_auc_score(y, group["pred_proba"]))
    return sum(aucs) / len(aucs) if aucs else float("nan")


def auc_by_category(test_pred):
    results = {}
    for category, group in test_pred.groupby("category"):
        y = group["is_breakout"].astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    return results


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    train_all, test_all = split_backtest(df)
    train_all = drop_missing(train_all, FEATURE_COLS)
    test_all = drop_missing(test_all, FEATURE_COLS)

    # 계절편차 NaN -> 0 완화(§5.2) 이후에도 내부 validation을 위해 날짜 대신
    # 정렬된 비율로 70/30 분할한다(날짜 기준 분할은 여전히 연초 구간에서
    # 내부 train이 작아지는 비대칭이 있을 수 있어 비율 분할이 더 안정적).
    train_sorted = train_all.sort_values("period")
    split_idx = int(len(train_sorted) * 0.7)
    inner_train = train_sorted.iloc[:split_idx]
    val = train_sorted.iloc[split_idx:]
    print(f"내부 train {len(inner_train)}행({inner_train['period'].min()}~{inner_train['period'].max()}) / "
          f"validation {len(val)}행({val['period'].min()}~{val['period'].max()}) / 기존 전체 train {len(train_all)}행")

    best_params, best_score = None, -1
    grid_results = []
    for max_depth, lr, n_est in itertools.product(GRID["max_depth"], GRID["learning_rate"], GRID["n_estimators"]):
        params = {"max_depth": max_depth, "learning_rate": lr, "n_estimators": n_est}
        val_pred = fit_predict(inner_train, val, params)
        score = mean_auc(val_pred)
        grid_results.append({**params, "val_mean_auc": score})
        if score > best_score:
            best_score, best_params = score, params

    grid_df = pd.DataFrame(grid_results).sort_values("val_mean_auc", ascending=False)
    print("\n=== 그리드서치 상위 5개 조합(validation 평균 AUC 기준) ===")
    print(grid_df.head(5).round(4).to_string(index=False))
    print(f"\n최적 조합: {best_params} (validation 평균 AUC={best_score:.4f})")
    print(f"기존 고정값: max_depth=3, learning_rate=0.05, n_estimators=200")

    # 최적 조합으로 전체 학습 구간에 재학습 → 테스트 구간 평가
    test_pred_tuned = fit_predict(train_all, test_all, best_params)
    auc_tuned = auc_by_category(test_pred_tuned)

    comparison = pd.DataFrame(
        {"auc_reported(fixed_params)": pd.Series(REPORTED_AUC), "auc_tuned": pd.Series(auc_tuned)}
    )
    comparison["diff(tuned - fixed)"] = comparison["auc_tuned"] - comparison["auc_reported(fixed_params)"]
    print("\n=== 테스트 구간 AUC: 기존 고정 하이퍼파라미터 vs 튜닝 후 ===")
    print(comparison.round(4))

    grid_df.to_csv("xgboost_tuning_grid.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv("xgboost_tuning_comparison.csv", encoding="utf-8-sig")
    print("\n완료: xgboost_tuning_grid.csv, xgboost_tuning_comparison.csv")


if __name__ == "__main__":
    main()
