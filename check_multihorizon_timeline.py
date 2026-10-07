"""
사용자 제안 "2. 다중 호라이즌 동시 스코어": §4.4에서 1주/26주/52주를 각각
따로 학습·평가했던 것을 하나의 "타임라인"으로 통합한다. 키워드 하나에
대해 "1개월 내 확률 X%, 6개월 내 Y%, 1년 내 Z%"를 동시에 제시하면,
제품기획·마케팅 리드타임에 맞춰 "지금 당장 볼 키워드"와 "중장기로
지켜볼 키워드"를 한 번에 구분할 수 있다.

§4.1에서 확정한 최종 모델(CatBoost + momentum + rank_in_category,
days_to_holiday 제외)과 동일한 feature·하이퍼파라미터를 4개 호라이즌
(1주/4주/26주/52주)에 각각 학습해, 테스트 구간 마지막 시점 기준으로
카테고리별 상위 키워드의 타임라인을 예시로 보여준다.

사용법:
  python3 check_multihorizon_timeline.py
"""

import pandas as pd
from xgboost import XGBClassifier as _unused  # noqa: F401 (미사용, 참고용)
from catboost import CatBoostClassifier

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_features, load_lag_windows
from check_long_horizon_prediction import add_forward_breakout_label
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]
HORIZONS_WEEKS = [1, 4, 26, 52]
HORIZON_LABELS = {1: "1주(즉시)", 4: "4주(1개월)", 26: "26주(6개월)", 52: "52주(1년)"}


def fit_score(train, test, label_col):
    train = drop_missing(train, FEATURE_COLS).loc[lambda d: d[label_col].notna()]
    test = drop_missing(test, FEATURE_COLS).loc[lambda d: d[label_col].notna()]

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train[label_col].astype(bool).astype(int)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    model = CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        scale_pos_weight=(n_neg / n_pos if n_pos > 0 else 1.0),
        random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)

    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]
    return test


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    horizon_scores = {}
    for horizon in HORIZONS_WEEKS:
        labeled = add_forward_breakout_label(df, horizon)
        train = labeled[labeled["period"] <= TRAIN_END]
        test = labeled[(labeled["period"] >= TEST_START) & (labeled["period"] <= TEST_END)]
        scored = fit_score(train, test, "forward_breakout")

        from sklearn.metrics import roc_auc_score
        auc_by_cat = {}
        for category, group in scored.groupby("category"):
            y = group["forward_breakout"].astype(bool).astype(int)
            auc_by_cat[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
        print(f"호라이즌 {HORIZON_LABELS[horizon]}: AUC = {pd.Series(auc_by_cat).round(4).to_dict()}")

        horizon_scores[horizon] = scored[["keyword", "category", "period", "pred_proba"]].rename(
            columns={"pred_proba": f"P_{horizon}w"}
        )

    # 4개 호라이즌 전부 값이 있는 가장 최근 시점(52주 라벨이 요구하는 미래
    # 데이터가 남아있는 마지막 시점) 기준으로 타임라인 테이블 구성
    merged = None
    for horizon, scored in horizon_scores.items():
        snap = scored[["keyword", "category", "period", f"P_{horizon}w"]]
        merged = snap if merged is None else merged.merge(snap, on=["keyword", "category", "period"], how="outer")
    complete = merged.dropna(subset=[f"P_{h}w" for h in HORIZONS_WEEKS])
    snapshot_period = complete["period"].max()
    print(f"\n=== {snapshot_period.date()} 기준(4개 호라이즌 모두 산출 가능한 마지막 시점) 카테고리별 상위 5개 키워드 타임라인 ===")
    merged = complete[complete["period"] == snapshot_period]

    merged = merged.sort_values("P_52w", ascending=False)
    for category, group in merged.groupby("category"):
        top5 = group.sort_values("P_1w", ascending=False).head(5)
        print(f"\n[{category}]")
        print(top5[["keyword", "P_1w", "P_4w", "P_26w", "P_52w"]].round(3).to_string(index=False))

    merged.to_csv("multihorizon_timeline_snapshot.csv", index=False, encoding="utf-8-sig")
    print("\n완료: multihorizon_timeline_snapshot.csv")


if __name__ == "__main__":
    main()
