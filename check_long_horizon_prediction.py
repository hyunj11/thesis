"""
"6개월(26주)/1년(52주) 안에 breakout이 발생할 것인가"로 라벨을 다시 정의해,
현재 가진 데이터만으로 더 긴 리드타임의 예측이 가능한지 검증한다.

배경(사용자 요청): 화장품은 제품기획·마케팅 실행에 시간이 걸리므로, "이번
주가 breakout인가"를 맞히는 것보다 "지금부터 6개월~1년 안에 뜰 것인가"를
미리 아는 게 실무적으로 더 메리트가 있다. 해외(미국) 선행 신호를 테스트해
봤으나(check_us_lead_lag.py) 지지되지 않아, 새 데이터 수집 없이 기존
파이프라인으로 직접 테스트 가능한 이 방향을 먼저 시도한다.

라벨 재정의:
  forward_breakout_Nw_t = keyword별로 t+1주 ~ t+N주 구간에 is_breakout이
  한 번이라도 True이면 True.
  - 미래 정보 누수 아님에 주의: feature(signal/volatility/계절편차)는 전부
    t 시점 "이전" 데이터만 쓰는 기존 방식 그대로이고, 바뀌는 것은 "맞혀야
    하는 타겟(target)"이 t 시점이 아니라 "t 이후 N주 윈도 안의 사건
    발생 여부"로 바뀌는 것뿐이다 — 이는 일반적인 "예측 호라이즌(forecast
    horizon)" 설정으로, 타겟 자체가 미래를 가리키는 것은 당연하고 정당한
    설계다(feature가 미래를 보는 것과는 다름).
  - 평가 가능 구간 제약: forward label을 계산하려면 t+N주까지의 실측
    데이터가 필요하므로, 전체 데이터의 마지막 N주는 라벨을 계산할 수 없어
    평가에서 제외한다(수집 종료 시점 2026-09-21 기준).

결과(2026-10-08, CatBoost+모멘텀+카테고리 내 순위 기준): 성분·제형·효능은
호라이즌이 길어질수록 AUC가 뚜렷이 떨어진다(성분 0.712→0.632, 제형
0.760→0.705, 효능 0.870→0.720) — 단기 급등 신호가 멀리 못 간다는 당연한
결과. 컨셉·클레임은 예외적으로 26주·52주에서 오히려 1주보다 AUC가 높다
(0.826→0.875/0.872) — 이 카테고리는 트렌드가 한번 형성되면 오래 지속되는
성격이 강해(§4.5 참고) "장기간 내 breakout 발생 여부"가 "이번 주 breakout
여부"보다 카테고리 내 순위·모멘텀 같은 feature로 더 쉽게 구분되는 것으로
보인다. XGBoost 시절과 방향·해석 모두 동일.

비교 대상: N=1주(기존 방식, 그대로 비교용으로 재현) / N=26주(6개월) /
N=52주(1년). 주 모델(§4.1: CatBoost+모멘텀+카테고리 내 순위)로 각각
학습·평가해 AUC가 호라이즌이 길어질수록 얼마나 떨어지는지 확인한다.

2026-10-08: §4.1 주 모델 전환(XGBoost 3-feature → CatBoost+모멘텀+
카테고리 내 순위)에 맞춰 재검증.

사용법:
  python3 check_long_horizon_prediction.py
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from baseline_model import TEST_END, TEST_START, TRAIN_END, add_signal, load_features, load_lag_windows
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]
HORIZONS_WEEKS = [1, 26, 52]


def add_forward_breakout_label(df, horizon_weeks):
    df = df.sort_values(["keyword", "period"]).copy()
    if horizon_weeks == 1:
        df["forward_breakout"] = df["is_breakout"]
        return df

    def forward_any(group):
        # shift(-1)..shift(-horizon_weeks)의 is_breakout 중 하나라도 True인지.
        # 끝부분(미래 데이터가 N주만큼 없는 구간)은 NaN으로 남겨 평가에서 제외.
        shifted = pd.concat(
            [group["is_breakout"].shift(-k) for k in range(1, horizon_weeks + 1)], axis=1
        )
        has_future = shifted.notna().all(axis=1)
        any_true = shifted.fillna(False).any(axis=1)
        return any_true.where(has_future)

    df["forward_breakout"] = df.groupby("keyword", group_keys=False).apply(forward_any)
    return df


def fit_auc(train, test, label_col):
    train = (
        drop_missing(train, FEATURE_COLS)
        .dropna(subset=["momentum", "rank_in_category"])
        .loc[lambda d: d[label_col].notna()]
    )
    test = (
        drop_missing(test, FEATURE_COLS)
        .dropna(subset=["momentum", "rank_in_category"])
        .loc[lambda d: d[label_col].notna()]
    )

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train[label_col].astype(bool).astype(int)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    model = CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)

    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]

    results = {}
    for category, group in test.groupby("category"):
        y = group[label_col].astype(bool).astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    return results, int(y_train.sum()), len(train), len(test)


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    all_results = []
    for horizon in HORIZONS_WEEKS:
        labeled = add_forward_breakout_label(df, horizon)
        train = labeled[labeled["period"] <= TRAIN_END]
        test = labeled[(labeled["period"] >= TEST_START) & (labeled["period"] <= TEST_END)]

        auc, n_pos_train, n_train, n_test = fit_auc(train, test, "forward_breakout")
        label_desc = "1주(기존)" if horizon == 1 else f"{horizon}주({horizon // 4.33:.0f}개월)"
        print(f"\n=== 호라이즌 {label_desc} — 학습 {n_train}행(양성 {n_pos_train}) / 테스트 {n_test}행 ===")
        print(pd.Series(auc).round(4))

        for category, value in auc.items():
            all_results.append({"horizon_weeks": horizon, "category": category, "auc": value})

    result = pd.DataFrame(all_results)
    pivot = result.pivot(index="category", columns="horizon_weeks", values="auc")
    print("\n=== 호라이즌별 AUC 요약 (카테고리 x 호라이즌) ===")
    print(pivot.round(4))

    result.to_csv("long_horizon_auc.csv", index=False, encoding="utf-8-sig")
    print("\n완료: long_horizon_auc.csv")


if __name__ == "__main__":
    main()
