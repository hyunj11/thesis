"""
5.2절 "학습 데이터가 실질적으로 2022년 한 해뿐"이라는 문제를 실제로 완화할
수 있는 방법을 검증한다 (사용자 피드백: "2022년 한 해만으로 학습한 건 좀
에바인듯" → 지적만 하지 말고 실제로 개선 가능한지 확인).

근본 원인: `full` feature(계절편차 포함)가 유효하려면 (1) lag 윈도(최대
24주)만큼의 과거 growth_rate와 (2) 계절 기준선의 "과거 연도 1개 이상"이
동시에 충족돼야 하는데, 수집 시작일(2020-12-28)로부터 1년이 지나야 (2)가
충족되므로 실제로는 2022-01-10부터만 유효하다. 이 세션에는 네이버 API
키가 없어 더 과거 데이터를 추가로 수집할 수는 없다(근본적 해결책은
`collect_naver_trends.py --start-date`를 더 이른 날짜로 다시 실행하는
것 — 향후 과제로 제안).

대신 지금 가진 데이터로 즉시 적용 가능한 완화책을 검증한다: signal·
volatility는 이미 유효하지만 계절편차만 NaN인 행(과거 연도가 아예 없어서
계산 자체가 불가능한, 2021-02~2022-12 구간의 2,253행)을 버리지 않고,
"계절적으로 특이사항 없음(평균적인 주)"이라는 중립값인 **0**으로 채워
학습에 포함시킨다. 이는 미래 정보를 쓰는 것이 아니라 "아직 계절 기준선을
계산할 수 없다 = 이례적 편차가 있다고 볼 근거가 없다"는 보수적 가정이므로
누수가 아니다.

방법: (A) 기존 방식(계절편차 NaN이면 행 자체를 버림, 학습 3,200행,
2022-01~2022-12) vs (B) 계절편차 NaN을 0으로 채워 학습에 포함(학습
5,453행, 2021-02~2022-12로 확장) — 두 방식으로 각각 재학습해 테스트
AUC를 비교한다.

사용법:
  python3 check_seasonal_impute_extension.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from baseline_model import add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
REPORTED_AUC = {"성분": 0.659156, "제형": 0.745129, "컨셉·클레임": 0.734491, "효능": 0.825055}


def fit_auc(train, test, label):
    train = train.dropna(subset=["signal", "volatility", "is_breakout"])
    test = test.dropna(subset=FEATURE_COLS + ["is_breakout"])

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(scaler.fit_transform(X_train), train["is_breakout"])

    test = test.copy()
    test["pred_proba"] = model.predict_proba(scaler.transform(X_test))[:, 1]
    results = {}
    for category, group in test.groupby("category"):
        y = group["is_breakout"].astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    print(f"\n[{label}] 학습 {len(train)}행 ({train['period'].min()} ~ {train['period'].max()})")
    return results


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    train_all, test_all = split_backtest(df)

    # (A) 기존 방식: seasonal NaN이면 통째로 dropna에서 걸러짐
    train_strict = train_all.dropna(subset=FEATURE_COLS + ["is_breakout"])
    auc_strict = fit_auc(train_strict, test_all, "A: 기존(계절편차 NaN 행 제외)")

    # (B) 완화 방식: signal/volatility만 유효하면 살리고 계절편차 NaN은 0으로 채움
    train_imputed = train_all.dropna(subset=["signal", "volatility", "is_breakout"]).copy()
    train_imputed["seasonal_deviation_pct_lag1"] = train_imputed["seasonal_deviation_pct_lag1"].fillna(0)
    auc_imputed = fit_auc(train_imputed, test_all, "B: 완화(계절편차 NaN -> 0으로 채워 포함)")

    comparison = pd.DataFrame(
        {
            "auc_A_기존(3200행,2022년만)": pd.Series(auc_strict),
            "auc_B_완화(학습확장)": pd.Series(auc_imputed),
        }
    )
    comparison["diff(B-A)"] = comparison["auc_B_완화(학습확장)"] - comparison["auc_A_기존(3200행,2022년만)"]
    print("\n=== 비교: 계절편차 NaN 처리 방식에 따른 테스트 AUC ===")
    print(comparison.round(4))

    comparison.to_csv("seasonal_impute_extension_comparison.csv", encoding="utf-8-sig")
    print("\n완료: seasonal_impute_extension_comparison.csv")


if __name__ == "__main__":
    main()
