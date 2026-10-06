"""
5.2절 "라벨링 방식과 트렌드 유형의 불일치 가능성" 한계에 대한 탐색적 검증.

가설(5.1/5.2절): 성분 카테고리는 "장기 지속형" 트렌드가 많아(half-life
censored 비율 35.7%, 4개 카테고리 중 최고 — category_volatility_summary.csv
참고), 단기 급등을 전제로 한 Tukey IQR breakout 라벨보다 "지속적으로 평소보다
높은 수준을 유지하는가"를 보는 라벨이 더 잘 맞을 수 있다.

방법: breakout(단기 급등) 라벨과 별개로 "sustained_trend" 라벨을 새로
정의해, 기존과 동일한 feature(signal/volatility/계절편차)·백테스팅 split으로
로지스틱회귀를 학습시켜 카테고리별 AUC를 비교한다. 성분 카테고리에서
sustained_trend AUC가 기존 breakout AUC(0.659)보다 유의미하게 높다면 가설을
지지하는 근거가 되고, 아니라면 가설을 기각하는 근거가 된다 — 둘 중 어느
쪽이 나오든 결과를 그대로 보고한다(사전에 원하는 결론을 정해두지 않음).

sustained_trend 라벨 정의(탐색적 — 선행연구에서 가져온 공식이 아니라 본
점검을 위해 새로 설계한 것임을 명시):
  - baseline_t = 키워드별 과거 26주(약 6개월) 이동 중앙값(ratio, 현재 주
    제외 — shift(1) 후 rolling(26))
  - above_t = ratio_t > baseline_t
  - sustained_trend_t = above_t가 "현재 주 포함 연속 4주" 모두 True인 경우 1
    (= 최근 1개월 내내 자기 자신의 최근 6개월 수준보다 높게 유지됐는지)

사용법:
  python3 sustained_trend_check.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baseline_model import add_signal, load_lag_windows, split_backtest
from baseline_model import CORE_FULL_COLS, SEASONAL_COL
from logistic_regression_model import add_lagged_seasonal, auc_by_category, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
BASELINE_WINDOW_WEEKS = 26
SUSTAIN_WEEKS = 4


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def add_sustained_label(df):
    df = df.copy()
    df["baseline_26w"] = df.groupby("keyword")["ratio"].transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW_WEEKS, min_periods=BASELINE_WINDOW_WEEKS).median()
    )
    df["above_baseline"] = df["ratio"] > df["baseline_26w"]

    def sustained_for_keyword(s):
        return s.rolling(SUSTAIN_WEEKS).sum() == SUSTAIN_WEEKS

    df["sustained_trend"] = (
        df.groupby("keyword")["above_baseline"].transform(sustained_for_keyword).fillna(False)
    ).astype("boolean")
    # baseline을 아직 계산할 수 없는 구간(초반 26주)은 라벨 자체를 NaN 처리
    df.loc[df["baseline_26w"].isna(), "sustained_trend"] = pd.NA
    return df


def prepare(df, label_col):
    # baseline_model.prepare_full_feature_rows와 동일한 결측 완화 규칙(계절편차
    # NaN -> 0)을 label_col이 is_breakout이 아닌 경우(sustained_trend)에도
    # 적용할 수 있도록 일반화한 버전(§5.2 "수집 기간" 항목 참고).
    df = df.dropna(subset=CORE_FULL_COLS + [label_col]).copy()
    df[SEASONAL_COL] = df[SEASONAL_COL].fillna(0)
    return df


def fit_and_auc(df, label_col):
    train, test = split_backtest(df)
    train = prepare(train, label_col)
    test = prepare(test, label_col)

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(scaler.fit_transform(X_train), train[label_col].astype(bool))

    test = test.copy()
    test["pred_proba"] = model.predict_proba(scaler.transform(X_test))[:, 1]
    test["is_breakout"] = test[label_col].astype(bool)  # auc_by_category가 이 컬럼명을 사용
    return auc_by_category(test), len(train), len(test), int(train[label_col].astype(bool).sum())


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_sustained_label(df)

    print(f"sustained_trend 정의: 과거 {BASELINE_WINDOW_WEEKS}주 중앙값보다 {SUSTAIN_WEEKS}주 연속 높게 유지")
    print(f"sustained_trend 양성 비율 전체: {df['sustained_trend'].mean():.1%}")
    print(df.groupby("category")["sustained_trend"].mean())
    print()

    print("=== breakout 라벨 기준 AUC (기존 결과, 비교용) ===")
    breakout_auc, n_tr_b, n_te_b, n_pos_b = fit_and_auc(df, "is_breakout")
    print(f"학습 {n_tr_b}행(양성 {n_pos_b}) / 테스트 {n_te_b}행")
    print(breakout_auc)
    print()

    print("=== sustained_trend 라벨 기준 AUC (신규) ===")
    sustained_auc, n_tr_s, n_te_s, n_pos_s = fit_and_auc(df, "sustained_trend")
    print(f"학습 {n_tr_s}행(양성 {n_pos_s}) / 테스트 {n_te_s}행")
    print(sustained_auc)
    print()

    print("=== 비교 (sustained - breakout) ===")
    comparison = pd.DataFrame({"breakout_auc": breakout_auc, "sustained_trend_auc": sustained_auc})
    comparison["diff"] = comparison["sustained_trend_auc"] - comparison["breakout_auc"]
    print(comparison.round(4))


if __name__ == "__main__":
    main()
