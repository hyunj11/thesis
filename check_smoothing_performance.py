"""
RQ1 예측 성능을 더 끌어올릴 방법 탐색 (2) — 검색량 스무딩.

§5.2에서 "저검색량 키워드를 통째로 제외"하는 방식은 오히려 성능을 떨어뜨림을
확인했다(올리브영 검증에서 실제 매칭되는 키워드가 섞여 있었기 때문). 이번에는
"제외" 대신 "완화"를 시도한다 — growth_rate를 계산하기 전에 ratio를 2주
이동평균으로 살짝 스무딩해, 저검색량 키워드 특유의 노이즈(작은 절대 변화가
큰 %로 뻥튀기되는 현상)를 줄이면 전체 성능이 개선되는지 확인한다.

방법: ratio를 keyword별로 2주 rolling mean으로 스무딩한 뒤, 기존과 동일한
파이프라인(카테고리별 Tukey IQR 라벨링 → lag 재선택 → 변동성 → 계절편차 →
모멘텀·순위 → CatBoost)을 그대로 재실행해 평균 AUC를 비교한다. 스무딩 자체도
라벨링 시점 T 이전 데이터만 사용하도록 과거 방향으로만 윈도를 잡는다(현재
시점 포함 과거 2주 평균 — center=False).

사용법:
  python3 check_smoothing_performance.py
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from logistic_regression_model import build_design_matrix, precision_recall_at_k

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"
SMOOTH_WINDOW = 2


def load_raw(path="naver_trend_labeled.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["keyword", "period"])


def relabel_with_smoothing(df, window):
    df = df.copy()
    df["ratio_smoothed"] = df.groupby("keyword")["ratio"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    df["growth_rate"] = df.groupby("keyword")["ratio_smoothed"].pct_change() * 100
    grouped = df.groupby("category")["growth_rate"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    df["breakout_threshold"] = q3 + 1.5 * (q3 - q1)
    df["is_breakout"] = df["growth_rate"] > df["breakout_threshold"]
    return df


def select_lag(df, max_lag=26, min_lag=2):
    rows = []
    for category, cat_df in df.groupby("category"):
        target = cat_df["is_breakout"].astype(int)
        for lag in range(min_lag, max_lag + 1):
            lagged = cat_df.groupby("keyword")["growth_rate"].shift(lag)
            valid = lagged.notna() & target.notna()
            corr = lagged[valid].corr(target[valid]) if valid.sum() >= 2 else np.nan
            rows.append({"category": category, "lag_weeks": lag, "correlation": corr})
    corr_df = pd.DataFrame(rows).dropna(subset=["correlation"])
    corr_df["abs_correlation"] = corr_df["correlation"].abs()
    idx = corr_df.groupby("category")["abs_correlation"].idxmax()
    best = corr_df.loc[idx]
    return dict(zip(best["category"], best["lag_weeks"]))


def add_signal_and_volatility(df, lag_by_category):
    df = df.copy()
    df["signal"] = float("nan")
    df["volatility"] = float("nan")
    for category, lag in lag_by_category.items():
        mask = df["category"] == category
        df.loc[mask, "signal"] = df.loc[mask].groupby("keyword")["growth_rate"].shift(lag)
        vol = (
            df.loc[mask]
            .groupby("keyword")["growth_rate"]
            .apply(lambda s: s.shift(1).rolling(lag, min_periods=lag).std())
        )
        df.loc[mask, "volatility"] = vol.droplevel("keyword").values
    return df


def add_seasonal(df):
    df = df.copy()
    df["iso_year"] = df["period"].dt.isocalendar().year
    df["iso_week"] = df["period"].dt.isocalendar().week

    def baseline_for_keyword(group):
        keyword = group.name
        group = group.sort_values("period")
        baselines = []
        for _, row in group.iterrows():
            past = group[(group["iso_week"] == row["iso_week"]) & (group["iso_year"] < row["iso_year"])]
            baselines.append(past["ratio_smoothed"].median() if len(past) >= 2 else float("nan"))
        group = group.copy()
        group["keyword"] = keyword
        group["seasonal_baseline"] = baselines
        return group

    df = df.groupby("keyword", group_keys=False).apply(baseline_for_keyword)
    df["seasonal_deviation_pct"] = (df["ratio_smoothed"] - df["seasonal_baseline"]) / df["seasonal_baseline"] * 100
    df.loc[df["seasonal_baseline"] == 0, "seasonal_deviation_pct"] = float("nan")
    df = df.sort_values(["keyword", "period"])
    df["seasonal_deviation_pct_lag1"] = df.groupby("keyword")["seasonal_deviation_pct"].shift(1)
    return df


def add_extra(df):
    df = df.sort_values(["category", "keyword", "period"]).copy()
    df["momentum"] = df.groupby("keyword")["signal"].diff()
    df["rank_in_category"] = df.groupby(["category", "period"])["signal"].rank(pct=True)
    return df


FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]


def fit_auc(df):
    train = df[df["period"] <= TRAIN_END].copy()
    test = df[(df["period"] >= TEST_START) & (df["period"] <= TEST_END)].copy()
    train[FEATURE_COLS[2]] = train[FEATURE_COLS[2]].fillna(0)
    test[FEATURE_COLS[2]] = test[FEATURE_COLS[2]].fillna(0)
    train = train.dropna(subset=["signal", "volatility", "momentum", "rank_in_category", "is_breakout"])
    test = test.dropna(subset=["signal", "volatility", "momentum", "rank_in_category", "is_breakout"])

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["is_breakout"].astype(int)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05,
                                scale_pos_weight=spw, random_state=42, verbose=False)
    model.fit(X_train, y_train)
    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]

    results = {}
    for category, group in test.groupby("category"):
        y = group["is_breakout"].astype(int)
        results[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    pr1 = precision_recall_at_k(test, 1)
    return pd.Series(results), pr1["precision_at_k"], len(train), len(test)


def main():
    raw = load_raw()

    print(f"=== 스무딩 없음(baseline, 기존 파이프라인과 동일) ===")
    df0 = relabel_with_smoothing(raw, window=1)  # window=1 => 스무딩 없음(원본 그대로)
    lag0 = select_lag(df0)
    df0 = add_signal_and_volatility(df0, lag0)
    df0 = add_seasonal(df0)
    df0 = add_extra(df0)
    auc0, pr0, ntr0, nte0 = fit_auc(df0)
    print(f"lag 선택: {lag0}")
    print(f"학습 {ntr0}행 / 테스트 {nte0}행")
    print(auc0.round(4))
    print(f"평균 AUC: {auc0.mean():.4f}, 평균 Precision@1: {pr0.mean():.4f}")

    print(f"\n=== {SMOOTH_WINDOW}주 이동평균 스무딩 적용 ===")
    df1 = relabel_with_smoothing(raw, window=SMOOTH_WINDOW)
    lag1 = select_lag(df1)
    df1 = add_signal_and_volatility(df1, lag1)
    df1 = add_seasonal(df1)
    df1 = add_extra(df1)
    auc1, pr1, ntr1, nte1 = fit_auc(df1)
    print(f"lag 선택: {lag1}")
    print(f"학습 {ntr1}행 / 테스트 {nte1}행")
    print(auc1.round(4))
    print(f"평균 AUC: {auc1.mean():.4f}, 평균 Precision@1: {pr1.mean():.4f}")

    print("\n=== 비교 ===")
    comparison = pd.DataFrame({"no_smoothing": auc0, "smoothed_2w": auc1})
    comparison["diff"] = comparison["smoothed_2w"] - comparison["no_smoothing"]
    print(comparison.round(4))


if __name__ == "__main__":
    main()
