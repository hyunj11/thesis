"""
5.2절 추가 검증: "한 번도 본 적 없는 신규 키워드"에 대한 일반화 검증.

본 연구의 백테스팅(baseline_model.py/logistic_regression_model.py)은 시간
기준 분할이다 — 같은 키워드의 과거(학습)·미래(테스트) 구간을 나눈 것이지,
"학습 때 전혀 등장하지 않았던 키워드"를 테스트하는 것이 아니다(키워드 자체는
feature로 쓰이지 않고 category만 쓰이므로 이론적으로는 신규 키워드에도
적용 가능해야 하지만, 실제로 검증된 적은 없었다).

방법: 카테고리별로 키워드의 20%를 무작위로 완전히 떼어내(이 키워드들의
모든 기간 데이터를 학습에서 제외) "신규 키워드"로 취급한다. 나머지 80%
키워드의 학습 구간(~2022-12-31)만으로 모델을 학습시킨 뒤, 테스트 구간
(2023~2025)에서 (A) 학습에 쓰인 80% 키워드(이미 본 키워드)의 AUC와
(B) 완전히 떼어낸 20% 키워드(신규 키워드)의 AUC를 각각 계산해 비교한다.
10회 반복(seed 0~9)해 평균·표준편차를 본다.

해석 기준: (B)가 (A)보다 뚜렷이 낮으면 "모델이 사실은 키워드별 특이값을
암묵적으로 학습해 신규 키워드에는 잘 일반화되지 않는다"는 뜻이고, 비슷하면
"카테고리·증가율·변동성·계절성 같은 일반화 가능한 신호를 학습했다"는 근거가
된다.

사용법:
  python3 check_keyword_holdout_generalization.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from baseline_model import add_signal, load_features, load_lag_windows, split_backtest
from logistic_regression_model import add_lagged_seasonal, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
N_DRAWS = 10
HOLDOUT_FRACTION = 0.2
SEED = 0


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    train_all, test_all = split_backtest(df)

    keywords_by_category = df.groupby("category")["keyword"].unique().to_dict()
    rng = np.random.default_rng(SEED)

    rows = []
    for draw in range(N_DRAWS):
        holdout_keywords = set()
        for category, kws in keywords_by_category.items():
            n_holdout = max(1, round(len(kws) * HOLDOUT_FRACTION))
            holdout_keywords.update(rng.choice(kws, size=n_holdout, replace=False))

        train = train_all[~train_all["keyword"].isin(holdout_keywords)].dropna(
            subset=FEATURE_COLS + ["is_breakout"]
        )
        test = test_all.dropna(subset=FEATURE_COLS + ["is_breakout"]).copy()
        test_seen = test[~test["keyword"].isin(holdout_keywords)]
        test_new = test[test["keyword"].isin(holdout_keywords)]

        if train["is_breakout"].nunique() < 2:
            continue

        X_train = build_design_matrix(train, FEATURE_COLS)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train_s, train["is_breakout"])

        for label, subset in [("seen_keywords", test_seen), ("new_keywords", test_new)]:
            X_test = build_design_matrix(subset, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
            subset = subset.copy()
            subset["pred_proba"] = model.predict_proba(scaler.transform(X_test))[:, 1]
            for category, group in subset.groupby("category"):
                y = group["is_breakout"].astype(int)
                auc = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
                rows.append({"draw": draw, "category": category, "group": label, "auc": auc, "n": len(group)})

    result = pd.DataFrame(rows)
    result.to_csv("keyword_holdout_draws.csv", index=False, encoding="utf-8-sig")

    summary = result.groupby(["category", "group"])["auc"].agg(["mean", "std", "count"]).round(4)
    print("=== 카테고리별 'seen_keywords'(학습에 쓰인 키워드) vs 'new_keywords'(완전 홀드아웃) AUC ===")
    print(summary)

    pivot = result.groupby(["category", "group"])["auc"].mean().unstack()
    pivot["gap(seen - new)"] = pivot["seen_keywords"] - pivot["new_keywords"]
    print("\n=== 요약: seen vs new 격차 ===")
    print(pivot.round(4))

    print("\n완료: keyword_holdout_draws.csv")


if __name__ == "__main__":
    main()
