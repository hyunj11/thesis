"""
5.2절 추가 검증: 단일 train/test 분할(2022-12-31 기준)에 대한 민감도 검증.

본 연구의 백테스팅은 2022-12-31을 기준으로 딱 한 번 자른 단일 분할이다.
이 분할 시점이 우연히 결과에 유리하게 작용했을 가능성을 배제하기 위해,
분할 시점을 여러 지점으로 바꿔가며(walk-forward) 같은 파이프라인(기존
lag_selected.csv의 N, full feature셋)으로 재학습·재평가하고 AUC가 분할
시점에 따라 얼마나 흔들리는지 확인한다.

방법: train_end를 2021-12-31 / 2022-06-30 / 2022-12-31(기존 논문 분할) /
2023-06-30 / 2023-12-31 5개 지점으로 바꿔가며, 각 지점 이전을 학습, 이후
전체(수집 종료 시점 2026-09-21까지)를 테스트로 사용해 로지스틱회귀 full
모델을 재학습하고 카테고리별 AUC를 계산한다. lag window(N)는 기존
lag_selected.csv 값을 그대로 쓴다(이 스크립트는 split 민감도만 검증하는
것이 목적이며, lag 선택 누수는 check_lag_selection_leakage.py에서 별도
검증했음).

사용법:
  python3 check_walkforward_splits.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from baseline_model import add_signal, load_features, load_lag_windows
from logistic_regression_model import add_lagged_seasonal, build_design_matrix

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1"]
SPLIT_POINTS = ["2021-12-31", "2022-06-30", "2022-12-31", "2023-06-30", "2023-12-31"]


def fit_auc_for_split(df, train_end):
    train = df[df["period"] <= train_end].dropna(subset=FEATURE_COLS + ["is_breakout"])
    test = df[df["period"] > train_end].dropna(subset=FEATURE_COLS + ["is_breakout"])

    if train["is_breakout"].nunique() < 2 or len(test) == 0:
        return None, len(train), len(test)

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
    return results, len(train), len(test)


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)

    rows = []
    for split in SPLIT_POINTS:
        auc, n_train, n_test = fit_auc_for_split(df, split)
        label = split + ("(기존 논문 분할)" if split == "2022-12-31" else "")
        print(f"\n=== train_end={label}  (train {n_train}행 / test {n_test}행) ===")
        if auc is None:
            print("  평가 불가(양성 클래스 부족 또는 테스트 데이터 없음)")
            continue
        print(pd.Series(auc).round(4))
        for category, value in auc.items():
            rows.append({"train_end": split, "category": category, "auc": value, "n_train": n_train, "n_test": n_test})

    result = pd.DataFrame(rows)
    pivot = result.pivot(index="category", columns="train_end", values="auc")
    valid_splits = [s for s in SPLIT_POINTS if s in pivot.columns]
    pivot["std_across_splits"] = pivot[valid_splits].std(axis=1)
    pivot["range"] = pivot[valid_splits].max(axis=1) - pivot[valid_splits].min(axis=1)
    print("\n=== 분할 시점별 AUC 요약 (카테고리 x train_end) ===")
    print(pivot.round(4))

    result.to_csv("walkforward_split_auc.csv", index=False, encoding="utf-8-sig")
    print("\n완료: walkforward_split_auc.csv")


if __name__ == "__main__":
    main()
