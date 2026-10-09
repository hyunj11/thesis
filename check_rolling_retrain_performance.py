"""
RQ1 예측 성능을 더 끌어올릴 방법 탐색 (3) — 분기별 롤링 재학습(walk-forward
retraining).

§5.2 "단일 train/test 분할에 대한 결과 민감도"는 분할 시점을 5개로 바꿔가며
각각 "그 시점 이전 전부 학습 / 그 이후 전부 테스트"를 봤을 뿐, 실제 운영
시나리오(분기마다 최신 데이터로 재학습해서 그 다음 분기만 평가)를 흉내낸
적은 없었다. 이 스크립트는 2023~2025 테스트 구간을 분기 단위로 쪼개, 각
분기 직전까지의 데이터로 재학습한 뒤 그 분기만 평가하는 롤링 백테스팅을
수행해, 고정 분할(2022-12-31 기준 1회 학습) 대비 평균 성능이 더 나은지
확인한다.

사용법:
  python3 check_rolling_retrain_performance.py
"""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from baseline_model import add_signal, load_features, load_lag_windows
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing, precision_recall_at_k
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]

# 2023~2025, 분기 단위 (분기 시작일, 분기 종료일)
QUARTERS = [
    ("2023-01-01", "2023-03-31"), ("2023-04-01", "2023-06-30"),
    ("2023-07-01", "2023-09-30"), ("2023-10-01", "2023-12-31"),
    ("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
]


def fit_predict_quarter(df, train_end, q_start, q_end):
    train = df[df["period"] <= train_end]
    test = df[(df["period"] >= q_start) & (df["period"] <= q_end)]
    train = drop_missing(train, FEATURE_COLS).dropna(subset=["momentum", "rank_in_category"])
    test = drop_missing(test, FEATURE_COLS).dropna(subset=["momentum", "rank_in_category"])
    if len(test) == 0 or train["is_breakout"].nunique() < 2:
        return None

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
    return test, len(train)


def main():
    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    print("=== 분기별 롤링 재학습 (각 분기 직전까지 데이터로 재학습 → 그 분기만 평가) ===")
    rows = []
    all_test_rolling = []
    for q_start, q_end in QUARTERS:
        train_end = (pd.Timestamp(q_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        result = fit_predict_quarter(df, train_end, q_start, q_end)
        if result is None:
            print(f"{q_start}~{q_end}: 평가 불가(데이터 부족)")
            continue
        test_pred, n_train = result
        all_test_rolling.append(test_pred)
        for category, group in test_pred.groupby("category"):
            y = group["is_breakout"].astype(int)
            auc = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
            rows.append({"quarter": q_start, "category": category, "auc": auc, "n_train": n_train, "n_test": len(group)})

    rolling_result = pd.DataFrame(rows)
    pivot = rolling_result.pivot(index="category", columns="quarter", values="auc")
    print(pivot.round(4))
    rolling_mean_by_cat = rolling_result.groupby("category")["auc"].mean()
    print(f"\n분기별 롤링 재학습 — 카테고리별 평균 AUC(분기 단순평균):\n{rolling_mean_by_cat.round(4)}")
    print(f"전체 평균: {rolling_mean_by_cat.mean():.4f}")

    # 롤링 전체 테스트 구간을 합쳐서(풀링) AUC도 계산 — 고정 분할과 직접 비교 가능하게
    rolling_pooled = pd.concat(all_test_rolling)
    pooled_auc = {}
    for category, group in rolling_pooled.groupby("category"):
        y = group["is_breakout"].astype(int)
        pooled_auc[category] = roc_auc_score(y, group["pred_proba"]) if y.nunique() > 1 else float("nan")
    pooled_auc = pd.Series(pooled_auc)
    pr1_rolling = precision_recall_at_k(rolling_pooled, 1)

    print(f"\n=== 롤링(풀링) vs 고정 분할(§4.1) 비교 ===")
    fixed_auc = pd.Series({"성분": 0.7161, "제형": 0.7637, "컨셉·클레임": 0.8253, "효능": 0.8691})
    comparison = pd.DataFrame({"fixed_split(§4.1)": fixed_auc, "rolling_retrain(pooled)": pooled_auc})
    comparison["diff"] = comparison["rolling_retrain(pooled)"] - comparison["fixed_split(§4.1)"]
    print(comparison.round(4))
    print(f"\n평균 AUC: fixed={fixed_auc.mean():.4f}, rolling={pooled_auc.mean():.4f}")
    print(f"롤링 Precision@1 평균: {pr1_rolling['precision_at_k'].mean():.4f} (고정 분할 §4.1: 0.2500)")

    rolling_result.to_csv("rolling_retrain_quarterly_auc.csv", index=False, encoding="utf-8-sig")
    print("\n완료: rolling_retrain_quarterly_auc.csv")


if __name__ == "__main__":
    main()
