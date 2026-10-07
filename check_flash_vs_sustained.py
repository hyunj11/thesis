"""
사용자 제안 "4. Flash vs Sustained 유형 예측": "이 키워드가 뜰 것인가"(1단계,
§4.1의 기존 분류기)에 이어, **"뜬다면 반짝 꺼질 것인가(flash), 오래 갈
것인가(sustained)"**를 예측하는 2단계 분류기를 추가한다. §4.2에서 이미
계산해 둔 half-life·censored 정보(`category_volatility_halflife_events.csv`)
를 라벨로 재활용한다 — 성분 카테고리가 "왜" 예측이 어려운지에 대한 §4.2·5.1의
해석(장기 지속형 패턴을 단기 급등 전제의 라벨링으로는 포착하기 어렵다)을
직접 다루는 후속 질문이기도 하다.

라벨 정의: 카테고리별로 half-life 중앙값(성분 5주/제형 8주/컨셉·클레임
3주/효능 4주, §4.2)보다 길게 유지되거나 아직도 꺾이지 않은(censored)
breakout 사건을 sustained=True, 그 외(중앙값 이하로 빨리 식은 사건)를
flash로 정의한다.

feature: breakout이 발생한 시점(T)의 signal/volatility/seasonal_deviation/
momentum/rank_in_category — "뜨는 순간의 특성"으로 "얼마나 갈지"를
예측할 수 있는지를 본다(1단계 분류기와 동일한 feature, T 시점 가치를
그대로 사용 — 이 사건 자체가 breakout이므로 미래 정보 누수가 아니라
"이 사건의 당시 특성"을 보는 것).

사용법:
  python3 check_flash_vs_sustained.py
"""

import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from baseline_model import TRAIN_END, add_signal, load_features, load_lag_windows
from logistic_regression_model import add_lagged_seasonal, build_design_matrix, drop_missing
from model_comparison_experiment import add_extra_features

FEATURE_COLS = ["signal", "volatility", "seasonal_deviation_pct_lag1", "momentum", "rank_in_category"]


def build_dataset():
    events = pd.read_csv("category_volatility_halflife_events.csv", encoding="utf-8-sig")
    events["breakout_period"] = pd.to_datetime(events["breakout_period"])

    median_halflife = (
        events[~events["censored"]].groupby("category")["halflife_weeks"].median()
    )
    events["halflife_median_for_category"] = events["category"].map(median_halflife)
    events["sustained"] = events["censored"] | (
        events["halflife_weeks"] > events["halflife_median_for_category"]
    )

    df = load_features()
    lag_by_category = load_lag_windows()
    df = add_signal(df, lag_by_category)
    df = add_lagged_seasonal(df)
    df = add_extra_features(df)

    merged = events.merge(
        df[["keyword", "category", "period"] + FEATURE_COLS],
        left_on=["keyword", "category", "breakout_period"],
        right_on=["keyword", "category", "period"],
        how="left",
    )
    return merged


def main():
    data = build_dataset()
    print(f"breakout 사건 {len(data)}건, sustained 비율: {data['sustained'].mean():.1%}")
    print(data.groupby("category")["sustained"].mean().round(3))

    train = data[data["breakout_period"] <= TRAIN_END]
    test = data[data["breakout_period"] > TRAIN_END]

    train = train.dropna(subset=FEATURE_COLS + ["sustained"]).copy()
    test = test.dropna(subset=FEATURE_COLS + ["sustained"]).copy()
    print(f"\n학습 {len(train)}건 / 테스트 {len(test)}건")

    X_train = build_design_matrix(train, FEATURE_COLS)
    X_test = build_design_matrix(test, FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)
    y_train = train["sustained"].astype(int)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    model = CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        scale_pos_weight=(n_neg / n_pos if n_pos > 0 else 1.0),
        random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)

    test = test.copy()
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]

    print("\n=== 카테고리별 flash(0) vs sustained(1) 예측 AUC ===")
    for category, group in test.groupby("category"):
        y = group["sustained"].astype(int)
        if y.nunique() > 1:
            print(f"  {category}: AUC={roc_auc_score(y, group['pred_proba']):.4f} (n={len(y)}, sustained {y.sum()}건)")
        else:
            print(f"  {category}: 평가 불가(한쪽 클래스 없음, n={len(y)})")

    overall_y = test["sustained"].astype(int)
    print(f"\n전체(4개 카테고리 풀링) AUC: {roc_auc_score(overall_y, test['pred_proba']):.4f}")

    importances = dict(zip(X_train.columns, model.get_feature_importance().round(3)))
    print("feature importance:", importances)

    print("\n=== SHAP 전역·방향성 (sustained=1 기준) ===")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_train)
    shap_df = pd.DataFrame(sv, columns=X_train.columns, index=X_train.index)
    for col in FEATURE_COLS:
        high = X_train[col] >= X_train[col].quantile(0.75)
        low = X_train[col] <= X_train[col].quantile(0.25)
        print(f"  {col}: mean|SHAP|={shap_df[col].abs().mean():.4f}  "
              f"high25%={shap_df.loc[high, col].mean():.4f}  low25%={shap_df.loc[low, col].mean():.4f}")

    print("\n=== 카테고리별 |SHAP| ===")
    rows = []
    for category in sorted(train["category"].unique()):
        idx = train.index[train["category"] == category].intersection(shap_df.index)
        for col in FEATURE_COLS:
            rows.append({"category": category, "feature": col, "mean_abs_shap": shap_df.loc[idx, col].abs().mean()})
    print(pd.DataFrame(rows).pivot(index="feature", columns="category", values="mean_abs_shap").round(4))

    test[["keyword", "category", "breakout_period", "sustained", "pred_proba"]].to_csv(
        "flash_vs_sustained_results.csv", index=False, encoding="utf-8-sig"
    )
    print("\n완료: flash_vs_sustained_results.csv")


if __name__ == "__main__":
    main()
