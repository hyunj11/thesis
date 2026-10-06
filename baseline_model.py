"""
RESEARCH_PLAN.md "4) 모델"의 베이스라인("최근 증가율 상위 K개를 단순 선택")과
"5) 검증(백테스팅)"을 구현한다.

방법론:
  - 신호(signal): select_lag_window.py가 카테고리별로 선택한 교차상관 최적
    시차 N(성분 14주/제형 22주/컨셉·클레임 7주/효능 24주)만큼 growth_rate를
    과거로 이동(shift)한 값. "최근 N주 전 증가율"을 Top-K 랭킹 기준으로
    쓴다는 점에서 RESEARCH_PLAN.md 3) Feature 설계의 lag 선택 결과와
    일관된 정의를 그대로 재사용한다(미래 정보 누수 없음 — is_breakout_t는
    growth_rate_t로부터 정의되는데, signal은 t-N 시점 값이라 N>=1이므로
    같은 시점 정보가 섞이지 않음).
  - 베이스라인 자체는 학습 파라미터가 없는 휴리스틱(규칙 기반)이라 "학습"이
    필요 없지만, 이후 로지스틱회귀/XGBoost와 동일한 조건에서 비교하기 위해
    RESEARCH_PLAN.md 5) 검증 절의 백테스팅 split을 동일하게 적용한다:
    ~2022는 학습 구간(베이스라인에는 미사용, 이후 본 모델과의 비교용으로
    구간만 맞춰둠), 2023~2025를 테스트(백테스팅) 구간으로 평가.
  - 매 (카테고리, period) 시점마다 해당 카테고리 키워드들을 signal 내림차순
    정렬 → 상위 K개를 breakout 예측(1)으로 간주, 나머지는 0.
  - 평가지표:
    - Precision@K / Recall@K: 모든 테스트 기간의 혼동행렬(tp/fp/fn)을
      합산한 뒤(micro-average) 계산 — 기간별로 실제 breakout이 드물어
      기간 단위로 평균 내면 분모가 0인 경우가 많아 불안정하기 때문.
    - AUC: 카테고리별로 signal(연속값)과 실제 is_breakout(이진) 간
      roc_auc_score로 계산 — Top-K처럼 임계치(K)에 의존하지 않는 랭킹
      품질 자체를 보기 위함.

사용법:
  python3 baseline_model.py
  python3 baseline_model.py --k 1 3 5 10
"""

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"


def load_features(path="naver_trend_features.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["category", "keyword", "period"])


def load_lag_windows(path="lag_selected.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(df["category"], df["lag_weeks"]))


def add_signal(df, lag_by_category):
    df = df.copy()
    df["signal"] = float("nan")
    for category, lag in lag_by_category.items():
        mask = df["category"] == category
        df.loc[mask, "signal"] = df.loc[mask].groupby("keyword")["growth_rate"].shift(lag)
    return df


def split_backtest(df):
    train = df[df["period"] <= TRAIN_END]
    test = df[(df["period"] >= TEST_START) & (df["period"] <= TEST_END)]
    return train, test


SEASONAL_COL = "seasonal_deviation_pct_lag1"
CORE_FULL_COLS = ["signal", "volatility"]


def prepare_full_feature_rows(df):
    """`full` feature셋(signal+volatility+계절편차)을 쓰는 모델이 공통으로
    적용해야 하는 결측 처리 규칙.

    계절편차는 "과거 연도 1개 이상"이 있어야 계산되는데, 수집 시작일
    (2020-12-28)로부터 1년이 지나야 충족되므로 그 이전 구간은 전부 NaN이다.
    signal·volatility까지 전부 유효한 행을 계절편차 NaN이라는 이유만으로
    통째로 버리면, 학습 구간이 실질적으로 2022-01~2022-12(3,200행, 1년
    남짓)로 줄어드는 문제가 있었다(§5.2 "수집 기간" 항목, 2026-10-05
    발견). 이 결측을 **0(= "아직 계절 기준선을 계산할 수 없으므로 이례적
    편차가 있다고 볼 근거가 없다"는 중립값)으로 채워** 포함시키면, 미래
    정보를 끌어오는 것이 아니면서도 학습 구간을 2021-02~2022-12로 확장할
    수 있다(`check_seasonal_impute_extension.py`로 검증: 테스트 AUC 변화
    -0.001~+0.002로 사실상 無영향, 학습 행은 3,200→5,453으로 확장).

    signal·volatility 자체가 없는 행(lag 윈도 자체가 아직 안 쌓인 극초반
    구간)은 여전히 제외한다 — 이건 보수적으로 채울 중립값이 없는 "진짜
    결측"이기 때문이다.
    """
    df = df.dropna(subset=CORE_FULL_COLS + ["is_breakout"]).copy()
    df[SEASONAL_COL] = df[SEASONAL_COL].fillna(0)
    return df


def precision_recall_at_k(test_df, k):
    rows = []
    for (category, period), group in test_df.groupby(["category", "period"]):
        group = group.dropna(subset=["signal"])
        if len(group) == 0:
            continue
        top_k = group.sort_values("signal", ascending=False).head(k)
        tp = int(top_k["is_breakout"].sum())
        fp = len(top_k) - tp
        fn = int(group["is_breakout"].sum()) - tp
        rows.append({"category": category, "period": period, "tp": tp, "fp": fp, "fn": fn})

    agg = pd.DataFrame(rows).groupby("category")[["tp", "fp", "fn"]].sum()
    agg["precision_at_k"] = agg["tp"] / (agg["tp"] + agg["fp"]).replace(0, pd.NA)
    agg["recall_at_k"] = agg["tp"] / (agg["tp"] + agg["fn"]).replace(0, pd.NA)
    return agg


def auc_by_category(test_df):
    results = {}
    for category, group in test_df.groupby("category"):
        group = group.dropna(subset=["signal"])
        y = group["is_breakout"].astype(int)
        if y.nunique() < 2:
            results[category] = float("nan")
            continue
        results[category] = roc_auc_score(y, group["signal"])
    return pd.Series(results, name="auc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="naver_trend_features.csv")
    parser.add_argument("--lag-input", default="lag_selected.csv")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", default="baseline_results.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    lag_by_category = load_lag_windows(args.lag_input)
    df = add_signal(df, lag_by_category)

    train, test = split_backtest(df)
    print(f"학습 구간(~{TRAIN_END}, 베이스라인 미사용): {len(train)}행")
    print(f"테스트(백테스팅) 구간({TEST_START}~{TEST_END}): {len(test)}행")

    auc = auc_by_category(test)

    all_results = []
    for k in args.k:
        pr = precision_recall_at_k(test, k)
        pr["k"] = k
        all_results.append(pr.reset_index())

    results = pd.concat(all_results, ignore_index=True)
    results = results.merge(auc.reset_index().rename(columns={"index": "category"}), on="category")

    print("\n베이스라인(최근 N주 전 증가율 Top-K) 백테스팅 결과:")
    print(
        results[["category", "k", "tp", "fp", "fn", "precision_at_k", "recall_at_k", "auc"]]
        .sort_values(["category", "k"])
        .to_string(index=False)
    )

    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n완료: {args.output}")


if __name__ == "__main__":
    main()
