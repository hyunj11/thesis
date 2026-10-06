"""
RQ3(해외 선행 신호 검증)를 위해 GOOGLE_TRENDS_KEYWORD_MAPPING.md에 정리된
성분 22개의 영문 검색어로 Google Trends(기본 geo=US) 주간 상대 검색량을
수집한다. collect_naver_trends.py와 동일한 설계 원칙(원시 응답 캐싱,
중단 후 재실행 시 이어서 진행)을 따른다.

사전 준비:
  pip install pytrends

⚠️ Google Trends 해상도 제약과 처리 방법:
  Google Trends는 조회 구간이 약 5.2년(1900일)을 넘으면 자동으로 "주간"이
  아니라 "월간" 해상도로 내려간다. 네이버 데이터와 동일한 주간 해상도를
  2016-01-01~2026-09-21(10.7년) 전체에서 얻으려면 구간을 4년 단위로 쪼개서
  여러 번 요청해야 한다. 그런데 Google Trends의 0~100 정규화는 "요청 구간
  내에서만" 상대적이므로, 서로 다른 요청(청크)의 값은 그대로 이어붙일 수
  없다 — 청크 B의 100이 청크 A의 100과 같은 실제 검색량을 의미하지 않는다.
  이를 보정하기 위해 **구간을 겹치게(overlap) 설계**하고, 겹치는 기간의
  평균값 비율로 다음 청크 전체를 리스케일한 뒤 이어붙인다(표준적인 Google
  Trends 장기 시계열 스티칭 기법).

  청크 설계(각 4년, 6개월씩 겹침):
    A: 2016-01-01 ~ 2020-01-01
    B: 2019-07-01 ~ 2023-07-01  (A와 2019-07-01~2020-01-01 구간 겹침)
    C: 2023-01-01 ~ 2026-09-21  (B와 2023-01-01~2023-07-01 구간 겹침)

  리스케일: B 전체 값에 (겹치는 기간 A 평균 / 겹치는 기간 B 평균)을 곱해
  A와 같은 스케일로 맞춘 뒤, A는 겹침 시작 전까지만, B는 그 이후 전체를
  사용해 이어붙인다. C도 같은 방식으로 리스케일된 B에 맞춘다.

Rate limit 대응: Google Trends는 과도한 요청에 429를 반환하므로, 요청
사이 대기시간을 두고 429 발생 시 지수적으로 더 오래 대기 후 재시도한다.

실행:
  python collect_google_trends.py
  python collect_google_trends.py --geo US
  python collect_google_trends.py --geo ""   (전세계)

결과:
  - google_trends_raw/<키워드>.json : 청크별 원본 응답(리스케일 전, 감사용)
  - google_trends.csv : keyword, category, period, ratio (리스케일·스티칭 완료)
    naver_trends.csv와 동일한 컬럼 구조라 이후 분석 스크립트에서 그대로 재사용 가능.

중단 후 재실행하면 이미 raw json이 있는 키워드는 건너뛰므로 이어서 돌려도 됨.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path("google_trends_raw")
OUTPUT_CSV = Path("google_trends.csv")
MAPPING_CSV = Path("google_trends_keyword_mapping.csv")

CHUNKS = [
    ("2016-01-01", "2020-01-01"),
    ("2019-07-01", "2023-07-01"),
    ("2023-01-01", "2026-09-21"),
]

REQUEST_DELAY_SEC = 8.0  # Google Trends rate limit이 네이버보다 훨씬 엄격함
MAX_RETRIES = 5

# GOOGLE_TRENDS_KEYWORD_MAPPING.md와 동일한 매핑(검토 후 이 딕셔너리를 그 문서와
# 맞춰 수정할 것). "(결정 보류)"는 검토 후 최종 확정 필요.
KEYWORD_MAPPING = {
    "나이아신아마이드": "niacinamide",
    "PDRN": "PDRN",
    "판테놀": "panthenol",
    "펩타이드": "peptide skincare",
    "히알루론산": "hyaluronic acid",
    "세라마이드": "ceramide",
    "레티놀": "retinol",
    "아하 성분": "AHA skincare",
    "살리실산": "salicylic acid",
    "콜라겐": "collagen skincare",
    "스쿠알란": "squalane",
    "병풀추출물": "centella asiatica",
    "엑소좀": "exosome skincare",
    "장미추출물": "rose extract skincare",
    "어성초": "houttuynia cordata",
    "알란토인": "allantoin",
    "아데노신": "adenosine skincare",
    "베타글루칸": "beta glucan skincare",
    "트레할로스": "trehalose",
    "감초추출물": "licorice extract skincare",
    "비타민E": "vitamin e skincare",
    "시어버터": "shea butter",
}


def already_fetched(keyword):
    return (RAW_DIR / f"{keyword}.json").exists()


def fetch_chunk(pytrends, term, start, end, geo):
    from pytrends.request import TrendReq  # noqa: F401 (타입 참고용)

    for attempt in range(MAX_RETRIES):
        try:
            pytrends.build_payload([term], timeframe=f"{start} {end}", geo=geo)
            df = pytrends.interest_over_time()
            return df
        except Exception as e:  # pytrends는 429를 requests.exceptions로 감싸서 던짐
            wait = REQUEST_DELAY_SEC * (2 ** attempt)
            print(f"    재시도 {attempt + 1}/{MAX_RETRIES} ({e}) — {wait:.0f}초 대기")
            time.sleep(wait)
    raise RuntimeError(f"{term} ({start}~{end}) 수집 실패 (재시도 초과)")


def stitch_chunks(dfs, term):
    """겹치는 구간의 평균 비율로 순차 리스케일 후 이어붙인다."""
    base = dfs[0][[term]].rename(columns={term: "ratio"}).copy()
    base.index.name = "period"

    for df in dfs[1:]:
        cur = df[[term]].rename(columns={term: "ratio"}).copy()
        cur.index.name = "period"
        overlap_idx = base.index.intersection(cur.index)
        if len(overlap_idx) == 0:
            raise ValueError(f"{term}: 청크 간 겹치는 구간이 없음 — CHUNKS 설계 확인 필요")

        base_overlap_mean = base.loc[overlap_idx, "ratio"].mean()
        cur_overlap_mean = cur.loc[overlap_idx, "ratio"].mean()
        scale = (base_overlap_mean / cur_overlap_mean) if cur_overlap_mean > 0 else 0.0
        cur["ratio"] = cur["ratio"] * scale

        # base는 겹침 시작 전까지만, cur는 겹침 시작부터 전체를 사용해 이어붙임
        cutoff = overlap_idx.min()
        base = pd.concat([base[base.index < cutoff], cur])

    return base.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", default="US", help="Google Trends geo 코드 (기본 US, 전세계는 \"\")")
    args = parser.parse_args()

    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)

    RAW_DIR.mkdir(exist_ok=True)

    todo = [(kw, term) for kw, term in KEYWORD_MAPPING.items() if not already_fetched(kw)]
    print(f"전체 {len(KEYWORD_MAPPING)}개 중 이미 수집됨 {len(KEYWORD_MAPPING) - len(todo)}개, 남은 작업 {len(todo)}개")

    for keyword, term in todo:
        print(f"[{keyword} -> \"{term}\"] 수집 중 (geo={args.geo or 'Worldwide'})")
        chunk_dfs = []
        for start, end in CHUNKS:
            df = fetch_chunk(pytrends, term, start, end, args.geo)
            if df.empty or term not in df.columns:
                print(f"    경고: {start}~{end} 구간 데이터 없음(검색량 0 가능성)")
                df = pd.DataFrame({term: []})
            chunk_dfs.append(df)
            time.sleep(REQUEST_DELAY_SEC)

        non_empty = [df for df in chunk_dfs if not df.empty]
        if not non_empty:
            print(f"    {keyword}: 전체 구간에서 검색량 없음 — 건너뜀(분석에서 자동 제외됨)")
            with open(RAW_DIR / f"{keyword}.json", "w", encoding="utf-8") as f:
                json.dump({"term": term, "empty": True}, f, ensure_ascii=False)
            continue

        stitched = stitch_chunks(non_empty, term)
        stitched.to_json(RAW_DIR / f"{keyword}.json", orient="records", date_format="iso", force_ascii=False)
        print(f"    완료: {len(stitched)}행")

    build_combined_csv()
    print(f"완료. 결과: {OUTPUT_CSV}")


def build_combined_csv():
    category = "성분"  # 이번 수집은 성분 카테고리 22개 한정(RQ3 1단계 범위)
    rows = []
    for keyword in KEYWORD_MAPPING:
        path = RAW_DIR / f"{keyword}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("empty"):
            continue
        for row in data:
            rows.append(
                {
                    "keyword": keyword,
                    "category": category,
                    "period": pd.to_datetime(row["period"]).strftime("%Y-%m-%d"),
                    "ratio": row["ratio"],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
