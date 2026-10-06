"""
RQ3(해외 선행 신호 검증)를 위해 GOOGLE_TRENDS_KEYWORD_MAPPING.md에 정리된
성분 22개의 영문 검색어로 Google Trends(기본 geo=US) 검색량을 수집한다.
collect_naver_trends.py와 동일한 설계 원칙(원시 응답 캐싱, 중단 후 재실행
시 이어서 진행)을 따른다.

사전 준비:
  pip install pytrends

⚠️ 해상도: 월간(이전 주간+청크스티칭 버전의 문제를 피하기 위한 설계 변경).
  최초 버전은 10.7년을 4년 단위 3개 청크로 나눠 주간 해상도로 받고 겹치는
  구간의 평균 비율로 리스케일해 이어붙이는 방식이었는데, 실제로 돌려보니
  나이아신아마이드처럼 수년간 폭발적으로 성장한 키워드에서 청크 경계마다
  값이 점프하는 문제가 발견됐다(겹침 구간 평균이 두 청크에서 크게 달라
  리스케일 배율이 왜곡됨 — 2026-10-06 실데이터로 확인). Google Trends는
  한 번의 요청으로 10년 전체를 조회하면 리스케일 없이 그 구간 전체에서
  일관된 0~100 척도를 돌려주므로(대신 해상도가 자동으로 월간으로 내려감),
  이 버전은 청크/스티칭을 없애고 단일 요청으로 전환해 이 문제를 근본적으로
  피한다. RQ3 1단계(선행관계 존재 여부 확인)는 월간 해상도로도 충분하다.

Rate limit 대응: Google Trends는 과도한 요청에 429를 반환하므로, 요청
사이 대기시간을 두고 429 발생 시 지수적으로 더 오래 대기 후 재시도한다.

실행:
  python collect_google_trends.py
  python collect_google_trends.py --geo US
  python collect_google_trends.py --geo ""   (전세계)

결과:
  - google_trends_raw/<키워드>.json : 원본 응답(월간, 리스케일 없음)
  - google_trends.csv : keyword, category, period, ratio
    naver_trends.csv와 동일한 컬럼 구조.

중단 후 재실행하면 이미 raw json이 있는 키워드는 건너뛰므로 이어서 돌려도 됨.
이전(청크/스티칭) 버전으로 받은 google_trends_raw/*.json·google_trends.csv는
반드시 삭제하고 다시 받아야 한다 — 그대로 두면 "이미 수집됨"으로 건너뛴다.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path("google_trends_raw")
OUTPUT_CSV = Path("google_trends.csv")

START_DATE = "2016-01-01"
END_DATE = "2026-09-21"

REQUEST_DELAY_SEC = 8.0  # Google Trends rate limit이 네이버보다 훨씬 엄격함
MAX_RETRIES = 5

# GOOGLE_TRENDS_KEYWORD_MAPPING.md와 동일한 매핑(검토 후 이 딕셔너리를 그 문서와
# 맞춰 수정할 것).
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


def fetch_full_range(pytrends, term, geo):
    for attempt in range(MAX_RETRIES):
        try:
            pytrends.build_payload([term], timeframe=f"{START_DATE} {END_DATE}", geo=geo)
            df = pytrends.interest_over_time()
            return df
        except Exception as e:  # pytrends는 429를 requests.exceptions로 감싸서 던짐
            wait = REQUEST_DELAY_SEC * (2 ** attempt)
            print(f"    재시도 {attempt + 1}/{MAX_RETRIES} ({e}) — {wait:.0f}초 대기")
            time.sleep(wait)
    raise RuntimeError(f"{term} 수집 실패 (재시도 초과)")


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
        print(f"[{keyword} -> \"{term}\"] 수집 중 (geo={args.geo or 'Worldwide'}, {START_DATE}~{END_DATE})")
        df = fetch_full_range(pytrends, term, args.geo)

        if df.empty or term not in df.columns:
            print(f"    {keyword}: 검색량 없음 — 건너뜀(분석에서 자동 제외됨)")
            with open(RAW_DIR / f"{keyword}.json", "w", encoding="utf-8") as f:
                json.dump({"term": term, "empty": True}, f, ensure_ascii=False)
            time.sleep(REQUEST_DELAY_SEC)
            continue

        records = df[[term]].rename(columns={term: "ratio"}).reset_index()
        records["period"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
        records = records[["period", "ratio"]]
        records.to_json(RAW_DIR / f"{keyword}.json", orient="records", force_ascii=False)
        print(f"    완료: {len(records)}행 (월간)")
        time.sleep(REQUEST_DELAY_SEC)

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
                    "period": row["period"],
                    "ratio": row["ratio"],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
