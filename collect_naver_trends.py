"""
네이버 데이터랩(DataLab) 검색어트렌드 API로 naver_keywords_final.csv의
키워드별 주간 검색량 추이를 수집한다.

사전 준비:
  1. NAVER API HUB(NCP 콘솔 > Application Service > NAVER API HUB)에서
     애플리케이션 등록 (사용 API: "Search Trend"/"검색어트렌드")
  2. 발급받은 Client ID(X-NCP-APIGW-API-KEY-ID) / Client Secret
     (X-NCP-APIGW-API-KEY)을 config.json에 기록
     (config.example.json을 복사해서 config.json으로 만들고 값만 채우면 됨.
      config.json은 .gitignore에 등록돼 있어 깃에 올라가지 않음)

  * 파이썬 기본 모듈(urllib)만 사용하므로 pip install이 따로 필요 없음.

실행:
  python collect_naver_trends.py
  python collect_naver_trends.py --start-date 2016-01-01 --end-date 2026-09-21 --time-unit week

결과:
  - naver_trends_raw/<키워드>.json : API 원본 응답(감사·재현용, 깃에는 안 올라감)
  - naver_trends.csv : keyword, category, period, ratio 형태로 정리된 결과(이건 커밋 대상)

중단 후 재실행하면 이미 raw json이 있는 키워드는 건너뛰므로 이어서 돌려도 됨.
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
CONFIG_PATH = Path("config.json")
KEYWORDS_CSV = Path("naver_keywords_final.csv")
RAW_DIR = Path("naver_trends_raw")
OUTPUT_CSV = Path("naver_trends.csv")

GROUPS_PER_REQUEST = 5   # 데이터랩 API 제한: 요청당 최대 5개 그룹
REQUEST_DELAY_SEC = 1.0  # 요청 사이 대기(rate limit 여유)


def load_config():
    if not CONFIG_PATH.exists():
        raise SystemExit(
            "config.json이 없습니다. config.example.json을 복사해서 "
            "config.json으로 만들고 client_id/client_secret을 채워주세요."
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_keywords():
    with open(KEYWORDS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def already_fetched(keyword):
    return (RAW_DIR / f"{safe_filename(keyword)}.json").exists()


def safe_filename(keyword):
    return keyword.replace("/", "_").replace(" ", "_")


def fetch_batch(headers, keyword_rows, start_date, end_date, time_unit):
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": [
            {"groupName": row["keyword"], "keywords": [row["keyword"]]}
            for row in keyword_rows
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API 오류 {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"네트워크 오류: {e.reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-09-21")
    parser.add_argument("--time-unit", default="week", choices=["date", "week", "month"])
    args = parser.parse_args()

    config = load_config()
    headers = {
        "X-NCP-APIGW-API-KEY-ID": config["client_id"],
        "X-NCP-APIGW-API-KEY": config["client_secret"],
        "Content-Type": "application/json",
    }

    RAW_DIR.mkdir(exist_ok=True)
    all_rows = load_keywords()
    todo = [r for r in all_rows if not already_fetched(r["keyword"])]
    print(f"전체 {len(all_rows)}개 중 이미 수집됨 {len(all_rows) - len(todo)}개, 남은 작업 {len(todo)}개")

    for i in range(0, len(todo), GROUPS_PER_REQUEST):
        batch = todo[i : i + GROUPS_PER_REQUEST]
        names = ", ".join(r["keyword"] for r in batch)
        print(f"[{i // GROUPS_PER_REQUEST + 1}] 요청: {names}")
        try:
            data = fetch_batch(headers, batch, args.start_date, args.end_date, args.time_unit)
        except Exception as e:
            print(f"  실패: {e}")
            time.sleep(REQUEST_DELAY_SEC * 3)
            continue

        for result in data.get("results", []):
            keyword = result["title"]
            with open(RAW_DIR / f"{safe_filename(keyword)}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        time.sleep(REQUEST_DELAY_SEC)

    build_combined_csv(all_rows)
    print(f"완료. 결과: {OUTPUT_CSV}")


def build_combined_csv(all_rows):
    category_by_keyword = {r["keyword"]: r["category"] for r in all_rows}
    out_rows = []
    for keyword, category in category_by_keyword.items():
        path = RAW_DIR / f"{safe_filename(keyword)}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        for point in result.get("data", []):
            out_rows.append(
                {
                    "keyword": keyword,
                    "category": category,
                    "period": point["period"],
                    "ratio": point["ratio"],
                }
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["keyword", "category", "period", "ratio"])
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
