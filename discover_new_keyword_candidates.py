"""
RESEARCH_PLAN.md의 "주기적 후보군 갱신 파이프라인"(§5.2 "고정 후보군의 한계")
① 단계(재스크래핑) + ② 단계(diff)를 구현한다.

상태(2026-10-06): 이 세션(샌드박스)은 올리브영 도메인이 egress proxy에서
차단돼 직접 실행할 수 없었으나, 사용자가 로컬 Windows PC에서
`--debug-html`로 실제 검색결과 페이지(query=화장품)를 받아 공유해줘서
그 HTML로 parse_facets()를 실제 구조에 맞게 구현·검증했다(아래 참고).
requests는 Cloudflare로 차단됐고 playwright(헤드리스 브라우저)는 성공함
— 로컬에 playwright가 없으면 `pip install playwright` 후
`playwright install chromium` 필요. 실제로 "주요성분/제품특징/주요기능/
기능" 4개 필터 그룹, 각 10개 값을 정상 파싱해 기존 69개 후보 대비 신규
26개(스쿠알렌·시카케어·아미노산·타우린 등 진짜 신규 성분과, "진정"처럼
기존엔 결합형("피부진정")으로만 있던 값이 분리형으로 다시 잡히는 동음이의어
재검증 필요 사례가 섞여 있음 — 둘 다 3.1(3) 수동 검증으로 가려낼 대상)를
찾아내는 것까지 확인했다.

식약처(화장품 성분사전) 쪽은 안정적인 공개 URL을 아직 확인하지 못해
이번 버전에는 포함하지 않았다 — 올리브영 필터만으로도 성분/컨셉·클레임/
효능 3개 카테고리의 후보 발견은 가능하다(제형은 올리브영 "카테고리"
분류 쪽이라 별도 로직 필요, 이번 버전은 미포함). 식약처 쪽은 URL을
확인한 뒤 `fetch_mfds_candidates()`에 채워 넣을 것.

사용법(로컬 Windows/PowerShell, collect_naver_trends.py와 같은 폴더):
  pip install requests beautifulsoup4 playwright
  playwright install chromium
  python discover_new_keyword_candidates.py --debug-html   # 먼저 HTML 구조 확인
  python discover_new_keyword_candidates.py                 # 실제 수집 + diff

출력:
  oliveyoung_facets_raw.json : 수집된 필터 전체 값(원본, 재실행 시 캐시로 재사용)
  new_candidate_keywords.csv : naver_keywords_final.csv에 아직 없는 신규 값만
                                (keyword, category_guess, source_facet) —
                                이후 3.1(3) 동음이의어 수동 검증을 거쳐야
                                collect_naver_trends.py 그룹에 추가 가능.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

OLIVEYOUNG_SEARCH_URL = "https://www.oliveyoung.co.kr/store/search/getSearchMain.do"
RAW_JSON = Path("oliveyoung_facets_raw.json")
EXISTING_KEYWORDS_CSV = Path("naver_keywords_final.csv")
OUTPUT_CSV = Path("new_candidate_keywords.csv")

# KEYWORD_CANDIDATES.md에 기록된 필터 -> 카테고리 매핑.
# "제형"은 올리브영에서 필터가 아니라 카테고리 분류 체계라 이번 버전엔 없음(향후 추가).
FACET_TO_CATEGORY = {
    "주요성분": "성분",
    "제품특징": "컨셉·클레임",
    "주요기능": "효능",
    "기능": "효능",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.oliveyoung.co.kr/",
}


def fetch_with_requests(query="화장품"):
    resp = requests.get(OLIVEYOUNG_SEARCH_URL, params={"query": query}, headers=HEADERS, timeout=15)
    if resp.status_code != 200 or "cloudflare" in resp.text.lower()[:2000]:
        return None
    return resp.text


def fetch_with_playwright(query="화장품"):
    from playwright.sync_api import sync_playwright

    url = f"{OLIVEYOUNG_SEARCH_URL}?query={query}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(url, timeout=30000)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
    return html


def parse_facets(html):
    """
    2026-10-06 실제 오프라인 HTML(oliveyoung_debug.html, 사용자가 로컬에서
    playwright로 받아 공유)로 구조를 확인해 구현을 확정했다. 체크박스 그룹
    컨테이너(ul.filter > li > button.chip)를 거치지 않고, 체크박스 각각이
    attribute-name(필터 그룹명)·attribute-value(필터 값) 속성을 그대로
    갖고 있어 더 간단·안정적으로 파싱 가능:

      <input ... attribute-name="주요기능" attribute-value="보습" ...>

    FACET_TO_CATEGORY에 없는 그룹명(피부타입 등)은 자동으로 무시된다.
    """
    soup = BeautifulSoup(html, "html.parser")
    facets = {}
    for input_el in soup.select("input[attribute-name][attribute-value]"):
        facet_name = input_el.get("attribute-name", "").strip()
        value = input_el.get("attribute-value", "").strip()
        if facet_name not in FACET_TO_CATEGORY or not value:
            continue
        facets.setdefault(facet_name, set()).add(value)
    return facets


def fetch_mfds_candidates():
    """식약처 화장품 성분사전 스크래핑 — URL을 원격으로 확인할 수 없어 아직 미구현.
    로컬에서 실제 성분사전 URL을 확인한 뒤 이 함수를 채울 것."""
    return {}


def load_existing_keywords():
    df = pd.read_csv(EXISTING_KEYWORDS_CSV, encoding="utf-8-sig")
    return set(df["keyword"].str.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-html", action="store_true", help="HTML을 raw로 저장만 하고 종료(selector 확인용)")
    parser.add_argument("--force-playwright", action="store_true", help="requests 시도 없이 바로 playwright 사용")
    args = parser.parse_args()

    html = None
    if not args.force_playwright:
        html = fetch_with_requests()
        if html is None:
            print("requests 차단됨(Cloudflare 추정) — playwright로 재시도")
    if html is None:
        html = fetch_with_playwright()

    if args.debug_html:
        Path("oliveyoung_debug.html").write_text(html, encoding="utf-8")
        print("저장: oliveyoung_debug.html — 이 파일을 열어 실제 filter_group/filter_tit class명을 확인하고 parse_facets()를 수정할 것")
        return

    facets = parse_facets(html)
    facets.update(fetch_mfds_candidates())

    if RAW_JSON.exists():
        cached = json.loads(RAW_JSON.read_text(encoding="utf-8"))
        for k, v in cached.items():
            facets.setdefault(k, set()).update(v)
    RAW_JSON.write_text(
        json.dumps({k: sorted(v) for k, v in facets.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    existing = load_existing_keywords()
    rows = []
    for facet_name, values in facets.items():
        category = FACET_TO_CATEGORY.get(facet_name, "미분류")
        for value in values:
            if value not in existing:
                rows.append({"keyword": value, "category_guess": category, "source_facet": facet_name})

    result = pd.DataFrame(rows).drop_duplicates(subset=["keyword"])
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"기존 {len(existing)}개 후보 대비 신규 발견: {len(result)}개 -> {OUTPUT_CSV}")
    print("다음 단계(수동): 3.1(3) 동음이의어 검증 -> naver_keywords_final.csv에 추가 -> collect_naver_trends.py로 백필 수집")


if __name__ == "__main__":
    main()
