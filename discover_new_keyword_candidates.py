"""
RESEARCH_PLAN.md의 "주기적 후보군 갱신 파이프라인"(§5.2 "고정 후보군의 한계")
① 단계(재스크래핑) + ② 단계(diff)를 구현한다.

⚠️ 중요한 한계(투명하게 밝힘): 이 스크립트는 이 세션(샌드박스)에서는 전혀
테스트하지 못했다 — 올리브영·식약처 등 외부 도메인이 이 환경의 egress
proxy에서 전부 403으로 차단되기 때문이다(collect_naver_trends.py/
collect_google_trends.py와 동일한 제약). 따라서:
  - 올리브영 검색결과 페이지의 실제 HTML 구조(class명 등)는
    KEYWORD_CANDIDATES.md에 기록된 과거 관찰(`getSearchMain.do?query=...`의
    "주요성분"·"제품특징"·"주요기능" 필터)에 근거해 작성했지만, 사이트
    개편으로 바뀌어 있을 수 있다 — 반드시 로컬에서 먼저
    `--debug-html`로 실제 HTML을 저장해 CSS selector를 맞게 조정할 것.
  - 3.1(4)에서 이미 올리브영 랭킹 페이지 수집 시 Cloudflare 봇 탐지에
    걸렸던 선례가 있다. 아래는 1차로 requests + 브라우저 유사 헤더를
    시도하고, 차단되면 playwright(헤드리스 브라우저)로 자동 전환한다.
    로컬에 playwright가 없으면 `pip install playwright` 후
    `playwright install chromium` 필요.
  - 식약처(화장품 성분사전) 쪽은 안정적인 공개 URL을 이 자리에서
    확신 있게 특정할 수 없어(원격 확인 불가), 이번 버전에는 포함하지
    않았다 — 올리브영 필터만으로도 성분/컨셉·클레임/효능 3개 카테고리의
    후보 발견은 가능하다(제형은 올리브영 "카테고리" 분류 쪽이라 별도
    로직 필요, 이번 버전은 미포함). 식약처 쪽은 URL을 직접 확인한 뒤
    `fetch_mfds_candidates()`에 채워 넣을 것.

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
    ⚠️ 로컬에서 --debug-html로 실제 구조를 먼저 확인하고 이 함수의 selector를
    맞게 고칠 것. 아래는 KEYWORD_CANDIDATES.md 관찰(필터 영역에 facet 그룹명과
    체크박스 라벨 텍스트가 존재)에 근거한 추정 구현이며, 실제 class/id명은
    확인되지 않았다.
    """
    soup = BeautifulSoup(html, "html.parser")
    facets = {}
    # 추정: 필터 그룹이 <div class="filter_group"> 안에 그룹명(.filter_tit)과
    # 체크박스 라벨(label) 목록으로 구성됨 — 실제 구조 확인 후 수정 필요.
    for group in soup.select(".filter_group"):
        title_el = group.select_one(".filter_tit")
        if not title_el:
            continue
        facet_name = title_el.get_text(strip=True)
        if facet_name not in FACET_TO_CATEGORY:
            continue
        values = [label.get_text(strip=True) for label in group.select("label") if label.get_text(strip=True)]
        facets.setdefault(facet_name, set()).update(values)
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
