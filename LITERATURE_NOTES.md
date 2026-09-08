# 선행연구 조사 노트 (1차 스크리닝)

> 웹 검색으로 찾은 관련 자료 목록입니다. 초록/스니펫만 확인한 상태라, 실제로 각 논문을 원문으로 읽고 우리 연구와의 관련성·인용 여부를 다시 판단해야 합니다. 특히 DBpia/KCI 논문은 을지대 도서관 계정으로 원문 접근 필요.

## 목적
Ⅱ장 "이론적 배경"을 아래 3개 축으로 구성하기로 했음 (RESEARCH_PLAN.md 참고):
- 2.1 화장품 트렌드와 소비자 검색 행동
- 2.2 시계열 예측 / 트렌드 조기탐지(breakout detection)
- 2.3 국내 뷰티 이커머스 데이터 활용

---

## 2.1 화장품 트렌드와 소비자 검색 행동

- **"빅데이터 분석을 통한 화장품 트렌드 변화 추이에 관한 연구 - 텍스트마이닝과 의미연결망 분석을 중심으로 -"**
  DBpia: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10700385
  KCI: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002706753
  메모: 네이버·다음 블로그/뉴스/카페 텍스트를 텍스트마이닝+의미연결망 분석. 코로나19 전후 화장품 트렌드 변화(기초화장품·성분 관심 증가, 브랜드 연결성 약화) 다룸. **가장 직접적으로 관련된 국내 논문으로 보임 — 원문 꼭 확인 필요.**

- **화해(Hwahae) 비즈니스 인사이트 "2026년 뷰티 트렌드 총정리"**
  https://business.hwahae.co.kr/insight/trendreport-2026-summary/
  메모: 학술논문은 아니고 업계 리포트지만, "왜 이 연구가 실무적으로 필요한가"의 배경 자료로 서론에 인용 가능.

- **Naver DataLab 관련 해외 소개 자료**
  https://www.theegg.com/seo/korea/naver-data-lab-how-to-gain-insights-on-korean-search-trends/
  메모: 학술자료 아님, 데이터 소스 설명용 참고자료로만 사용 가능.

- **K-beauty 성분 트렌드 리포트 (Spate 데이터 기반, BeautyMatter)**
  https://beautymatter.com/articles/2026-k-beauty-forecast-top-7-data-backed-trends
  메모: PDRN, 엑소좀, 트라넥삼산 등이 "떠오르는 성분"으로 언급됨 — 우리가 API 테스트에서 확인한 PDRN 급상승 패턴과 실무적으로 교차검증되는 자료. 서론/실험 부분에서 실제 사례로 인용 가능.

## 2.2 시계열 예측 / 트렌드 조기탐지(Breakout Detection)

- **"New Fashion Products Performance Forecasting: A Survey on Evolutions, Models and Emerging Trends"**
  https://arxiv.org/html/2501.10324v1
  메모: 패션 트렌드/신제품 성과 예측 모델들을 정리한 서베이 논문. **방법론 챕터 인용에 유용** — 우리 연구가 이 흐름의 어디에 위치하는지 설명할 때 좋음.

- **HERMES: Hybrid Error-corrector Model with inclusion of External Signals for nonstationary fashion time series**
  https://arxiv.org/pdf/2202.03224
  메모: 외부 신호(소셜/검색 등)를 결합한 비정상 시계열(패션) 예측 모델. 우리의 "검색량 시계열 기반 예측" 설계와 방법론적으로 유사한 선례.

- **Multimodal Quasi-AutoRegression (MuQAR): Forecasting the visual popularity of new fashion products**
  https://arxiv.org/pdf/2204.04014
  메모: LSTM, ConvNet-LSTM, DA-RNN 등 비교 실험. 우리 연구의 "베이스라인 vs 로지스틱회귀 vs XGBoost 비교" 설계와 유사한 방법론적 접근.

- **Heuritech "Next" 모델 소개** (기업 자료, 학술논문 아님)
  https://heuritech.com/articles/forecasting-model-trend-forecasting/
  메모: 딥러닝+HMM 결합, 소셜 신호로 "Retro Football Sneakers" 유행을 1년 전에 예측한 사례. 산업 적용 사례로 서론에 인용 가능.

- **"구글 트렌드 빅데이터를 통한 바이오의약품의 시장 점유율 분석과 추정"**
  https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202019762877326
  메모: 검색량 빅데이터 → 실제 시장 점유율 추정. 국내 논문, 검색량-실제 시장 반응 간 관계를 다룬 방법론적으로 유사한 선행연구.

- **"구글 트렌드를 이용한 기업의 매출액 및 주가 예측: BMW와 Mercedes-Benz 키워드를 중심으로"**
  https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002398353
  메모: 검색량 기반 예측의 국내 선례. 방법론 인용 가능.

- **"네이버 데이터랩 검색어 트렌드 서비스를 이용한 온라인 포털에서의 한약재 검색 트렌드와 의미에 대한 고찰"** ⭐
  https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202128054459762&dbt=NART
  메모: **지금까지 찾은 것 중 방법론적으로 가장 직접적인 선례** — 우리가 쓰는 것과 똑같은 네이버 데이터랩 검색어트렌드 API를, 한약재라는 다른 도메인 키워드(606개)에 적용해 2020.1~2021.6 검색 트렌드를 분석함. Ⅲ장 방법론 정당화에 핵심적으로 인용할 만함.

- **Enhancing Cosmetic Supply Chain Efficiency Through Demand Forecasting Using Machine Learning** (Springer, 2024)
  https://link.springer.com/chapter/10.1007/978-3-031-75923-9_13
  메모: 화장품 실제 데이터로 여러 머신러닝 수요예측 알고리즘의 적용가능성·효과성을 비교한 논문. 화장품+ML 예측을 직접 다룬 해외 논문이라 인용 가치 높음.

## 2.3 국내 뷰티 이커머스 데이터 활용

- **"텍스트 마이닝 기법을 활용한 고객 리뷰 감성분석 : 올리브영 기초 화장품 리뷰 빅데이터 중심으로"**
  https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=DIKO0016275283
  메모: 올리브영 리뷰 빅데이터로 기초화장품 유형별 선택속성·만족/불만족 요인 분석. **올리브영 데이터를 학술적으로 다룬 국내 선례 — 우리의 올리브영 스냅샷 활용 부분 정당화에 인용 가능.**

- **"화장품 소비자는 어떤 기호가치로 소비를 하는가?: 텍스트 마이닝을 이용한 화장품 브랜드 평판과 브랜드 선택속성 분석"** (송만석, 아시안뷰티화장품학술지, 2021)
  https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002729886

- **"뉴 노멀 시대의 빅 데이터 분석을 통한 화장품 트렌드에 대한 연구"** (박혜진)
  https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002672716
  메모: '기초', '여드름', '성분', '천연' 등 키워드 빈도 분석, 코로나19 이후 성분 관심도 증가 확인. 2.1과도 겹치는 내용.

- **올리브영 데이터 규모 관련 업계 자료** (학술논문 아님, 배경 설명용)
  https://ceofeeding.oopy.io/a842e8a2-92df-4c9d-97ab-038926e9904e
  메모: 올리브영이 연 1억 건 이상 구매 데이터 기반 트렌드 리포트를 정례화한다는 내용 — 서론에서 "업계도 이미 이런 데이터를 트렌드 파악에 쓰고 있다"는 근거로 활용 가능.

**공백 확인**: 지금까지 찾은 국내 논문들은 대부분 **텍스트마이닝/감성분석 기반의 사후적(descriptive) 트렌드 분석**이고, **검색량 시계열을 활용한 예측(predictive) 모델**이나 **성분/컨셉/제형/효능처럼 카테고리를 나눠 예측 가능성을 비교**한 연구는 확인되지 않음. → 이 지점이 우리 연구의 차별점(서론 문제제기)으로 명확히 자리잡음.

---

## 다음 액션
1. 위 목록 중 2.1의 "빅데이터 분석을 통한 화장품 트렌드 변화 추이" 논문은 원문 확보 우선순위 1위 (가장 유사 주제)
2. 2.2의 arXiv 논문들은 무료로 원문 열람 가능 (패션 도메인이지만 방법론적으로 우리 연구와 가장 가까움 — "breakout/조기예측"을 다른 도메인에서 어떻게 풀었는지 참고)
3. 2.3(국내 뷰티 이커머스 데이터) 관련 선행연구는 추가 검색 필요 — 이 부분이 비어있으면 "이 지점이 선행연구의 공백이고 우리 연구의 기여점"이라는 서술로 활용 가능
4. 원문 확인 후, 우리 연구의 차별점(성분/컨셉/제형/효능 카테고리별 예측가능성 비교)이 기존 연구에 없다는 것을 명확히 하는 문장을 Ⅰ장 서론에 반영
