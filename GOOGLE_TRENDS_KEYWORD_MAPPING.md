# 성분 키워드 → Google Trends 검색어 매핑 (검토용)

RQ3(해외 선행 신호) 검증을 위해 `naver_keywords_final.csv`의 성분 22개를
Google Trends(미국, US geo) 검색어로 매핑한 초안. 동음이의어/모호성이 있는
단어는 "skincare" 수식어를 붙였다. 실행 전에 이 매핑이 맞는지 검토 부탁.

| 한글 원어 | Google Trends 검색어 | 비고 |
|---|---|---|
| 나이아신아마이드 | `niacinamide` | 특이적, 수식어 불필요 |
| PDRN | `PDRN` | 특이적 (단, 검색량 자체가 매우 적을 수 있음) |
| 판테놀 | `panthenol` | 특이적 |
| 펩타이드 | `peptide skincare` | "peptide"만 쓰면 생화학·운동보조제 검색과 섞임 |
| 히알루론산 | `hyaluronic acid` | 특이적 (화장품 맥락이 이미 지배적) |
| 세라마이드 | `ceramide` | 특이적 |
| 레티놀 | `retinol` | 특이적 |
| 아하 성분 | `AHA skincare` | "AHA"만 쓰면 미국마취과학회(American Heart Association) 등과 충돌 |
| 살리실산 | `salicylic acid` | 특이적 (화장품 맥락이 지배적이나 의약품 맥락도 일부 섞일 수 있음) |
| 콜라겐 | `collagen skincare` | "collagen"만 쓰면 건강보조제·관절 건강 검색과 섞임 |
| 스쿠알란 | `squalane` | 특이적 (squalene와 철자 다름에 주의) |
| 병풀추출물 | `centella asiatica` | "cica"도 고려 가능하나 K-beauty 용어라 서구권엔 덜 알려짐 |
| 엑소좀 | `exosome skincare` | "exosome"만 쓰면 생물학·의학 연구 검색과 섞임 |
| 장미추출물 | `rose extract skincare` | "rose extract"만 쓰면 요리·향수 등과 섞일 수 있음 |
| 어성초 | `houttuynia cordata` | 매우 니치한 용어라 미국 검색량이 거의 0일 가능성 높음 |
| 알란토인 | `allantoin` | 특이적 |
| 아데노신 | `adenosine skincare` | "adenosine"만 쓰면 심장의학(아데노신 주사) 검색과 크게 섞임 |
| 베타글루칸 | `beta glucan skincare` | "beta glucan"만 쓰면 건강보조제·면역 관련 검색과 섞임 |
| 트레할로스 | `trehalose` | 특이적 |
| 감초추출물 | `licorice extract skincare` | "licorice"만 쓰면 식품(사탕) 검색과 크게 섞임 |
| 비타민E | `vitamin e skincare` | "vitamin E"만 쓰면 건강보조제 검색이 압도적으로 많음 |
| 시어버터 | `shea butter` | 요리용과 일부 섞일 수 있으나 화장품 맥락이 이미 우세 |

## 검토 필요 사항
1. 위 매핑에 동의하는지(특히 "skincare" 수식어 붙인 8개 — 펩타이드/아하성분/콜라겐/엑소좀/장미추출물/아데노신/베타글루칸/감초추출물/비타민E, 9개).
2. "어성초(houttuynia cordata)"는 미국에서 검색량이 0에 가까워 분석에서 제외해야 할 수도 있음 — 수집 후 확인 필요.
3. geo를 미국(US)으로 한정할지, 전세계(Worldwide)로 할지 — 우선 US로 진행 제안(가장 명확한 단일 선행 시장, 언어적으로도 깨끗함).
