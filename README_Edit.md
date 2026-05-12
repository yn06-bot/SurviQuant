# SurviQuant — 생존분석 기반 S&P 500 AI 투자 대시보드

배포 URL: https://surviquant.streamlit.app

일반적인 주가 예측 모델은 방향성(상승/하락)만 다룹니다. SurviQuant는 여기에 **'언제 도달하는가'** 라는 시간 차원을 추가했습니다. 매수 시점 대비 +10% 수익 또는 −10% 손실에 도달하기까지의 확률과 소요 시간을 생존분석으로 정량화한 투자 분석 시스템입니다.

---

## 왜 생존분석인가

생존분석은 의학 임상 연구에서 '치료 후 얼마나 생존하는가'를 추정하는 기법입니다. 여기에 착안해 주식 투자에 적용했습니다. '매수 후 목표 수익에 얼마나 빨리 도달하는가'와 '손실이 발생하기까지 얼마나 버티는가'를 같은 프레임으로 모델링할 수 있기 때문입니다.

Random Survival Forest로 두 모델을 각각 학습했고, 실측 C-index는 수익 모델 **0.7087**, 손실 모델 **0.7681**입니다. C-index 0.7 이상은 임상·금융 분야 모두에서 실용적 수준으로 인정받는 기준입니다.

모델 개발 과정에서 가장 인상적이었던 결과는 Volatility(20일 로그수익률 표준편차)였습니다. 단순 변동성 지표가 수익 도달 예측(HR 2.87)보다 손실 발생 예측(HR 5.28)에서 훨씬 강하게 작동했습니다. 이 결과가 안정형 성향의 방어 가중치를 높이게 된 핵심 근거입니다.

실측 Top Pick (중립형 기준): **ANET → AMAT → COHR**

---

## AI Score 산출 공식

투자자 성향별로 가중치를 달리해 동일 종목도 다른 점수가 나옵니다. 손실 회피 계수 λ는 Tversky & Kahneman(1992) Prospect Theory에서 실증된 값을 직접 적용했습니다.

```
AI Score = Profit_Chance × w_profit + (100 − Loss_Risk) × w_defense

공격형: w_profit=0.69, w_defense=0.31  (λ=0.44)
중립형: w_profit=0.50, w_defense=0.50  (λ=1.00)  ← 기본값
안정형: w_profit=0.31, w_defense=0.69  (λ=2.25)
```

시그널은 중립형 기준으로 60점 이상(강력 매수) / 50~59점(매수 고려) / 40~49점(관망) / 40점 미만(비중 축소) 4단계로 분류됩니다.

---

## 아키텍처 — 연산과 렌더링의 분리

서버 비용과 로딩 지연을 줄이기 위해 백엔드 추론과 프론트엔드 렌더링을 완전히 분리했습니다.

GitHub Actions가 매일 장 마감 후 yfinance로 최신 OHLCV를 수집하고 사전 학습된 model.pkl로 AI Score를 추론해 scores.json을 자동 커밋합니다. Streamlit 대시보드는 이 JSON 파일만 읽어 렌더링하므로 별도의 모델 연산 없이 빠르게 동작합니다.

---

## 파일 구조

```
SurviQuant/
├── README.md
├── dashboard.py             # Streamlit 프론트엔드 — scores.json만 읽음
├── requirements.txt         # 배포용 (Streamlit Cloud): pandas, plotly, streamlit, pyarrow
├── requirements-dev.txt     # 개발용 (백엔드 파이프라인): scikit-survival, yfinance 등
│
├── .github/workflows/
│   ├── daily_update.yml     # 평일 장 마감 후: OHLCV 수집 → AI Score 추론 → scores.json 커밋
│   └── weekly_retrain.yml   # 매주 일요일: RSF 재학습 + C-index 검증 + 롤백
│
├── skills/                  # 분석 규칙 명세서
│   ├── 00_index.md          # 전체 라우팅
│   ├── 01_core_rules.md     # 데이터 스키마 · 지표 연산 · 시각화 규칙
│   ├── 02_domain_scoring.md # 생존분석 Event · AI Score 공식 · 성향별 가중치
│   └── 03_ops_pipeline.md   # 일일 갱신 · 주간 재학습 · 롤백 정책
│
├── src/                     # 백엔드 추론 파이프라인
│   ├── data_loader.py       # yfinance로 OHLCV + TNX(미국채) 수집
│   ├── indicators.py        # MA20 / RSI / ADX / BB / ATR / ROC / Volatility 연산
│   ├── survival_events.py   # +10% / -10% 도달 시간 라벨링
│   ├── models.py            # RSF 학습 + 롤백 정책
│   ├── scoring.py           # 모델 추론 → Profit_Chance, Loss_Risk
│   └── export.py            # scores_raw.csv → scores.json 변환
│
└── data/
    ├── ohlcv_cache.csv      # OHLCV 차트 데이터 (2024.04 ~ 현재, 50종목)
    ├── model_profit.pkl     # 수익 도달 예측 모델 (C-index 0.7087)
    ├── model_loss.pkl       # 손실 발생 예측 모델 (C-index 0.7681)
    └── scores.json          # dashboard.py가 읽는 최종 정적 파일
```

모델 학습에는 S&P 500 전종목 기준 2010.01 ~ 2026.04 데이터를 사용했습니다. ohlcv_cache는 대시보드 차트 표시용으로 최근 2년치(2024.04~)를 유지합니다.

---

## Skills 문서 구조

분석 규칙을 범용(Core)과 도메인(Domain)으로 분리했습니다. 코어 규칙은 유지한 채 도메인 규칙만 교체하면 KOSPI, ETF 등 다른 자산군으로 확장할 수 있습니다. 각 Python 파일의 docstring에 참조 명세서가 명시되어 있어 코드와 문서가 연결됩니다.

---

## 실행 방법

대시보드만 실행:
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

백엔드 파이프라인 전체 재현:
```bash
pip install -r requirements-dev.txt
python src/data_loader.py
python src/survival_events.py
python src/models.py
python src/scoring.py
python src/export.py
streamlit run dashboard.py
```

---

본 도구는 학술·시연 목적으로 제작되었습니다. 투자 권유 또는 자문이 아닙니다.
