# 생존분석 기반 AI 투자 대시보드 시스템

본 프로젝트는 단순한 주가 방향성 예측을 넘어, 매수 후 특정 수익/손실 조건에 도달하기까지의 '시간'을 예측하는 생존분석 기반의 주식 분석 시스템입니다. 

## 1. 핵심 아키텍처 (연산과 렌더링의 분리)

과도한 서버 비용과 로딩 지연을 방지하기 위해, 데이터 수집/추론 파이프라인과 대시보드 렌더링 파이프라인을 완벽히 분리했습니다.

1. Daily Inference (백엔드 로직): 매일 장 마감 후 GitHub Actions가 `yfinance`를 통해 최신 S&P 500 OHLCV 데이터를 수집하고, 사전 학습된 생존분석 모델(`model.pkl`)을 통해 종목별 최신 AI Score를 산출하여 `scores.json`을 생성합니다.

2. Dashboard Rendering (프론트엔드): Streamlit 대시보드는 무거운 모델 연산 없이, 매일 갱신되는 `scores.json` 정적 파일만 가볍게 읽어와 렌더링하므로 매우 빠르고 안정적입니다.

## 2. Skills 문서 활용 가이드 (skills/ 폴더)

시스템의 유연한 확장과 유지보수를 위해 4개의 명세서로 분리하여 관리합니다.

1. `00_index.md`: 전체 분석 규칙 및 파일 라우팅 구조 안내

2. `01_core_rules.md`: 범용 데이터 스키마 및 기술적 지표 시각화 렌더링 규칙

3. `02_domain_scoring.md`: S&P 500 특화 생존분석 사건 정의 및 AI Score 산출 로직

4. `03_ops_pipeline.md`: 시스템 유지보수 및 데이터 자동 갱신 운영 명세

## 3. 실행 방법 (추후 배포 시)

1. 패키지 설치: `pip install -r requirements.txt`

2. 데이터 수집 및 추론: `python src/data_loader.py`

3. 정적 스코어 파일 생성: `python src/export.py`

4. 대시보드 실행: `streamlit run dashboard.py`