# Domain Scoring Rules: S&P 500 Survival Analysis

이 문서는 S&P 500 종목 대상 '수익 및 손실 도달 시간'을 예측하는 생존분석 모델링 규칙과, 이를 사용자 친화적인 AI Score로 변환하는 도메인 로직입니다.

## 1. 생존분석 사건(Event) 및 실증적 증명

1. 수익 사건 (Profit Event): 매수 시점 대비 +10% 수익률 도달.

2. 손실 사건 (Risk Event): 매수 시점 대비 -10% 손실률 도달. (예측 Horizon: 20영업일 고정)

[모델링 검증 결과]
Kaplan-Meier 탐색 및 Cox PH 검증 결과, 기술적 지표 중 'Volatility(변동성)'가 수익 도달(HR 2.87)과 손실 발생(Risk HR 5.28) 모두에서 가장 강력한 예측력을 가짐을 확인했습니다.

실측 모델 성능: 수익 모델 C-index 0.7087 / 손실 모델 C-index 0.7681

## 2. 통합 AI Score 산출 로직

Random Survival Forest(RSF) 사전 학습 모델(`model_profit.pkl`, `model_loss.pkl`)을 통해 확률을 추론합니다.

1. AI Score 공식 (성향별 가중치 적용):

   AI Score = Profit_Chance × w_profit + (100 − Loss_Risk) × w_defense

2. 산출 근거: Tversky·Kahneman(1992) Prospect Theory의 손실 회피 계수 λ를 가중치에 직접 반영하여 투자자 심리를 정량화합니다. 성향별로 λ가 달라지므로 동일한 종목도 투자자마다 다른 Score를 산출합니다.

## 3. 투자 성향별 가중치 차등 및 시그널 임계값

1. 공격형 성향 (λ=0.44): w_profit=0.69 / w_defense=0.31 (고수익 추구)

2. 중립형 성향 (λ=1.00): w_profit=0.50 / w_defense=0.50 (기본값 — 손익 동등 고려)

3. 안정형 성향 (λ=2.25): w_profit=0.31 / w_defense=0.69 (손실 최소화 — Kahneman 손실 회피 계수 직접 적용)

[중립형 기준 4단계 시그널]
* 강력 매수 (우선 검토): Score 60점 이상 (예: 모델 실측 결과 ANET, AMAT, COHR 등이 Top Pick으로 배정)
* 매수 고려: Score 50점 ~ 59점
* 관망: Score 40점 ~ 49점
* 비중 축소: Score 40점 미만
