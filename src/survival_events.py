"""
survival_events.py — 생존분석 사건(Event) 정의 및 라벨링
============================================================
Skills 참조: 02_domain_scoring.md § 1 (생존분석 사건 정의)

사건 정의:
  - 수익 사건 (Profit Event): 매수 시점 대비 +10% 도달
  - 손실 사건 (Risk Event) : 매수 시점 대비 -10% 도달
  - 예측 Horizon: 20영업일 고정

출력 컬럼:
  T_profit  : 수익 도달 소요 영업일 (미도달 시 horizon)
  E_profit  : 수익 도달 여부 (1=도달, 0=중도 절단)
  T_loss    : 손실 도달 소요 영업일
  E_loss    : 손실 도달 여부
"""

import numpy as np
import pandas as pd

PROFIT_THR = 0.10   # +10%
LOSS_THR   = -0.10  # -10%
HORIZON    = 20     # 영업일


def label_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    종목별로 각 시점에서 향후 HORIZON일 이내
    수익/손실 사건 발생 여부와 소요 일수를 라벨링.

    Parameters
    ----------
    df : OHLCV + 기술지표 DataFrame (Date, Ticker, Close 필수)

    Returns
    -------
    df : T_profit, E_profit, T_loss, E_loss 컬럼 추가
    """
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    t_profit_list, e_profit_list = [], []
    t_loss_list,   e_loss_list   = [], []

    for ticker, grp in df.groupby("Ticker"):
        closes = grp["Close"].values
        n = len(closes)

        t_profit = np.full(n, HORIZON, dtype=float)
        e_profit = np.zeros(n, dtype=int)
        t_loss   = np.full(n, HORIZON, dtype=float)
        e_loss   = np.zeros(n, dtype=int)

        for i in range(n):
            buy_price = closes[i]
            end_idx   = min(i + HORIZON, n)

            for j in range(i + 1, end_idx):
                ret = (closes[j] - buy_price) / buy_price
                if e_profit[i] == 0 and ret >= PROFIT_THR:
                    t_profit[i] = j - i
                    e_profit[i] = 1
                if e_loss[i] == 0 and ret <= LOSS_THR:
                    t_loss[i] = j - i
                    e_loss[i] = 1
                if e_profit[i] == 1 and e_loss[i] == 1:
                    break

        t_profit_list.extend(t_profit)
        e_profit_list.extend(e_profit)
        t_loss_list.extend(t_loss)
        e_loss_list.extend(e_loss)

    df["T_profit"] = t_profit_list
    df["E_profit"] = e_profit_list
    df["T_loss"]   = t_loss_list
    df["E_loss"]   = e_loss_list

    print(f"  ✅ 라벨링 완료: {len(df):,}행")
    print(f"     수익 사건 발생률: {df['E_profit'].mean():.1%}")
    print(f"     손실 사건 발생률: {df['E_loss'].mean():.1%}")

    return df


if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ohlcv_cache.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = label_events(df)
    print(df[["Ticker", "Date", "Close", "T_profit", "E_profit", "T_loss", "E_loss"]].head(10))
