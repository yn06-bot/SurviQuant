"""
indicators.py — 기술적 지표 연산
==================================
Skills 참조: 01_core_rules.md § 2 (기술적 지표 연산 기준)

연산 지표:
  Overlay  : MA20, Bollinger Band(20일, 2σ)
  Sub-chart: RSI(14), ADX(14), ATR(14), ROC(10), Volatility(20일)

NaN 처리 원칙: 초기 계산 NaN은 0 대치 없이 모두 Drop (core_rules § 2)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 추세 지표
# ─────────────────────────────────────────────
def add_ma20(df: pd.DataFrame) -> pd.DataFrame:
    """MA20: 20일 단순 이동평균."""
    df["MA20"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(20).mean()
    )
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Band: 중심선 MA20 기준 ±2σ."""
    def _bb(close: pd.Series):
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        return mid + n_std * std, mid - n_std * std

    upper, lower = zip(
        *df.groupby("Ticker")["Close"].apply(
            lambda x: list(zip(*[_bb(x)]))
        ).explode().tolist()
    )
    df["BB_Upper"] = list(upper)
    df["BB_Lower"] = list(lower)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX(14): 추세 강도 지표."""
    def _adx(g: pd.DataFrame) -> pd.Series:
        high, low, close = g["High"], g["Low"], g["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        dm_pos = high.diff().clip(lower=0)
        dm_neg = (-low.diff()).clip(lower=0)
        dm_pos = dm_pos.where(dm_pos > (-low.diff()).clip(lower=0), 0)
        dm_neg = dm_neg.where(dm_neg > high.diff().clip(lower=0), 0)

        di_pos = 100 * dm_pos.ewm(alpha=1 / period, adjust=False).mean() / atr
        di_neg = 100 * dm_neg.ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg)).replace([np.inf], np.nan)
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    df["ADX"] = df.groupby("Ticker", group_keys=False).apply(_adx).reset_index(level=0, drop=True)
    return df


# ─────────────────────────────────────────────
# 모멘텀 및 변동성 지표
# ─────────────────────────────────────────────
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI(14): 상대강도지수."""
    def _rsi(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    df["RSI"] = df.groupby("Ticker")["Close"].transform(_rsi)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR(14): 평균 실제 범위."""
    def _atr(g: pd.DataFrame) -> pd.Series:
        tr = pd.concat([
            g["High"] - g["Low"],
            (g["High"] - g["Close"].shift()).abs(),
            (g["Low"] - g["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    df["ATR"] = df.groupby("Ticker", group_keys=False).apply(_atr).reset_index(level=0, drop=True)
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """ROC(10): 10일 변화율."""
    df["ROC"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.pct_change(period) * 100
    )
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volatility: 20일 로그 수익률 표준편차."""
    df["Volatility"] = df.groupby("Ticker")["Close"].transform(
        lambda x: np.log(x / x.shift()).rolling(window).std()
    )
    return df


# ─────────────────────────────────────────────
# 통합 파이프라인
# ─────────────────────────────────────────────
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame → 전체 기술지표 연산 후 NaN 행 Drop.

    NaN 처리 원칙 (core_rules § 2):
      초기 window 구간에서 발생하는 NaN은 0으로 대치하지 않고 모두 제거.
    """
    df = df.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

    df = add_ma20(df)
    df = add_bollinger(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_roc(df)
    df = add_volatility(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"  ℹ NaN Drop: {before - after:,}행 제거 → 잔여 {after:,}행")

    return df


if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ohlcv_cache.csv")
    raw = pd.read_csv(path, parse_dates=["Date"])
    result = compute_all(raw)
    print(result.tail())
    print(f"컬럼: {list(result.columns)}")
