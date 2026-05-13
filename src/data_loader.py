"""
data_loader.py — OHLCV 데이터 수집 및 캐시 저장
================================================
Skills 참조: 01_core_rules.md § 1 (데이터 스키마 및 전처리 원칙)

수집 대상:
  - S&P 500 5개 섹터 대표 약 200개 종목 (OHLCV)
  - 거시지표: 10년물 미국채 수익률 (TNX) — 날짜 기준 병합
출력:
  - data/ohlcv_cache.csv
"""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf

# ── 5개 섹터 대표 종목 (약 200개, S&P 500 구성 종목 기준)
SECTOR_TICKERS = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "AMD", "QCOM", "TXN",
        "AMAT", "LRCX", "KLAC", "MU", "MRVL", "FTNT", "PANW", "CRWD",
        "NOW", "INTU", "ADBE", "CRM", "SNPS", "CDNS", "ANSS", "ANET",
        "HPE", "STX", "WDC", "NTAP", "COHR", "KEYS",
    ],
    "Health Care": [
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "ABT", "TMO", "DHR",
        "AMGN", "ISRG", "SYK", "BSX", "MDT", "EW", "ZBH", "BAX",
        "BDX", "IQV", "A", "MTD", "WAT", "HOLX", "PODD",
        "INSP", "ALGN", "DXCM", "RMD", "IDXX", "MRNA",
    ],
    "Financials": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
        "CB", "PGR", "ALL", "TRV", "MET", "PRU", "AFL", "AIG",
        "ICE", "CME", "SPGI", "MCO", "V", "MA", "AXP", "COF",
        "DFS", "SYF", "USB", "PNC", "TFC", "FITB",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "LOW", "TJX", "ROST", "BKNG", "MAR",
        "HLT", "NKE", "LULU", "RL", "PVH", "HAS", "MAT", "POOL",
        "PHM", "DHI", "LEN", "TOL", "NVR", "MHO", "GRMN", "ORLY",
        "AZO", "AAP", "MCD", "SBUX", "YUM", "DRI",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "VLO",
        "PSX", "OXY", "HAL", "BKR", "DVN", "FANG", "APA", "MRO",
        "HES", "CTRA", "EQT", "RRC", "AR", "SM", "MTDR", "PDCE",
        "NOV", "FTI", "WHD", "NE", "PTEN", "HP",
    ],
}

START_DATE = "2010-01-01"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "ohlcv_cache.csv")
MISSING_THRESHOLD = 0.05


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    최신 yfinance가 반환하는 MultiIndex 컬럼을 단일 레벨로 평탄화.
    예: ('Close', 'AAPL') → 'Close'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_tnx(start: str, end: str) -> pd.DataFrame:
    """10년물 미국채 수익률(^TNX) 수집."""
    tnx = yf.download("^TNX", start=start, end=end, progress=False, auto_adjust=True)
    tnx = _flatten_columns(tnx)          # ← MultiIndex 평탄화
    tnx = tnx[["Close"]].rename(columns={"Close": "TNX"})
    tnx.index = pd.to_datetime(tnx.index)
    tnx.index.name = "Date"
    return tnx


def fetch_ohlcv(ticker: str, sector: str, start: str, end: str):
    """단일 종목 OHLCV 수집 및 기본 전처리."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df = _flatten_columns(df)         # ← MultiIndex 평탄화
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df["Ticker"] = ticker
        df["Sector"] = sector
        return df
    except Exception as e:
        print(f"  ⚠ {ticker} 수집 실패: {e}")
        return None


def check_missing(df: pd.DataFrame, ticker: str) -> None:
    """결측치 5% 초과 시 Alert 출력."""
    missing_pct = df.isnull().mean().max()
    if missing_pct > MISSING_THRESHOLD:
        print(f"  🚨 ALERT: {ticker} 결측치 {missing_pct:.1%} > 5% 임계값")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"📡 데이터 수집 시작 | 기간: {START_DATE} ~ {end_date}")

    # 1. TNX 수집
    print("  → 10년물 미국채(TNX) 수집 중...")
    tnx_df = fetch_tnx(START_DATE, end_date)

    # 2. 종목별 OHLCV 수집
    frames = []
    total = sum(len(v) for v in SECTOR_TICKERS.values())
    count = 0

    for sector, tickers in SECTOR_TICKERS.items():
        for ticker in tickers:
            count += 1
            print(f"  [{count}/{total}] {ticker} ({sector})")
            df = fetch_ohlcv(ticker, sector, START_DATE, end_date)
            if df is not None:
                df = df.join(tnx_df, how="left")
                df["TNX"] = df["TNX"].ffill()
                check_missing(df, ticker)
                frames.append(df.reset_index())

    if not frames:
        raise RuntimeError("수집된 데이터가 없습니다.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(CACHE_PATH, index=False)
    print(f"\n✅ OHLCV 캐시 저장 완료: {CACHE_PATH}")
    print(f"   총 {combined['Ticker'].nunique()}개 종목 / {len(combined):,}행")


if __name__ == "__main__":
    main()
