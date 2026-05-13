"""
fetch_company_info.py
=====================
scores.json에 등록된 전체 티커의 기업 정보를 yfinance에서 수집하여
data/company_info.json 으로 저장합니다.

실행 시점: GitHub Actions daily_update.yml 또는 최초 1회 로컬 실행
출력 파일: data/company_info.json
"""

import json
import time
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_company_info(ticker: str) -> dict:
    """yfinance에서 기업 기본 정보를 조회합니다."""
    try:
        info = yf.Ticker(ticker).info
        raw_summary = info.get("longBusinessSummary", "")
        summary = raw_summary[:400] + "..." if len(raw_summary) > 400 else raw_summary
        return {
            "name":       info.get("longName", ticker),
            "summary":    summary if summary else "기업 개요 정보를 불러올 수 없습니다.",
            "website":    info.get("website", ""),
            "market_cap": info.get("marketCap", None),
            "employees":  info.get("fullTimeEmployees", None),
            "industry":   info.get("industry", ""),
        }
    except Exception as e:
        print(f"  [WARN] {ticker} 조회 실패: {e}")
        return {
            "name": ticker,
            "summary": "기업 정보를 불러올 수 없습니다.",
            "website": "",
            "market_cap": None,
            "employees": None,
            "industry": "",
        }


def main():
    # scores.json에서 티커 목록 로드
    scores_path = DATA_DIR / "scores.json"
    if not scores_path.exists():
        print(f"[ERROR] {scores_path} 파일을 찾을 수 없습니다. data_loader.py 먼저 실행하세요.")
        return

    with open(scores_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tickers = [row["Ticker"] for row in payload["scores"]]
    print(f"[INFO] 총 {len(tickers)}개 티커 수집 시작")

    # 기존 company_info.json이 있으면 로드 (이미 수집된 항목 스킵)
    output_path = DATA_DIR / "company_info.json"
    existing = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[INFO] 기존 캐시 {len(existing)}개 항목 로드")

    result = dict(existing)

    for i, ticker in enumerate(tickers):
        if ticker in result:
            print(f"  [{i+1}/{len(tickers)}] {ticker} — 캐시 사용 (스킵)")
            continue

        print(f"  [{i+1}/{len(tickers)}] {ticker} — API 조회 중...")
        result[ticker] = fetch_company_info(ticker)
        time.sleep(0.5)  # yfinance rate limit 방지

    # 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] company_info.json 저장 완료 → {output_path}")
    print(f"       총 {len(result)}개 티커 수록")


if __name__ == "__main__":
    main()
