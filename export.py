"""
export.py — scores.json 정적 파일 생성
========================================
Skills 참조: 03_ops_pipeline.md § 1 (일일 데이터 갱신 파이프라인)

daily_update.yml에서 호출:
  python src/export.py

입력:  data/scores_raw.csv  (scoring.py 출력)
출력:  data/scores.json     (dashboard.py 가 읽는 정적 파일)
"""

import json
import os
from datetime import datetime

import pandas as pd

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH   = os.path.join(DATA_DIR, "scores_raw.csv")
OUT_PATH   = os.path.join(DATA_DIR, "scores.json")

PERSONA_WEIGHTS = {
    "aggressive":   {"profit": 0.69, "loss_defense": 0.31, "lambda": 0.44},
    "neutral":      {"profit": 0.50, "loss_defense": 0.50, "lambda": 1.00},
    "conservative": {"profit": 0.31, "loss_defense": 0.69, "lambda": 2.25},
}

SIGNAL_THRESHOLDS = {
    "강력 매수": 60,
    "매수 고려": 50,
    "관망":      40,
    "비중 축소":  0,
}

HORIZON_DAYS     = 20
MISSING_THRESHOLD = 0.05


def build_scores_json(df: pd.DataFrame, reference_date: str) -> dict:
    """
    scores_raw.csv → scores.json 구조 변환.

    결측치 검사 (03_ops_pipeline § 1):
      Profit_Chance 또는 Loss_Risk 결측 비율이 5% 초과 시 Alert.
    """
    missing_pct = df[["Profit_Chance", "Loss_Risk"]].isnull().mean().max()
    if missing_pct > MISSING_THRESHOLD:
        print(f"🚨 ALERT: 결측치 {missing_pct:.1%} > {MISSING_THRESHOLD:.0%} 임계값")

    records = []
    for _, row in df.iterrows():
        records.append({
            "Ticker":        row["Ticker"],
            "Sector":        row["Sector"],
            "Profit_Chance": round(float(row["Profit_Chance"]), 2),
            "Loss_Risk":     round(float(row["Loss_Risk"]),     2),
        })

    payload = {
        "meta": {
            "project":          "SurviQuant",
            "reference_date":   reference_date,
            "horizon_days":     HORIZON_DAYS,
            "event_definition": {
                "profit_event": "매수 시점 대비 +10% 도달",
                "loss_event":   "매수 시점 대비 -10% 도달",
            },
            "universe": {
                "survival_records_tickers": df["Ticker"].nunique(),
                "modeled_tickers":          len(df),
                "note": (
                    "EDA·생존레코드는 S&P500 약 200개 종목 전체 활용, "
                    "RSF 학습·추론은 5섹터 대표 50종목"
                ),
            },
            "model_performance_c_index": {
                "profit_model": 0.7087,
                "loss_model":   0.7681,
            },
            "persona_weights":    PERSONA_WEIGHTS,
            "signal_thresholds":  SIGNAL_THRESHOLDS,
            "extensibility": (
                "코어 규칙(01_core_rules.md) 유지 + 도메인 규칙(02_domain_scoring.md) 교체 방식으로 "
                "KOSPI / ETF / 섹터 ETF 등 타 자산군으로 확장 가능한 모듈형 구조"
            ),
        },
        "scores": records,
    }
    return payload


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} 없음. scoring.py를 먼저 실행하세요."
        )

    df = pd.read_csv(RAW_PATH)
    reference_date = datetime.today().strftime("%Y-%m-%d")

    payload = build_scores_json(df, reference_date)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ scores.json 생성 완료: {OUT_PATH}")
    print(f"   종목 수: {len(df)} / 기준일: {reference_date}")


if __name__ == "__main__":
    main()
