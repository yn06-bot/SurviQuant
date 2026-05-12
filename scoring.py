"""
scoring.py — AI Score 산출 및 성능 검증
=========================================
Skills 참조: 02_domain_scoring.md § 2, § 3

실행:
  추론:  python src/scoring.py
  검증:  python src/scoring.py --validate

AI Score 공식:
  score = Profit_Chance × w_profit + (100 - Loss_Risk) × w_defense

성향별 가중치 (λ = Tversky·Kahneman 손실 회피 계수):
  공격형: w_profit=0.69 / w_defense=0.31  (λ=0.44)
  중립형: w_profit=0.50 / w_defense=0.50  (λ=1.00)  ← 기본값
  안정형: w_profit=0.31 / w_defense=0.69  (λ=2.25)
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "MA20", "BB_Upper", "BB_Lower",
    "RSI", "ATR", "ADX",
    "Volatility", "ROC", "TNX",
]

PERSONA_WEIGHTS = {
    "aggressive":   {"w_profit": 0.69, "w_defense": 0.31, "lambda": 0.44},
    "neutral":      {"w_profit": 0.50, "w_defense": 0.50, "lambda": 1.00},
    "conservative": {"w_profit": 0.31, "w_defense": 0.69, "lambda": 2.25},
}

SIGNAL_THRESHOLDS = {
    "강력 매수": 60,
    "매수 고려": 50,
    "관망":      40,
    "비중 축소":  0,
}

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
PROFIT_MODEL = os.path.join(DATA_DIR, "model_profit.pkl")
LOSS_MODEL   = os.path.join(DATA_DIR, "model_loss.pkl")
OHLCV_PATH   = os.path.join(DATA_DIR, "ohlcv_cache.csv")
ROLLBACK_THR = 0.05


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def survival_prob_at_horizon(model, X: pd.DataFrame, horizon: int = 20) -> np.ndarray:
    """
    RSF 생존 함수에서 horizon 시점의 생존 확률 추출.
    → Profit 모델: horizon 시점까지 수익 미도달 확률 (1 - 이걸 Profit_Chance로 변환)
    → Loss 모델:   horizon 시점까지 손실 미발생 확률
    """
    surv_funcs = model.predict_survival_function(X, return_array=False)
    probs = []
    for fn in surv_funcs:
        times = fn.x
        # horizon 이상인 첫 시점의 생존 확률
        mask = times >= horizon
        if mask.any():
            prob = fn.y[mask][0]
        else:
            prob = fn.y[-1]
        probs.append(float(prob))
    return np.array(probs)


def compute_ai_score(profit_chance: float, loss_risk: float, persona: str = "neutral") -> float:
    w = PERSONA_WEIGHTS[persona]
    return profit_chance * w["w_profit"] + (100 - loss_risk) * w["w_defense"]


def classify_signal(score: float) -> str:
    for label, threshold in SIGNAL_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "비중 축소"


def run_inference():
    """최신 OHLCV 데이터로 전 종목 AI Score 추론 → scores_raw.csv 저장."""
    print("🔮 추론 시작...")
    df = pd.read_csv(OHLCV_PATH, parse_dates=["Date"])
    df = df.dropna(subset=FEATURE_COLS)

    # 종목별 최신 행만 추출
    latest = df.sort_values("Date").groupby("Ticker").last().reset_index()
    X = latest[FEATURE_COLS]

    model_p = load_model(PROFIT_MODEL)
    model_l = load_model(LOSS_MODEL)

    # 생존 확률 추출
    surv_profit = survival_prob_at_horizon(model_p, X)   # 수익 미도달 생존
    surv_loss   = survival_prob_at_horizon(model_l, X)   # 손실 미발생 생존

    # Profit_Chance = 수익 도달 확률 = 1 - 생존(수익 미도달)
    latest["Profit_Chance"] = ((1 - surv_profit) * 100).round(2)
    latest["Loss_Risk"]     = ((1 - surv_loss)   * 100).round(2)

    out_path = os.path.join(DATA_DIR, "scores_raw.csv")
    latest[["Ticker", "Sector", "Profit_Chance", "Loss_Risk"]].to_csv(out_path, index=False)
    print(f"  ✅ scores_raw.csv 저장: {len(latest)}종목")
    return latest


def validate():
    """
    재학습 모델 성능 검증 — 롤백 기준 (03_ops_pipeline § 2).
    C-index가 5% 이상 하락하면 exit code 1 반환.
    """
    from sksurv.metrics import concordance_index_censored

    df = pd.read_csv(OHLCV_PATH, parse_dates=["Date"])
    df = df.dropna(subset=FEATURE_COLS + ["T_profit", "E_profit", "T_loss", "E_loss"])

    df_sorted = df.sort_values("Date")
    split_idx = int(len(df_sorted) * 0.8)
    val_df    = df_sorted.iloc[split_idx:]
    X_val     = val_df[FEATURE_COLS]

    model_p = load_model(PROFIT_MODEL)
    model_l = load_model(LOSS_MODEL)

    risk_p    = model_p.predict(X_val)
    risk_l    = model_l.predict(X_val)
    cindex_p  = concordance_index_censored(val_df["E_profit"].astype(bool), val_df["T_profit"], risk_p)[0]
    cindex_l  = concordance_index_censored(val_df["E_loss"].astype(bool),   val_df["T_loss"],   risk_l)[0]

    print(f"  📊 검증 C-index — Profit: {cindex_p:.4f} / Loss: {cindex_l:.4f}")

    # 기준 성능 (실측값, 03_ops_pipeline 기준)
    BASE_PROFIT = 0.7087
    BASE_LOSS   = 0.7681

    failed = False
    for name, new, base in [("Profit", cindex_p, BASE_PROFIT), ("Loss", cindex_l, BASE_LOSS)]:
        drop = (base - new) / base
        if drop > ROLLBACK_THR:
            print(f"  🚨 롤백 권고: {name} C-index {base:.4f}→{new:.4f} ({drop:.1%} 하락)")
            failed = True

    if failed:
        raise SystemExit(1)
    print("  ✅ 성능 기준 통과")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="모델 성능 검증만 실행")
    args = parser.parse_args()

    if args.validate:
        validate()
    else:
        run_inference()
