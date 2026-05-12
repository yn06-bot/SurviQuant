"""
models.py — Random Survival Forest 학습 및 롤백 정책
======================================================
Skills 참조: 03_ops_pipeline.md § 2 (주간 모델 재학습 파이프라인)

실행:
  학습:   python src/models.py
  검증:   python src/scoring.py --validate   (scoring.py 에서 담당)

롤백 정책:
  재학습 모델 C-index가 기존 모델 대비 5% 이상 하락 시
  model_profit.pkl / model_loss.pkl 덮어쓰기 취소, 이전 버전 유지
"""

import os
import pickle

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# 특징 컬럼 (core_rules § 2 기반)
FEATURE_COLS = [
    "MA20", "BB_Upper", "BB_Lower",
    "RSI", "ATR", "ADX",
    "Volatility", "ROC", "TNX",
]

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH  = os.path.join(MODEL_DIR, "ohlcv_cache.csv")
ROLLBACK_THRESHOLD = 0.05   # C-index 5% 하락 시 롤백


def load_labeled_data() -> pd.DataFrame:
    """라벨링된 데이터 로드 (T_profit, E_profit, T_loss, E_loss 필요)."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    required = FEATURE_COLS + ["T_profit", "E_profit", "T_loss", "E_loss", "Ticker", "Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}\n→ survival_events.py 먼저 실행하세요.")
    return df.dropna(subset=FEATURE_COLS)


def build_structured_array(event: pd.Series, time: pd.Series):
    """sksurv 전용 구조화 배열 생성."""
    return np.array(
        [(bool(e), t) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )


def train_rsf(X: pd.DataFrame, y) -> RandomSurvivalForest:
    """RSF 학습 (n_estimators=100, 재현성 seed=42)."""
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rsf.fit(X, y)
    return rsf


def evaluate_cindex(model: RandomSurvivalForest, X: pd.DataFrame, y) -> float:
    """C-index 계산."""
    risk_scores = model.predict(X)
    result = concordance_index_censored(
        y["event"], y["time"], risk_scores
    )
    return result[0]


def save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f" 모델 저장: {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    print(" 데이터 로드 중...")
    df = load_labeled_data()

    # 시계열 분할: 최근 20% → 검증
    df_sorted = df.sort_values("Date")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    val_df   = df_sorted.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS]
    X_val   = val_df[FEATURE_COLS]

    # ── 수익 모델
    print("\n 수익 모델 학습 중...")
    y_profit_train = build_structured_array(train_df["E_profit"], train_df["T_profit"])
    y_profit_val   = build_structured_array(val_df["E_profit"],   val_df["T_profit"])
    model_profit   = train_rsf(X_train, y_profit_train)
    cindex_profit  = evaluate_cindex(model_profit, X_val, y_profit_val)
    print(f"  검증 C-index (수익): {cindex_profit:.4f}")

    # ── 손실 모델
    print("\n 손실 모델 학습 중...")
    y_loss_train = build_structured_array(train_df["E_loss"], train_df["T_loss"])
    y_loss_val   = build_structured_array(val_df["E_loss"],   val_df["T_loss"])
    model_loss   = train_rsf(X_train, y_loss_train)
    cindex_loss  = evaluate_cindex(model_loss, X_val, y_loss_val)
    print(f"  검증 C-index (손실): {cindex_loss:.4f}")

    # ── 롤백 정책 (03_ops_pipeline § 2)
    profit_path = os.path.join(MODEL_DIR, "model_profit.pkl")
    loss_path   = os.path.join(MODEL_DIR, "model_loss.pkl")

    def should_save(new_cindex: float, model_path: str, label: str) -> bool:
        if not os.path.exists(model_path):
            return True
        old_model = load_model(model_path)
        old_cindex = evaluate_cindex(
            old_model, X_val,
            y_profit_val if label == "profit" else y_loss_val
        )
        drop = (old_cindex - new_cindex) / old_cindex
        if drop > ROLLBACK_THRESHOLD:
            print(f" 롤백: {label} C-index {old_cindex:.4f}→{new_cindex:.4f} "
                  f"({drop:.1%} 하락, 임계 {ROLLBACK_THRESHOLD:.0%})")
            return False
        print(f" 모델 갱신: {label} C-index {old_cindex:.4f}→{new_cindex:.4f}")
        return True

    if should_save(cindex_profit, profit_path, "profit"):
        save_model(model_profit, profit_path)
    if should_save(cindex_loss, loss_path, "loss"):
        save_model(model_loss, loss_path)

    print("\n 학습 파이프라인 완료")


if __name__ == "__main__":
    main()
