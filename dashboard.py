"""
SurviQuant — 생존분석 기반 S&P500 AI 투자 대시보드
==========================================================
실행: streamlit run dashboard.py
데이터: ./data/scores.json, ./data/ohlcv_cache.csv, ./data/company_info.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ============================================================
# 0. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="SurviQuant | 생존분석 기반 AI 투자 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 1. 데이터 로딩
# ============================================================
@st.cache_data(show_spinner="📡 AI Score 데이터 로딩 중...")
def load_scores():
    with open(DATA_DIR / "scores.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["meta"], pd.DataFrame(payload["scores"])

@st.cache_data(show_spinner="📊 OHLCV·기술지표 로딩 중...")
def load_ohlcv():
    df = pd.read_csv(DATA_DIR / "ohlcv_cache.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data(show_spinner="🏢 기업 정보 로딩 중...")
def load_company_info_all() -> dict:
    """data/company_info.json 정적 파일에서 전체 기업 정보를 로드합니다."""
    path = DATA_DIR / "company_info.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_company_info(ticker: str, company_db: dict) -> dict:
    """company_info.json에서 단일 티커 정보를 반환합니다. 없으면 기본값 반환."""
    if ticker in company_db:
        return company_db[ticker]
    return {
        "name": ticker,
        "summary": "기업 정보를 불러올 수 없습니다.",
        "website": "",
        "market_cap": None,
        "employees": None,
        "industry": "",
    }

# ============================================================
# 2. 도메인 로직 (02_domain_scoring.md 최종 버전)
# ============================================================
PERSONA_PRESETS = {
    "conservative": {"label": "🛡 안정형", "w_profit": 0.31, "w_defense": 0.69, "lambda": 2.25, "desc": "손실 회피 우선 — Kahneman λ=2.25 그대로 적용"},
    "neutral":      {"label": "⚖ 중립형", "w_profit": 0.50, "w_defense": 0.50, "lambda": 1.00, "desc": "수익·방어 균형 (대시보드 기본값)"},
    "aggressive":   {"label": "🚀 공격형", "w_profit": 0.69, "w_defense": 0.31, "lambda": 0.44, "desc": "고수익 추구 — λ 역수 적용"},
}

SIGNAL_RULES = [
    ("강력 매수", 60, "#10B981"),
    ("매수 고려", 50, "#84CC16"),
    ("관망",      40, "#F59E0B"),
    ("비중 축소",  0, "#EF4444"),
]

PERIOD_OPTIONS = {"3개월 (63일)": 63, "6개월 (126일)": 126, "1년 (252일)": 252, "전체": 9999}

INDICATOR_HELP = {
    "MA20":       "📘 MA20 (20일 이동평균): 최근 20일 종가 평균선. 주가가 MA20 위에 있으면 단기 상승 추세.",
    "BB":         "📘 볼린저 밴드: 가격 변동 범위. 상단 돌파 → 과매수 가능성, 하단 이탈 → 과매도 신호.",
    "RSI":        "📘 RSI (14일): 0~100 범위. 70 이상 과매수(조정 가능성), 30 이하 과매도(반등 가능성).",
    "ADX":        "📘 ADX (14일): 추세 강도. 25 이상이면 의미 있는 추세 진행 중, 25 미만이면 횡보.",
    "Volatility": "📘 Volatility (20일): 일간 로그 수익률의 표준편차. 높을수록 변동성 커 리스크 증가.",
}

def compute_ai_score(profit_chance, loss_risk, persona):
    w = PERSONA_PRESETS[persona]
    return profit_chance * w["w_profit"] + (100 - loss_risk) * w["w_defense"]

def classify_signal(score):
    for label, threshold, color in SIGNAL_RULES:
        if score >= threshold:
            return label, color
    return SIGNAL_RULES[-1][0], SIGNAL_RULES[-1][2]

# ============================================================
# 3. 인사이트 자동 생성 함수
# ============================================================
def generate_insights(rsi: float, adx: float, volatility: float,
                      close: float, ma20: float) -> list[tuple[str, str, str]]:
    insights = []

    if rsi >= 70:
        insights.append(("⚠️", "RSI 과매수",
                         f"RSI {rsi:.0f} — 단기 조정 가능성, 신규 진입 시 주의가 필요합니다."))
    elif rsi <= 30:
        insights.append(("✅", "RSI 과매도",
                         f"RSI {rsi:.0f} — 반등 가능성 구간, 분할 매수를 고려해볼 수 있습니다."))
    else:
        insights.append(("🔵", "RSI 중립",
                         f"RSI {rsi:.0f} — 과매수·과매도 아님, 추세 지속 가능한 구간입니다."))

    if adx >= 40:
        insights.append(("🔥", "강한 추세",
                         f"ADX {adx:.0f} — 매우 강한 추세가 진행 중입니다."))
    elif adx >= 25:
        insights.append(("📈", "추세 존재",
                         f"ADX {adx:.0f} — 방향성 추세가 확인됩니다."))
    else:
        insights.append(("😴", "횡보 구간",
                         f"ADX {adx:.0f} — 뚜렷한 추세 없음, 관망을 고려하세요."))

    if volatility >= 0.4:
        insights.append(("🌊", "고변동성",
                         f"Volatility {volatility:.3f} — 가격 변동 폭이 매우 큽니다. 손절 라인을 설정하세요."))
    elif volatility >= 0.2:
        insights.append(("〰️", "중간 변동성",
                         f"Volatility {volatility:.3f} — 보통 수준의 변동성입니다."))
    else:
        insights.append(("🧘", "저변동성",
                         f"Volatility {volatility:.3f} — 안정적인 가격 흐름입니다."))

    gap_pct = (close - ma20) / ma20 * 100
    if gap_pct > 5:
        insights.append(("📗", "MA20 상회",
                         f"종가가 MA20보다 {gap_pct:.1f}% 위 — 단기 상승 추세 유지 중입니다."))
    elif gap_pct < -5:
        insights.append(("📕", "MA20 하회",
                         f"종가가 MA20보다 {abs(gap_pct):.1f}% 아래 — 단기 하락 압력이 있습니다."))
    else:
        insights.append(("📒", "MA20 근접",
                         f"종가가 MA20과 {gap_pct:+.1f}% 차이 — 지지·저항 테스트 구간입니다."))

    return insights

# ============================================================
# 4-A. 최신 가격 fetch (ohlcv_cache 이후 ~ 오늘)
# ============================================================
@st.cache_data(ttl=3600, show_spinner="📡 최신 주가 데이터 가져오는 중...")
def fetch_recent_prices(ticker: str, last_cache_date: str) -> pd.DataFrame:
    """
    ohlcv_cache.csv의 마지막 날짜 이후부터 오늘까지의 실제 주가를 yfinance로 fetch.
    yfinance 미설치 환경에서는 빈 DataFrame 반환 (몬테카를로는 캐시 기반으로 동작).
    """
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        start = pd.to_datetime(last_cache_date) + pd.Timedelta(days=1)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Date": "Date", "Open": "Open", "High": "High",
                                  "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df["Ticker"] = ticker
        return df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]
    except Exception:
        return pd.DataFrame()

# ============================================================
# 4-B. 몬테카를로 시뮬레이션 (향후 20영업일)
# ============================================================
def run_monte_carlo(last_close: float, volatility: float,
                    n_days: int = 20, n_sim: int = 500, seed: int = 42) -> dict:
    """
    GBM(기하 브라운 운동) 기반 몬테카를로 시뮬레이션.
    volatility: 20일 log-return std (ohlcv_cache Volatility 컬럼값)
    Returns: dict of percentile arrays (shape: n_days+1)
    """
    rng = np.random.default_rng(seed)
    dt = 1
    # 일간 변동성으로 환산 (20일 vol → 1일 vol)
    daily_vol = volatility / np.sqrt(20)
    # drift = 0 (보수적 중립 가정)
    paths = np.zeros((n_sim, n_days + 1))
    paths[:, 0] = last_close
    z = rng.standard_normal((n_sim, n_days))
    for t in range(1, n_days + 1):
        paths[:, t] = paths[:, t-1] * np.exp(-0.5 * daily_vol**2 * dt
                                               + daily_vol * np.sqrt(dt) * z[:, t-1])
    return {
        "p05":    np.percentile(paths, 5,  axis=0),
        "p25":    np.percentile(paths, 25, axis=0),
        "p50":    np.percentile(paths, 50, axis=0),
        "p75":    np.percentile(paths, 75, axis=0),
        "p95":    np.percentile(paths, 95, axis=0),
        "paths":  paths,
    }

# ============================================================
# 4-C. 통합 예측 차트 (과거 캔들 + 최신 실제 + 미래 몬테카를로)
# ============================================================
def build_forecast_chart(ticker: str, ticker_ohlcv: pd.DataFrame,
                         volatility: float, display_days: int = 60) -> go.Figure:
    """
    [과거 캔들] + [최신 실제 가격] + [몬테카를로 미래 예측 밴드] 통합 차트.
    """
    hist = ticker_ohlcv.sort_values("Date").tail(display_days).copy()
    last_date  = hist["Date"].iloc[-1]
    last_close = float(hist["Close"].iloc[-1])

    # 최신 실제 데이터 fetch
    recent = fetch_recent_prices(ticker, last_date.strftime("%Y-%m-%d"))

    # 미래 날짜 축 생성 (영업일 기준)
    base_date = recent["Date"].iloc[-1] if not recent.empty else last_date
    future_dates = pd.bdate_range(start=base_date + pd.Timedelta(days=1), periods=20)

    # 몬테카를로 실행
    mc_start_price = float(recent["Close"].iloc[-1]) if not recent.empty else last_close
    mc = run_monte_carlo(mc_start_price, volatility)
    mc_dates = [base_date] + list(future_dates)  # t=0 포함

    fig = go.Figure()

    # ① 과거 캔들 (캐시 데이터)
    fig.add_trace(go.Candlestick(
        x=hist["Date"], open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        name="과거 (캐시)", increasing_line_color="#6B7280",
        decreasing_line_color="#9CA3AF", opacity=0.7))

    # ② 최신 실제 가격 (yfinance, 캔들)
    if not recent.empty:
        fig.add_trace(go.Candlestick(
            x=recent["Date"], open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"],
            name="최신 실제가", increasing_line_color="#10B981",
            decreasing_line_color="#EF4444"))

    # ③ 몬테카를로 밴드 (5~95%, 25~75%, 중앙값)
    fig.add_trace(go.Scatter(
        x=mc_dates, y=mc["p95"], name="95th",
        line=dict(color="rgba(99,102,241,0)", width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=mc_dates, y=mc["p05"], name="5~95% 구간",
        fill="tonexty", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(99,102,241,0)", width=0)))
    fig.add_trace(go.Scatter(
        x=mc_dates, y=mc["p75"], name="75th",
        line=dict(color="rgba(99,102,241,0)", width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=mc_dates, y=mc["p25"], name="25~75% 구간",
        fill="tonexty", fillcolor="rgba(99,102,241,0.20)",
        line=dict(color="rgba(99,102,241,0)", width=0)))
    fig.add_trace(go.Scatter(
        x=mc_dates, y=mc["p50"], name="중앙값 (50th)",
        line=dict(color="#6366F1", width=2.5, dash="dot"),
        hovertemplate="중앙값: $%{y:.2f}<extra></extra>"))

    # ④ +10% / -10% 기준선
    profit_line = mc_start_price * 1.10
    loss_line   = mc_start_price * 0.90
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
        y0=profit_line, y1=profit_line, yref="y",
        line=dict(color="#10B981", width=1.5, dash="dash"))
    fig.add_annotation(x=1, y=profit_line, xref="paper", yref="y",
        text=f"+10% (${profit_line:.1f})", showarrow=False,
        font=dict(color="#10B981", size=11), xanchor="left")

    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
        y0=loss_line, y1=loss_line, yref="y",
        line=dict(color="#EF4444", width=1.5, dash="dash"))
    fig.add_annotation(x=1, y=loss_line, xref="paper", yref="y",
        text=f"-10% (${loss_line:.1f})", showarrow=False,
        font=dict(color="#EF4444", size=11), xanchor="left")

    # ⑤ 현재/예측 경계선 (plotly 6.x 호환 방식)
    fig.add_shape(type="line",
        x0=str(base_date)[:10], x1=str(base_date)[:10],
        y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="#F59E0B", width=1.5, dash="dot"))
    fig.add_annotation(
        x=str(base_date)[:10], y=1, xref="x", yref="paper",
        text="예측 시작", showarrow=False,
        font=dict(color="#F59E0B", size=11),
        xanchor="left", yanchor="top", bgcolor="white", opacity=0.8)

    fig.update_layout(
        height=480, margin=dict(l=10, r=80, t=30, b=10),
        xaxis_rangeslider_visible=False, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        title=dict(text=f"{ticker} — 과거·실제·예측 통합 차트 (몬테카를로 500경로)", font_size=14))
    return fig


def build_chart(ticker_df: pd.DataFrame, period_days: int) -> go.Figure:
    df = ticker_df.sort_values("Date").tail(period_days).copy()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16], vertical_spacing=0.025,
        subplot_titles=["", "RSI (14)", "ADX (14)", "Volatility (20일)"])
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#10B981", decreasing_line_color="#EF4444"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20",
        line=dict(color="#3B82F6", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB 상단",
        line=dict(color="#8B5CF6", width=1, dash="dot"), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB 하단",
        line=dict(color="#8B5CF6", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(139,92,246,0.07)", opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
        line=dict(color="#F59E0B", width=1.8)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#EF4444", opacity=0.55, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#10B981", opacity=0.55, row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["ADX"], name="ADX",
        line=dict(color="#06B6D4", width=1.8)), row=3, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color="#6B7280", opacity=0.55, row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Volatility"], name="Volatility",
        line=dict(color="#EC4899", width=1.8),
        fill="tozeroy", fillcolor="rgba(236,72,153,0.10)"), row=4, col=1)
    fig.update_layout(height=680, margin=dict(l=10, r=10, t=28, b=10),
        xaxis_rangeslider_visible=False, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="ADX", range=[0, 65], row=3, col=1)
    fig.update_yaxes(title_text="변동성", row=4, col=1)
    return fig

# ============================================================
# 5. 사이드바
# ============================================================
meta, scores_df = load_scores()
ohlcv_df = load_ohlcv()
company_db = load_company_info_all()  # ← 전체 기업 정보 한 번에 로드 (정적 JSON)

with st.sidebar:
    st.markdown("## 📈 SurviQuant")
    st.caption("생존분석 × AI 기반 S&P500 대시보드")
    st.divider()
    persona_key = st.radio("🎯 투자 성향 선택", options=list(PERSONA_PRESETS.keys()),
        format_func=lambda k: PERSONA_PRESETS[k]["label"], index=1,
        help="성향 변경 시 AI Score가 실시간 재계산됩니다.")
    st.caption(f"💡 {PERSONA_PRESETS[persona_key]['desc']}")
    st.caption(f"가중치 — 수익 **{PERSONA_PRESETS[persona_key]['w_profit']:.2f}** / 방어 **{PERSONA_PRESETS[persona_key]['w_defense']:.2f}** (λ={PERSONA_PRESETS[persona_key]['lambda']})")
    st.divider()
    st.markdown("##### 📋 모델 정보")
    st.caption(f"📅 기준일: **{meta['reference_date']}**")
    st.caption(f"⏱ Horizon: **{meta['horizon_days']}영업일**")
    st.caption(f"📊 C-index — Profit **{meta['model_performance_c_index']['profit_model']}** / Loss **{meta['model_performance_c_index']['loss_model']}**")
    universe = meta.get("universe", {})
    st.caption(f"🔬 생존분석 대상: **{universe.get('survival_records_tickers', 223)}개** 종목")
    st.caption(f"🤖 RSF 추론 대상: **{universe.get('modeled_tickers', 50)}개** (5섹터 대표주)")

# ============================================================
# 6. AI Score 계산 + 정렬
# ============================================================
scores_df = scores_df.copy()
scores_df["AI_Score"] = compute_ai_score(
    scores_df["Profit_Chance"], scores_df["Loss_Risk"], persona_key).round(2)
scores_df[["Signal", "SignalColor"]] = scores_df["AI_Score"].apply(
    lambda s: pd.Series(classify_signal(s)))
scores_df = scores_df.sort_values("AI_Score", ascending=False).reset_index(drop=True)

# ============================================================
# 7. 헤더 + FAQ 팝업 + 탭
# ============================================================
title_col, info_col = st.columns([11, 1])
with title_col:
    st.title("📈 SurviQuant")
    st.markdown("**생존분석 기반 S&P 500 AI 투자 대시보드** — 단순 방향성이 아닌 *'도달 확률 + 시점'* 을 정량 제공")
with info_col:
    with st.popover("ℹ️", use_container_width=True):
        st.markdown("### 📖 SurviQuant 방법론 FAQ")
        st.divider()
        st.markdown("**Q. 왜 생존분석인가?**")
        st.markdown("""
일반적인 모델은 방향성(상승/하락)만 예측합니다.
SurviQuant는 **'언제 도달하는가'** 를 함께 예측합니다.
- 수익 사건: 매수 시점 대비 **+10% 도달**까지의 시간
- 손실 사건: 매수 시점 대비 **−10% 도달**까지의 시간
- 예측 Horizon: 고정 **20영업일**
        """)
        st.divider()
        st.markdown("**Q. 데이터 범위는?**")
        st.markdown("""
- 학습 기간: **2010.01 ~ 2026.04 (약 16년)**
- 생존분석 레코드: **853,504건**
- EDA 대상: S&P 500 **223개** 종목
- RSF 추론 대상: **50개** (5섹터 대표주)
        """)
        st.divider()
        st.markdown("**Q. AI Score는 왜 이 공식인가?**")
        st.code("AI Score = Profit_Chance × w_profit + (100−Loss_Risk) × w_defense", language="python")
        st.markdown("수익 동력과 하방 방어를 동시에 반영해 단순 확률보다 실용적인 지표를 제공합니다.")
        st.divider()
        st.markdown("**Q. 가중치 설정 근거는?**")
        st.markdown("""
**Tversky & Kahneman (1992)** Prospect Theory 손실 회피 계수 **λ ≈ 2.25**
- 안정형: 31/69 (λ=2.25) — 손실 최소화 우선
- 중립형: 50/50 (λ=1.0) — 균형
- 공격형: 69/31 (λ=0.44) — 고수익 추구
        """)
        st.divider()
        st.markdown("**Q. 확장 가능성은?**")
        st.success("코어 규칙 유지 + 도메인 규칙 교체만으로 KOSPI / ETF / 섹터 심화 분석 모두 적용 가능한 모듈형 구조입니다.")
        st.divider()
        st.markdown("**Q. 모델 성능은?**")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("수익 C-index", "0.7087")
        with mc2:
            st.metric("손실 C-index", "0.7681")
        st.caption("C-index 0.7 이상 = 임상·금융 분야 실용 수준. 핵심 변수: Volatility (HR 수익 2.87 / 손실 5.28)")
        st.caption("⚠️ 본 도구는 참고용 정량 지표이며 투자 권유가 아닙니다.")

tab1, tab2 = st.tabs(["🏆 Today's Top Picks", "📊 종목 상세 분석"])

# ============================================================
# Tab 1 — Today's Top Picks
# ============================================================
with tab1:
    sector_options = ["전체"] + sorted(scores_df["Sector"].unique().tolist())
    col_filter, col_persona = st.columns([3, 1])
    with col_filter:
        sel_sector = st.selectbox("🏭 섹터 필터", options=sector_options,
            help="섹터별로 Top10을 따로 보려면 선택하세요.")
    with col_persona:
        st.metric("선택된 성향", PERSONA_PRESETS[persona_key]["label"])

    filtered = (scores_df if sel_sector == "전체"
                else scores_df[scores_df["Sector"] == sel_sector].reset_index(drop=True))

    st.markdown("#### 📍 시그널 분포")
    sig_counts = filtered["Signal"].value_counts()
    chip_cols = st.columns(4)
    for i, (label, threshold, color) in enumerate(SIGNAL_RULES):
        with chip_cols[i]:
            count = int(sig_counts.get(label, 0))
            st.markdown(
                f"""<div style="background:{color}15;border-left:5px solid {color};
                    padding:14px 18px;border-radius:8px;min-height:88px;">
                    <div style="font-size:13px;color:#555;font-weight:500;">{label}</div>
                    <div style="font-size:30px;font-weight:700;color:{color};line-height:1.2;">
                        {count}<span style="font-size:14px;color:#999;font-weight:400;"> 종목</span>
                    </div>
                    <div style="font-size:11px;color:#999;">
                        {"≥ " + str(threshold) + "점" if threshold > 0 else "< 40점"}
                    </div></div>""",
                unsafe_allow_html=True)

    st.divider()
    top_n = min(10, len(filtered))
    st.markdown(f"#### 🏆 Top {top_n} — {sel_sector}")
    st.caption("성향 토글 시 AI Score가 즉시 재계산되어 순위가 바뀝니다.")

    for i, row in filtered.head(top_n).iterrows():
        rank = i + 1
        color = row["SignalColor"]
        rank_color = "#FFD700" if rank == 1 else ("#C0C0C0" if rank == 2 else ("#CD7F32" if rank == 3 else "#AAAAAA"))
        cols = st.columns([0.5, 1.6, 2.2, 1.4, 1.4, 1.4])
        with cols[0]:
            st.markdown(f"<h1 style='color:{rank_color};margin:8px 0 0 0;font-size:34px;'>#{rank}</h1>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"### {row['Ticker']}")
            st.caption(f"📂 {row['Sector']}")
        with cols[2]:
            st.markdown(f"""<div style="margin-top:14px;"><span style="background:{color};color:white;
                padding:8px 18px;border-radius:22px;font-weight:600;font-size:14px;">
                {row['Signal']}</span></div>""", unsafe_allow_html=True)
        with cols[3]:
            st.metric("AI Score", f"{row['AI_Score']:.1f}", help="0~100점. 성향별 가중치로 즉시 재계산됩니다.")
        with cols[4]:
            st.metric("수익 도달 확률", f"{row['Profit_Chance']:.1f}%", help="+10% 수익 도달 확률 (Horizon: 20영업일)")
        with cols[5]:
            st.metric("손실 발생 확률", f"{row['Loss_Risk']:.1f}%", help="-10% 손실 도달 확률 (Horizon: 20영업일)", delta_color="inverse")
        st.divider()

    with st.expander(f"📋 전체 {len(filtered)}종목 보기"):
        display_df = filtered[["Ticker", "Sector", "AI_Score", "Profit_Chance", "Loss_Risk", "Signal"]].copy()
        display_df.columns = ["티커", "섹터", "AI Score", "수익 확률(%)", "손실 확률(%)", "시그널"]
        st.dataframe(display_df, use_container_width=True, hide_index=True,
            column_config={
                "AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.1f"),
                "수익 확률(%)": st.column_config.NumberColumn(format="%.1f%%"),
                "손실 확률(%)": st.column_config.NumberColumn(format="%.1f%%"),
            })

# ============================================================
# Tab 2 — 종목 상세 분석 (2컬럼: 분석 | 차트)
# ============================================================
with tab2:
    # 상단 컨트롤
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        sel_ticker = st.selectbox("🔍 종목 선택", options=scores_df["Ticker"].tolist(),
            help="AI Score 내림차순 정렬")
    with ctrl2:
        sel_period_label = st.selectbox("📅 조회 기간", options=list(PERIOD_OPTIONS.keys()), index=1)
    with ctrl3:
        sel_sector_detail = st.selectbox("🏭 섹터",
            options=["전체"] + sorted(scores_df["Sector"].unique().tolist()), index=0)

    period_days  = PERIOD_OPTIONS[sel_period_label]
    row          = scores_df[scores_df["Ticker"] == sel_ticker].iloc[0]
    ticker_ohlcv = ohlcv_df[ohlcv_df["Ticker"] == sel_ticker]
    w            = PERSONA_PRESETS[persona_key]
    latest       = ticker_ohlcv.sort_values("Date").iloc[-1]

    contrib_profit  = round(row["Profit_Chance"] * w["w_profit"], 2)
    contrib_defense = round((100 - row["Loss_Risk"]) * w["w_defense"], 2)
    ai_score        = round(contrib_profit + contrib_defense, 2)
    signal_label, signal_color = classify_signal(ai_score)

    # 종목 헤더
    st.markdown(
        f"### {sel_ticker} &nbsp;"
        f"<span style='background:{signal_color};color:white;padding:5px 16px;"
        f"border-radius:20px;font-size:15px;font-weight:600;'>{signal_label}</span>",
        unsafe_allow_html=True)
    st.caption(f"📂 {row['Sector']}  |  성향: {PERSONA_PRESETS[persona_key]['label']}")
    st.divider()

    # ── 2컬럼 메인 레이아웃
    col_left, col_right = st.columns([1, 1.6])

    # ════════════════════════════════
    # 왼쪽: 분석 패널
    # ════════════════════════════════
    with col_left:

        # 기업 개요
        company = get_company_info(sel_ticker, company_db)
        with st.expander(f"🏢 {company['name']} — 기업 개요", expanded=False):
            st.markdown(f"**{company['name']}** ({sel_ticker})")
            if company["industry"]:
                st.caption(f"업종: {company['industry']}")
            st.markdown(company["summary"])
            if company["website"]:
                st.markdown(f"🔗 [공식 웹사이트]({company['website']})")
            info_c1, info_c2 = st.columns(2)
            with info_c1:
                if company["market_cap"]:
                    st.metric("시가총액", f"${company['market_cap']/1e9:,.1f}B")
            with info_c2:
                if company["employees"]:
                    st.metric("임직원 수", f"{company['employees']:,}명")

        st.markdown("#### 🏆 AI Score")

        # AI Score 카드
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("AI Score", f"{ai_score:.1f}", help="수익 기여 + 방어 기여의 합산 (0~100점)")
        with c2:
            st.metric("📈 수익 기여", f"{contrib_profit:.1f}점",
                      f"Profit {row['Profit_Chance']:.1f}% × {w['w_profit']:.2f}")
        with c3:
            st.metric("🛡 방어 기여", f"{contrib_defense:.1f}점",
                      f"Defense {100-row['Loss_Risk']:.1f}% × {w['w_defense']:.2f}")

        # AI Score 분해 막대
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=[contrib_profit], y=["AI Score"], orientation="h",
            name="수익 기여", marker_color="#10B981",
            text=[f"수익 기여 {contrib_profit:.1f}점"],
            textposition="inside", textfont=dict(color="white", size=12)))
        fig_bar.add_trace(go.Bar(x=[contrib_defense], y=["AI Score"], orientation="h",
            name="방어 기여", marker_color="#3B82F6",
            text=[f"방어 기여 {contrib_defense:.1f}점"],
            textposition="inside", textfont=dict(color="white", size=12)))
        fig_bar.update_layout(barmode="stack", height=72,
            margin=dict(l=0, r=0, t=4, b=4),
            xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", y=1.8, x=0), template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # 생존 곡선
        surv_title_col, surv_info_col = st.columns([5, 1])
        with surv_title_col:
            st.markdown("#### 📉 생존분석 — 도달 확률 추이")
        with surv_info_col:
            with st.popover("ℹ️"):
                st.markdown("**생존분석 도달 확률 추이**")
                st.markdown("""
RSF 모델의 t=20 종점 확률을 지수분포로 역산한 **근사 누적 곡선**입니다.

- 🟢 **초록선**: 매수 후 +10% 수익에 도달할 누적 확률
- 🔴 **빨간선**: 매수 후 -10% 손실에 도달할 누적 확률
- 두 곡선의 격차가 클수록 수익/손실 비대칭이 유리합니다.

> 실제 RSF가 출력한 곡선이 아닌 종점 확률 기반 근사치입니다.
                """)

        days = np.arange(1, 21)
        p_profit = row["Profit_Chance"] / 100
        p_loss   = row["Loss_Risk"] / 100
        lam_profit = -np.log(1 - p_profit + 1e-9) / 20
        lam_loss   = -np.log(1 - p_loss   + 1e-9) / 20
        cum_profit = (1 - np.exp(-lam_profit * days)) * 100
        cum_loss   = (1 - np.exp(-lam_loss   * days)) * 100

        fig_surv = go.Figure()
        fig_surv.add_trace(go.Scatter(
            x=days, y=cum_profit, name="수익 도달 (+10%)",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
            hovertemplate="Day %{x}: %{y:.1f}%<extra>수익 도달</extra>"))
        fig_surv.add_trace(go.Scatter(
            x=days, y=cum_loss, name="손실 도달 (-10%)",
            line=dict(color="#EF4444", width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
            hovertemplate="Day %{x}: %{y:.1f}%<extra>손실 도달</extra>"))
        fig_surv.add_trace(go.Scatter(
            x=[20], y=[cum_profit[-1]], mode="markers+text",
            marker=dict(color="#10B981", size=9),
            text=[f"{cum_profit[-1]:.1f}%"], textposition="top right", showlegend=False))
        fig_surv.add_trace(go.Scatter(
            x=[20], y=[cum_loss[-1]], mode="markers+text",
            marker=dict(color="#EF4444", size=9),
            text=[f"{cum_loss[-1]:.1f}%"], textposition="bottom right", showlegend=False))
        fig_surv.add_shape(type="line", x0=20, x1=20, y0=0, y1=1,
            xref="x", yref="paper", line=dict(color="#9CA3AF", dash="dot", width=1.2))
        fig_surv.add_annotation(x=20, y=1, xref="x", yref="paper",
            text="Horizon", showarrow=False, font=dict(color="#9CA3AF", size=10),
            xanchor="right", yanchor="top")
        fig_surv.update_layout(
            height=240, margin=dict(l=0, r=10, t=10, b=0),
            xaxis=dict(title="영업일", tickmode="linear", dtick=4),
            yaxis=dict(title="누적 확률 (%)",
                       range=[0, max(cum_profit[-1], cum_loss[-1]) * 1.3]),
            legend=dict(orientation="h", y=1.15, x=0),
            template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_surv, use_container_width=True)

        st.divider()

        # AI 인사이트
        insight_title_col, insight_info_col = st.columns([5, 1])
        with insight_title_col:
            st.markdown("#### 💡 AI 인사이트")
        with insight_info_col:
            with st.popover("ℹ️"):
                st.markdown("**AI 인사이트란?**")
                st.markdown("""
최신 기술지표 값을 기반으로 자동 생성된 해석입니다.

- **RSI**: 과매수/과매도 판단
- **ADX**: 추세 강도 확인
- **Volatility**: 변동성 수준 경고
- **MA20**: 단기 추세 위치 확인

> 각 지표의 임계값 기준으로 4가지 인사이트가 자동 생성됩니다.
                """)

        insights = generate_insights(
            rsi=float(latest["RSI"]), adx=float(latest["ADX"]),
            volatility=float(latest["Volatility"]),
            close=float(latest["Close"]), ma20=float(latest["MA20"]),
        )
        for icon, label, message in insights:
            st.markdown(
                f"""<div style="background:#F8FAFC;border-left:4px solid #3B82F6;
                    padding:10px 14px;border-radius:8px;margin-bottom:8px;">
                    <div style="font-size:12px;font-weight:600;color:#374151;">
                        {icon} {label}
                    </div>
                    <div style="font-size:12px;color:#6B7280;margin-top:3px;">
                        {message}
                    </div></div>""",
                unsafe_allow_html=True)

        st.divider()
        st.caption("⚠️ 본 대시보드는 학술·시연 목적입니다. 투자 권유 또는 자문이 아닙니다.")

    # ════════════════════════════════
    # 오른쪽: 차트 패널
    # ════════════════════════════════
    with col_right:

        # 몬테카를로 차트
        mc_title_col, mc_info_col = st.columns([5, 1])
        with mc_title_col:
            st.markdown("#### 🔮 미래 가격 시나리오")
        with mc_info_col:
            with st.popover("ℹ️"):
                st.markdown("**몬테카를로 시뮬레이션**")
                st.markdown("""
GBM(기하 브라운 운동) 기반 **500개 경로** 시뮬레이션입니다.

- 🩶 **회색 캔들**: 캐시 기반 과거 데이터
- 🟢🔴 **컬러 캔들**: yfinance 최신 실제 가격
- 🟣 **보라 밴드**: 향후 20영업일 예측 시나리오
  - 진한 밴드: 25~75% 구간 (50% 경로가 이 안에)
  - 연한 밴드: 5~95% 구간 (90% 경로가 이 안에)
  - 점선: 중앙값(50th) 경로
- 🟢 **초록 점선**: +10% 수익 기준선
- 🔴 **빨간 점선**: -10% 손실 기준선

> 통계적 시나리오이며 실제 수익을 보장하지 않습니다.
                """)

        if not YF_AVAILABLE:
            st.caption("*(yfinance 미설치 환경 — 캐시 데이터 기반으로 시뮬레이션)*")

        fig_mc = build_forecast_chart(
            ticker=sel_ticker,
            ticker_ohlcv=ticker_ohlcv,
            volatility=float(latest["Volatility"]),
            display_days=60,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # 기술지표 차트
        chart_title_col, chart_info_col = st.columns([5, 1])
        with chart_title_col:
            st.markdown("#### 📊 기술지표 차트")
        with chart_info_col:
            with st.popover("ℹ️"):
                st.markdown("**기술지표 안내**")
                st.markdown("""
**① MA20 (오버레이)**
20일 단순 이동평균선. 종가가 MA20 위 → 단기 상승 추세.

**② 볼린저 밴드 (오버레이)**
가격 변동 범위. 상단 돌파 → 과매수, 하단 이탈 → 과매도 신호.

**③ RSI (서브차트)**
0~100 범위. **70 이상 과매수** (조정 가능성), **30 이하 과매도** (반등 가능성).

**④ ADX (서브차트)**
추세 강도. **25 이상** → 의미 있는 추세, 25 미만 → 횡보.

**⑤ Volatility (서브차트)**
20일 로그 수익률 표준편차. **높을수록 리스크 증가. SurviQuant 핵심 예측 변수** (손실 HR 5.28).
                """)

        fig_chart = build_chart(ticker_ohlcv, period_days)
        st.plotly_chart(fig_chart, use_container_width=True)

        # 최신 지표 스냅샷
        st.markdown("##### 📌 최신 지표 스냅샷")
        snap_cols = st.columns(5)
        for col, (label, value, tip) in zip(snap_cols, [
            ("종가",       f"${latest['Close']:.2f}",    None),
            ("RSI",        f"{latest['RSI']:.1f}",        "70↑ 과매수 / 30↓ 과매도"),
            ("ADX",        f"{latest['ADX']:.1f}",        "25 이상이면 추세 존재"),
            ("Volatility", f"{latest['Volatility']:.3f}", "높을수록 변동성 큼 — 핵심 예측 변수"),
            ("MA20",       f"${latest['MA20']:.2f}",      "20일 이동평균선"),
        ]):
            with col:
                st.metric(label, value, help=tip)

# ============================================================
# 8. 푸터
# ============================================================
st.divider()
st.caption("ℹ️ 본 대시보드는 학술·시연 목적으로 제작되었습니다. 투자 권유 또는 자문이 아닙니다.")
# ============================================================
# 8. 푸터
# ============================================================
st.divider()
st.caption("ℹ️ 본 대시보드는 학술·시연 목적으로 제작되었습니다. 투자 권유 또는 자문이 아닙니다.")
