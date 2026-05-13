"""
SurviQuant — 생존분석 기반 S&P500 AI 투자 대시보드
실행: streamlit run dashboard.py
데이터: ./data/scores.json, ./data/ohlcv_cache.parquet, ./data/company_info.json
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
# 0. 페이지 설정 + 전역 CSS
# ============================================================
st.set_page_config(
    page_title="SurviQuant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 컨테이너 border-radius 및 전체 폰트 밸런스 조정 CSS
st.markdown("""
<style>
[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
}
/* Metric(숫자) 폰트 크기 살짝 축소하여 조화롭게 */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 1. 데이터 로딩 (기존과 동일)
# ============================================================
@st.cache_data(show_spinner="데이터 로딩 중...")
def load_scores():
    with open(DATA_DIR / "scores.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["meta"], pd.DataFrame(payload["scores"])

@st.cache_data(show_spinner="OHLCV 데이터 로딩 중...")
def load_ohlcv():
    parquet_path = DATA_DIR / "ohlcv_cache.parquet"
    csv_path     = DATA_DIR / "ohlcv_cache.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data(show_spinner="기업 정보 로딩 중...")
def load_company_info_all() -> dict:
    path = DATA_DIR / "company_info.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_company_info(ticker: str, company_db: dict) -> dict:
    if ticker in company_db:
        return company_db[ticker]
    return {
        "name": ticker, "summary": "기업 정보를 불러올 수 없습니다.",
        "website": "", "market_cap": None, "employees": None, "industry": "",
    }

# ============================================================
# 2. 도메인 로직
# ============================================================
PERSONA_PRESETS = {
    "conservative": {"label": "안정형", "w_profit": 0.31, "w_defense": 0.69, "lambda": 2.25},
    "neutral":      {"label": "중립형", "w_profit": 0.50, "w_defense": 0.50, "lambda": 1.00},
    "aggressive":   {"label": "공격형", "w_profit": 0.69, "w_defense": 0.31, "lambda": 0.44},
}

SIGNAL_RULES = [
    ("강력 매수", 60, "#10B981"),
    ("매수 고려", 50, "#84CC16"),
    ("관망",      40, "#F59E0B"),
    ("비중 축소",  0, "#EF4444"),
]

PERIOD_OPTIONS = {
    "3개월 (63일)": 63,
    "6개월 (126일)": 126,
    "1년 (252일)": 252,
    "3년 (756일)": 756,
    "5년 (1260일)": 1260,
    "전체": 9999,
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
# 3. 인사이트 자동 생성
# ============================================================
def generate_insights(rsi, adx, volatility, close, ma20):
    insights = []
    if rsi >= 70:
        insights.append(("RSI 과매수", f"RSI {rsi:.0f} — 단기 조정 가능성, 신규 진입 시 주의가 필요합니다."))
    elif rsi <= 30:
        insights.append(("RSI 과매도", f"RSI {rsi:.0f} — 반등 가능성 구간, 분할 매수를 고려해볼 수 있습니다."))
    else:
        insights.append(("RSI 중립", f"RSI {rsi:.0f} — 과매수·과매도 구간 아님, 추세 지속 가능한 구간입니다."))

    if adx >= 40:
        insights.append(("강한 추세", f"ADX {adx:.0f} — 매우 강한 추세가 진행 중입니다."))
    elif adx >= 25:
        insights.append(("추세 존재", f"ADX {adx:.0f} — 방향성 추세가 확인됩니다."))
    else:
        insights.append(("횡보 구간", f"ADX {adx:.0f} — 뚜렷한 추세 없음, 관망을 고려하세요."))

    if volatility >= 0.4:
        insights.append(("고변동성", f"Vol {volatility:.3f} — 가격 변동 폭이 매우 큽니다. 손절 라인 설정 필수."))
    elif volatility >= 0.2:
        insights.append(("중간 변동성", f"Vol {volatility:.3f} — 보통 수준의 변동성입니다."))
    else:
        insights.append(("저변동성", f"Vol {volatility:.3f} — 안정적인 가격 흐름입니다."))

    gap_pct = (close - ma20) / ma20 * 100
    if gap_pct > 5:
        insights.append(("MA20 상회", f"종가가 MA20보다 {gap_pct:.1f}% 위 — 단기 상승 추세 유지 중입니다."))
    elif gap_pct < -5:
        insights.append(("MA20 하회", f"종가가 MA20보다 {abs(gap_pct):.1f}% 아래 — 단기 하락 압력이 있습니다."))
    else:
        insights.append(("MA20 근접", f"종가가 MA20과 {gap_pct:+.1f}% 차이 — 지지·저항 테스트 구간입니다."))
    return insights

# ============================================================
# 4. 기능 함수들 (fetch, mc, chart) - 생략 없이 그대로 유지
# ============================================================
@st.cache_data(ttl=3600, show_spinner="최신 주가 데이터 가져오는 중...")
def fetch_recent_prices(ticker: str, last_cache_date: str) -> pd.DataFrame:
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        start = pd.to_datetime(last_cache_date) + pd.Timedelta(days=1)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["Ticker"] = ticker
        return df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]
    except Exception:
        return pd.DataFrame()

def run_monte_carlo(last_close, volatility, n_days=20, n_sim=500, seed=42):
    rng = np.random.default_rng(seed)
    daily_vol = volatility / np.sqrt(20)
    paths = np.zeros((n_sim, n_days + 1))
    paths[:, 0] = last_close
    z = rng.standard_normal((n_sim, n_days))
    for t in range(1, n_days + 1):
        paths[:, t] = paths[:, t-1] * np.exp(-0.5 * daily_vol**2 + daily_vol * z[:, t-1])
    return {
        "p05": np.percentile(paths, 5,  axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }

def build_unified_chart(ticker, ticker_df, volatility, period_days):
    hist = ticker_df.sort_values("Date").tail(period_days).copy()
    last_date      = hist["Date"].iloc[-1]
    last_close     = float(hist["Close"].iloc[-1])
    recent         = fetch_recent_prices(ticker, last_date.strftime("%Y-%m-%d"))
    base_date      = recent["Date"].iloc[-1] if not recent.empty else last_date
    mc_start_price = float(recent["Close"].iloc[-1]) if not recent.empty else last_close
    future_dates   = pd.bdate_range(start=base_date + pd.Timedelta(days=1), periods=20)
    mc             = run_monte_carlo(mc_start_price, volatility)
    mc_dates       = [base_date] + list(future_dates)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16], vertical_spacing=0.02,
        subplot_titles=["", "RSI (14)", "ADX (14)", "Volatility (20일)"])

    fig.add_trace(go.Candlestick(
        x=hist["Date"], open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="과거",
        increasing_line_color="#10B981", decreasing_line_color="#EF4444"), row=1, col=1)
    if not recent.empty:
        fig.add_trace(go.Candlestick(
            x=recent["Date"], open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"], name="최신 실제가",
            increasing_line_color="#059669", decreasing_line_color="#DC2626"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MA20"], name="MA20",
        line=dict(color="#3B82F6", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["BB_Upper"], name="BB 상단",
        line=dict(color="#8B5CF6", width=1, dash="dot"), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["BB_Lower"], name="BB 하단",
        line=dict(color="#8B5CF6", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(139,92,246,0.06)", opacity=0.8), row=1, col=1)

    fig.add_trace(go.Scatter(x=mc_dates, y=mc["p95"], line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=mc_dates, y=mc["p05"], name="5~95% 시나리오", fill="tonexty", fillcolor="rgba(99,102,241,0.09)", line=dict(color="rgba(0,0,0,0)", width=0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=mc_dates, y=mc["p75"], line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=mc_dates, y=mc["p25"], name="25~75% 시나리오", fill="tonexty", fillcolor="rgba(99,102,241,0.18)", line=dict(color="rgba(0,0,0,0)", width=0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=mc_dates, y=mc["p50"], name="중앙값 예측", line=dict(color="#6366F1", width=2, dash="dot"), hovertemplate="중앙값: $%{y:.2f}<extra></extra>"), row=1, col=1)

    for y_val, color, label in [
        (mc_start_price * 1.10, "#10B981", f"+10%  ${mc_start_price*1.10:.1f}"),
        (mc_start_price * 0.90, "#EF4444", f"-10%  ${mc_start_price*0.90:.1f}"),
    ]:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=y_val, y1=y_val, yref="y", line=dict(color=color, width=1.2, dash="dash"))
        fig.add_annotation(x=1.01, y=y_val, xref="paper", yref="y", text=label, showarrow=False, font=dict(color=color, size=10), xanchor="left")

    boundary = str(base_date)[:10]
    fig.add_shape(type="line", x0=boundary, x1=boundary, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#F59E0B", width=1.5, dash="dot"))
    fig.add_annotation(x=boundary, y=0.97, xref="x", yref="paper", text="예측 시작", showarrow=False, font=dict(color="#F59E0B", size=10), xanchor="left", yanchor="top")

    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["RSI"], name="RSI", line=dict(color="#F59E0B", width=1.6)), row=2, col=1)
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=70, y1=70, yref="y2", line=dict(color="#EF4444", dash="dot", width=1))
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=30, y1=30, yref="y2", line=dict(color="#10B981", dash="dot", width=1))
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["ADX"], name="ADX", line=dict(color="#06B6D4", width=1.6)), row=3, col=1)
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=25, y1=25, yref="y3", line=dict(color="#6B7280", dash="dot", width=1))
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Volatility"], name="Volatility", line=dict(color="#EC4899", width=1.6), fill="tozeroy", fillcolor="rgba(236,72,153,0.09)"), row=4, col=1)

    fig.update_layout(
        height=760, margin=dict(l=10, r=90, t=28, b=10),
        xaxis_rangeslider_visible=False, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="ADX", range=[0, 65], row=3, col=1)
    fig.update_yaxes(title_text="변동성", row=4, col=1)
    return fig

# ============================================================
# 5. 데이터 로드
# ============================================================
meta, scores_df = load_scores()
ohlcv_df        = load_ohlcv()
company_db      = load_company_info_all()

# ============================================================
# 6. 사이드바
# ============================================================
with st.sidebar:
    st.markdown("## 📈 SurviQuant")
    st.caption("생존분석 기반 S&P 500 AI 투자 대시보드")
    st.divider()

    st.markdown("#### 투자 성향")
    persona_key = st.radio(
        "성향 선택",
        options=list(PERSONA_PRESETS.keys()),
        format_func=lambda k: PERSONA_PRESETS[k]["label"],
        index=1,
        label_visibility="collapsed",
    )
    w_preset = PERSONA_PRESETS[persona_key]
    st.caption(f"수익 {w_preset['w_profit']*100:.0f}% / 방어 {w_preset['w_defense']*100:.0f}%")
    st.divider()

    st.markdown("#### 모델 정보")
    universe = meta.get("universe", {})
    c_index  = meta.get("model_performance_c_index", {})
    today    = pd.Timestamp.now().strftime("%Y-%m-%d")
    profit_pct = round(c_index.get("profit_model", 0) * 100)
    loss_pct   = round(c_index.get("loss_model",   0) * 100)

    st.caption(f"기준일: **{today}**")
    st.caption("신뢰도 (C-index)")
    st.caption(f"◦ 수익 모델: **약 {profit_pct}%**")
    st.caption(f"◦ 손실 모델: **약 {loss_pct}%**")
    st.divider()
    st.caption(f"생존분석 대상: **{universe.get('survival_records_tickers', 223)}개** 종목")
    st.caption(f"RSF 학습·추론: **{universe.get('modeled_tickers', 50)}개** 종목")

    with st.expander("50개 종목 선정 기준"):
        st.markdown("""
5개 섹터(IT·헬스케어·소비재·에너지·통신) 대표주를 시가총액 기준으로 선별하였습니다.
전체 223개 종목으로 RSF를 학습하면 GitHub Actions 제한 시간(6시간)을 초과할 수 있어, 섹터 다양성을 유지하면서 연산 효율을 확보하기 위해 50개로 한정하였습니다. EDA 및 생존분석 레코드는 223개 전 종목 기반으로 산출되었습니다.
        """)

# ============================================================
# 7. AI Score 계산 + 정렬
# ============================================================
scores_df = scores_df.copy()
scores_df["AI_Score"] = compute_ai_score(
    scores_df["Profit_Chance"], scores_df["Loss_Risk"], persona_key).round(2)
scores_df[["Signal", "SignalColor"]] = scores_df["AI_Score"].apply(
    lambda s: pd.Series(classify_signal(s)))
scores_df = scores_df.sort_values("AI_Score", ascending=False).reset_index(drop=True)

# ============================================================
# 8. 헤더
# ============================================================
title_col, info_col = st.columns([11, 1])
with title_col:
    st.title("📈 SurviQuant")
    st.caption("생존분석 기반 S&P 500 AI 투자 대시보드")
with info_col:
    with st.popover("ⓘ", use_container_width=True):
        st.markdown("### SurviQuant 방법론")
        st.divider()
        st.markdown("**Q. 왜 생존분석인가?**")
        st.markdown("""
일반적인 모델은 방향성(상승/하락)만 예측합니다.
SurviQuant는 **'언제 도달하는가'** 를 함께 예측합니다.
◦ 수익 사건: 매수 시점 대비 +10% 도달까지의 시간
◦ 손실 사건: 매수 시점 대비 -10% 도달까지의 시간
◦ 예측 Horizon: 고정 20영업일
        """)
        st.divider()
        st.markdown("**Q. 데이터 범위는?**")
        st.markdown("""
◦ 학습 기간: 2010.01 ~ 2026.04
◦ 생존분석 레코드: 853,504건
◦ EDA 대상: S&P 500 223개 종목
◦ RSF 추론 대상: 50개 (5섹터 대표주)
        """)
        st.divider()
        st.markdown("**Q. AI Score 공식은?**")
        st.code("AI Score = Profit_Chance × w_profit\n         + (100 - Loss_Risk) × w_defense", language="python")
        st.markdown("수익 동력과 하방 방어를 동시에 반영해 단순 확률보다 실용적인 지표를 제공합니다.")
        st.divider()
        st.markdown("**Q. 가중치 설정 근거는?**")
        st.markdown("""
Tversky & Kahneman (1992) 손실 회피 계수 λ ≈ 2.25 기반 설계
◦ 안정형 (λ=2.25): 손실 최소화 우선
◦ 중립형 (λ=1.0): 균형
◦ 공격형 (λ=0.44): 고수익 추구
        """)

st.divider()

# ============================================================
# 9. 컨트롤 [섹터 | 종목선택 | 조회기간]
# ============================================================
ctrl1, ctrl2, ctrl3 = st.columns([1, 2, 2])
with ctrl1:
    sel_sector = st.selectbox("섹터", options=["전체"] + sorted(scores_df["Sector"].unique().tolist()), index=0)
with ctrl2:
    ticker_options = scores_df["Ticker"].tolist() if sel_sector == "전체" else scores_df[scores_df["Sector"] == sel_sector]["Ticker"].tolist()
    sel_ticker = st.selectbox("종목 선택", options=ticker_options, help="AI Score 내림차순 정렬")
with ctrl3:
    sel_period_label = st.selectbox("조회 기간", options=list(PERIOD_OPTIONS.keys()), index=len(PERIOD_OPTIONS) - 1)

period_days  = PERIOD_OPTIONS[sel_period_label]
row          = scores_df[scores_df["Ticker"] == sel_ticker].iloc[0]
ticker_ohlcv = ohlcv_df[ohlcv_df["Ticker"] == sel_ticker]
latest       = ticker_ohlcv.sort_values("Date").iloc[-1]

contrib_profit  = round(row["Profit_Chance"] * w_preset["w_profit"], 2)
contrib_defense = round((100 - row["Loss_Risk"]) * w_preset["w_defense"], 2)
ai_score        = round(contrib_profit + contrib_defense, 2)
signal_label, signal_color = classify_signal(ai_score)

# ============================================================
# 10. 티커 헤더 + 기업 개요
# ============================================================
ticker_col, company_col = st.columns([1, 2])

with ticker_col:
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>"
            f"<h2 style='margin:0; padding:0;'>{sel_ticker}</h2>"
            f"<span style='background:{signal_color};color:white;padding:4px 12px;border-radius:12px;font-size:14px;font-weight:600;'>{signal_label}</span>"
            f"</div>",
            unsafe_allow_html=True)
        st.caption(f"**{row['Sector']}** · {w_preset['label']} 기준")

with company_col:
    company = get_company_info(sel_ticker, company_db)
    with st.container(border=True):
        st.markdown(f"**{company['name']}**")
        info_c1, info_c2, info_c3 = st.columns(3)
        with info_c1:
            if company["market_cap"]: st.metric("시가총액", f"${company['market_cap']/1e9:,.1f}B")
        with info_c2:
            if company["employees"]: st.metric("임직원 수", f"{company['employees']:,}명")
        with info_c3:
            if company["website"]: st.markdown(f"<div style='margin-top:20px;'><a href='{company['website']}' target='_blank'>🌐 공식 웹사이트</a></div>", unsafe_allow_html=True)
        
        # 기업 설명 토글 처리
        with st.expander("📖 기업 설명 보기"):
            st.write(company["summary"])

st.divider()

# ============================================================
# 11. 3열: 수익·손실 도달 추이 | AI 인사이트 | 최신 지표 스냅샷
# ============================================================
# height=380 지정으로 세 박스 높이 완벽 통일
col_surv, col_ins, col_snap = st.columns(3)

# ── 수익·손실 도달 추이
with col_surv:
    with st.container(border=True, height=380):
        surv_h, surv_i = st.columns([5, 1])
        with surv_h:
            st.markdown("#### 📉 수익·손실 추이")
        with surv_i:
            with st.popover("ⓘ"):
                st.markdown("RSF 모델의 t=20 종점 확률을 지수분포로 역산한 근사 누적 곡선입니다.\n\n두 곡선의 격차가 클수록 수익/손실 비대칭이 유리합니다.")

        days  = np.arange(1, 21)
        lam_p = -np.log(1 - row["Profit_Chance"]/100 + 1e-9) / 20
        lam_l = -np.log(1 - row["Loss_Risk"]/100   + 1e-9) / 20
        cum_p = (1 - np.exp(-lam_p * days)) * 100
        cum_l = (1 - np.exp(-lam_l * days)) * 100

        fig_surv = go.Figure()
        fig_surv.add_trace(go.Scatter(x=days, y=cum_p, name="수익 (+10%)", line=dict(color="#10B981", width=2.2), fill="tozeroy", fillcolor="rgba(16,185,129,0.08)"))
        fig_surv.add_trace(go.Scatter(x=days, y=cum_l, name="손실 (-10%)", line=dict(color="#EF4444", width=2.2), fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
        fig_surv.add_shape(type="line", x0=20, x1=20, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#9CA3AF", dash="dot", width=1.2))
        fig_surv.update_layout(
            height=260, margin=dict(l=0, r=10, t=10, b=0),
            xaxis=dict(title="영업일", tickmode="linear", dtick=4),
            yaxis=dict(title="누적 확률 (%)", range=[0, max(cum_p[-1], cum_l[-1]) * 1.3]),
            legend=dict(orientation="h", y=1.15, x=0),
            template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_surv, use_container_width=True)

# ── AI 인사이트
with col_ins:
    with st.container(border=True, height=380):
        ins_h, ins_i = st.columns([5, 1])
        with ins_h:
            st.markdown("#### 💡 AI 인사이트")
        with ins_i:
            with st.popover("ⓘ"):
                st.markdown("최신 기술지표 값을 기반으로 자동 생성된 해석입니다.")

        insights = generate_insights(
            float(latest["RSI"]), float(latest["ADX"]),
            float(latest["Volatility"]),
            float(latest["Close"]), float(latest["MA20"]))
        
        for label, message in insights:
            st.markdown(
                f"""<div style="background:#F8FAFC;border-left:4px solid #3B82F6;
                    padding:10px 14px;border-radius:6px;margin-bottom:10px;">
                    <div style="font-size:13px;font-weight:700;color:#1F2937;">{label}</div>
                    <div style="font-size:12px;color:#4B5563;margin-top:4px;line-height:1.4;">{message}</div>
                </div>""", unsafe_allow_html=True)

# ── 최신 지표 스냅샷
with col_snap:
    with st.container(border=True, height=380):
        st.markdown("#### 📌 최신 지표 스냅샷")
        st.metric("현재 종가", f"${latest['Close']:.2f}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        with r1c1: st.metric("RSI", f"{latest['RSI']:.1f}")
        with r1c2: st.metric("ADX", f"{latest['ADX']:.1f}")
        with r2c1: st.metric("Volatility", f"{latest['Volatility']:.3f}")
        with r2c2: st.metric("MA20", f"${latest['MA20']:.2f}")

st.divider()

# ============================================================
# 12. 종목 차트 (전체 너비)
# ============================================================
with st.container(border=True):
    chart_h, chart_i = st.columns([10, 1])
    with chart_h:
        st.markdown("#### 📊 종합 차트 및 몬테카를로 시뮬레이션")
    with chart_i:
        with st.popover("ⓘ"):
            st.markdown("노란 점선 우측은 GBM 기반 500경로 시뮬레이션 결과입니다.")

    fig_unified = build_unified_chart(
        ticker=sel_ticker,
        ticker_df=ticker_ohlcv,
        volatility=float(latest["Volatility"]),
        period_days=period_days,
    )
    st.plotly_chart(fig_unified, use_container_width=True)

st.divider()

# ============================================================
# 13. AI Score 설명 (전체 너비)
# ============================================================
with st.container(border=True):
    st.markdown("#### 🎯 AI Score 분석")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("최종 AI Score", f"{ai_score:.1f}점")
    with sc2:
        st.metric("수익 기여도", f"{contrib_profit:.1f}점")
    with sc3:
        st.metric("방어 기여도", f"{contrib_defense:.1f}점")
    with sc4:
        st.markdown(
            f"<div style='padding-top:14px; text-align:center;'>"
            f"<span style='background:{signal_color};color:white;"
            f"padding:8px 24px;border-radius:24px;"
            f"font-weight:700;font-size:16px;'>종합 판정: {signal_label}</span>"
            f"</div>",
            unsafe_allow_html=True)

    # 수식 설명 박스
    st.markdown(
        f"<div style='margin-top:20px; padding:15px; background-color:#F3F4F6; border-radius:8px; text-align:center;'>"
        f"<span style='color:#4B5563; font-size:14px;'>"
        f"<b>계산식:</b> (수익 도달 확률 <b>{row['Profit_Chance']:.1f}%</b> × 가중치 <b>{w_preset['w_profit']:.2f}</b>) "
        f"+ (손실 방어 확률 <b>{100 - row['Loss_Risk']:.1f}%</b> × 가중치 <b>{w_preset['w_defense']:.2f}</b>) "
        f"=&nbsp;<strong style='color:#111;font-size:16px;'>{ai_score:.1f}점</strong>"
        f"</span></div>",
        unsafe_allow_html=True)

# ============================================================
# 14. 푸터
# ============================================================
st.divider()
st.caption("⚠️ 본 대시보드는 해커톤 시연 목적으로 제작되었습니다. 실제 투자 권유 또는 자문이 아닙니다.")
