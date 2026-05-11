"""
SurviQuant — 생존분석 기반 S&P500 AI 투자 대시보드
==========================================================
실행: streamlit run dashboard.py
데이터: ./data/scores.json, ./data/ohlcv_cache.parquet
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="SurviQuant | 생존분석 기반 AI 투자 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner="📡 AI Score 데이터 로딩 중...")
def load_scores():
    with open(DATA_DIR / "scores.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["meta"], pd.DataFrame(payload["scores"])

@st.cache_data(show_spinner="📊 OHLCV·기술지표 로딩 중...")
def load_ohlcv():
    df = pd.read_parquet(DATA_DIR / "ohlcv_cache.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

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

def build_chart(ticker_df, period_days):
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

meta, scores_df = load_scores()
ohlcv_df = load_ohlcv()

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
    st.caption(f"🎯 분석 종목: **{meta['universe']['modeled_tickers']}개** (5섹터 대표주)")

scores_df = scores_df.copy()
scores_df["AI_Score"] = compute_ai_score(scores_df["Profit_Chance"], scores_df["Loss_Risk"], persona_key).round(2)
scores_df[["Signal", "SignalColor"]] = scores_df["AI_Score"].apply(lambda s: pd.Series(classify_signal(s)))
scores_df = scores_df.sort_values("AI_Score", ascending=False).reset_index(drop=True)

st.title("📈 SurviQuant")
st.markdown("**생존분석 기반 S&P 500 AI 투자 대시보드** — 단순 방향성이 아닌 *'도달 확률 + 시점'* 을 정량 제공")

tab1, tab2, tab3 = st.tabs(["🏆 Today's Top Picks", "📊 종목 상세 차트", "ℹ️ About / 방법론"])

with tab1:
    sector_options = ["전체"] + sorted(scores_df["Sector"].unique().tolist())
    col_filter, col_persona = st.columns([3, 1])
    with col_filter:
        sel_sector = st.selectbox("🏭 섹터 필터", options=sector_options, help="섹터별로 Top10을 따로 보려면 선택하세요.")
    with col_persona:
        st.metric("선택된 성향", PERSONA_PRESETS[persona_key]["label"])
    filtered = scores_df if sel_sector == "전체" else scores_df[scores_df["Sector"] == sel_sector].reset_index(drop=True)

    st.markdown("#### 📍 시그널 분포")
    sig_counts = filtered["Signal"].value_counts()
    chip_cols = st.columns(4)
    for i, (label, threshold, color) in enumerate(SIGNAL_RULES):
        with chip_cols[i]:
            count = int(sig_counts.get(label, 0))
            st.markdown(f"""<div style="background:{color}15;border-left:5px solid {color};padding:14px 18px;border-radius:8px;min-height:88px;"><div style="font-size:13px;color:#555;font-weight:500;">{label}</div><div style="font-size:30px;font-weight:700;color:{color};line-height:1.2;">{count}<span style="font-size:14px;color:#999;font-weight:400;"> 종목</span></div><div style="font-size:11px;color:#999;">{"≥ " + str(threshold) + "점" if threshold > 0 else "< 40점"}</div></div>""", unsafe_allow_html=True)

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
            st.markdown(f"""<div style="margin-top:14px;"><span style="background:{color};color:white;padding:8px 18px;border-radius:22px;font-weight:600;font-size:14px;">{row['Signal']}</span></div>""", unsafe_allow_html=True)
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
            column_config={"AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.1f"),
                           "수익 확률(%)": st.column_config.NumberColumn(format="%.1f%%"),
                           "손실 확률(%)": st.column_config.NumberColumn(format="%.1f%%")})

with tab2:
    col_tick, col_period = st.columns([2, 2])
    with col_tick:
        sel_ticker = st.selectbox("🔍 종목 선택", options=scores_df["Ticker"].tolist(), help="AI Score 내림차순 정렬")
    with col_period:
        sel_period_label = st.selectbox("📅 조회 기간", options=list(PERIOD_OPTIONS.keys()), index=1)
    period_days = PERIOD_OPTIONS[sel_period_label]
    row = scores_df[scores_df["Ticker"] == sel_ticker].iloc[0]
    ticker_ohlcv = ohlcv_df[ohlcv_df["Ticker"] == sel_ticker]
    w = PERSONA_PRESETS[persona_key]
    contrib_profit  = round(row["Profit_Chance"] * w["w_profit"], 2)
    contrib_defense = round((100 - row["Loss_Risk"]) * w["w_defense"], 2)
    ai_score        = round(contrib_profit + contrib_defense, 2)
    signal_label, signal_color = classify_signal(ai_score)

    st.markdown("---")
    st.markdown(f"### {sel_ticker} &nbsp;<span style='background:{signal_color};color:white;padding:5px 16px;border-radius:20px;font-size:15px;font-weight:600;'>{signal_label}</span>", unsafe_allow_html=True)
    st.caption(f"📂 {row['Sector']}  |  성향: {PERSONA_PRESETS[persona_key]['label']}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🏆 AI Score", f"{ai_score:.1f} / 100", help="수익 기여 + 방어 기여의 합산 점수")
    with c2:
        st.metric("📈 수익 기여", f"{contrib_profit:.1f}점", f"Profit {row['Profit_Chance']:.1f}% × {w['w_profit']:.2f}")
    with c3:
        st.metric("🛡 방어 기여", f"{contrib_defense:.1f}점", f"Defense {100 - row['Loss_Risk']:.1f}% × {w['w_defense']:.2f}")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=[contrib_profit], y=["AI Score"], orientation="h", name="수익 기여",
        marker_color="#10B981", text=[f"수익 기여 {contrib_profit:.1f}점"],
        textposition="inside", textfont=dict(color="white", size=13)))
    fig_bar.add_trace(go.Bar(x=[contrib_defense], y=["AI Score"], orientation="h", name="방어 기여",
        marker_color="#3B82F6", text=[f"방어 기여 {contrib_defense:.1f}점"],
        textposition="inside", textfont=dict(color="white", size=13)))
    fig_bar.update_layout(barmode="stack", height=80, margin=dict(l=10, r=10, t=8, b=8),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", y=1.6, x=0), template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📖 차트 지표 설명 (클릭하면 펼쳐집니다)"):
        for desc in INDICATOR_HELP.values():
            st.markdown(f"- {desc}")

    st.markdown("---")
    fig_chart = build_chart(ticker_ohlcv, period_days)
    st.plotly_chart(fig_chart, use_container_width=True)

    latest = ticker_ohlcv.sort_values("Date").iloc[-1]
    st.markdown("##### 📌 최근 지표 스냅샷")
    snap_cols = st.columns(5)
    for col, (label, value, tip) in zip(snap_cols, [
        ("종가",       f"${latest['Close']:.2f}",    None),
        ("RSI",        f"{latest['RSI']:.1f}",        "70↑ 과매수 / 30↓ 과매도"),
        ("ADX",        f"{latest['ADX']:.1f}",        "25 이상이면 추세 존재"),
        ("Volatility", f"{latest['Volatility']:.3f}", "높을수록 변동성 큼"),
        ("MA20",       f"${latest['MA20']:.2f}",      "20일 이동평균선"),
    ]):
        with col:
            st.metric(label, value, help=tip)

with tab3:
    st.markdown("## ℹ️ SurviQuant — 프로젝트 소개 및 방법론")
    st.divider()

    st.markdown("### 🎯 왜 생존분석인가?")
    st.markdown("""
일반적인 주가 예측 모델은 **방향성(상승/하락)**만 예측합니다.  
SurviQuant는 한 걸음 더 나아가 **'언제 도달하는가'** 를 함께 예측합니다.

- 📈 **수익 사건**: 매수 시점 대비 **+10% 도달**까지의 시간
- 📉 **손실 사건**: 매수 시점 대비 **−10% 도달**까지의 시간
- ⏱ **예측 Horizon**: 고정 **20영업일** (약 1개월)

> 확률만 아는 것과, 언제 그 확률이 실현될지 아는 것은 전혀 다른 정보입니다.
    """)

    st.divider()
    st.markdown("### 🧮 AI Score 산출 공식")
    st.code("""
AI Score = Profit_Chance × w_profit + (100 − Loss_Risk) × w_defense

# 성향별 가중치 (λ = Tversky·Kahneman 손실 회피 계수)
안정형: w_profit=0.31, w_defense=0.69  (λ=2.25)
중립형: w_profit=0.50, w_defense=0.50  (λ=1.00)
공격형: w_profit=0.69, w_defense=0.31  (λ=0.44)
    """, language="python")

    st.divider()
    st.markdown("### 📐 λ=2.25 학술 근거")
    st.markdown("""
안정형 성향의 가중치 비율(31:69)은 행동경제학의 핵심 이론에서 가져왔습니다.

- **Tversky & Kahneman (1992)** — Prospect Theory에서 실증된 손실 회피 계수 **λ ≈ 2.25**
- 동일 금액의 손실이 이익보다 심리적으로 **2.25배 강하게 느껴진다**는 것을 의미
- 중립형(λ=1.0)은 손익을 동등하게, 공격형(λ=0.44)은 수익을 더 중시

**금융투자협회 투자자 성향 분류 준칙**의 3단계 구조(안정·중립·공격)와도 일치합니다.
    """)

    st.divider()
    st.markdown("### 📊 모델 성능")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("수익 모델 C-index", "0.7087",
                  help="C-index: 0.5=무작위, 1.0=완벽. 0.7 이상은 임상/금융 분야에서 실용적 수준으로 평가됩니다.")
    with m2:
        st.metric("손실 모델 C-index", "0.7681",
                  help="손실 모델이 수익 모델보다 높은 것은 Volatility(HR 5.28)의 강력한 손실 예측력 덕분입니다.")
    st.caption("모델: Random Survival Forest (RSF) | 검증: Kaplan-Meier + Cox PH | 핵심 변수: Volatility (수익 HR 2.87 / 손실 HR 5.28)")

    st.divider()
    st.markdown("### ⚠️ 데이터 범위 및 한계")
    st.warning("""
**현재 버전의 분석 대상은 50종목으로 제한됩니다.**

- 분석 기간: 2024.04 ~ 2026.04 (약 502영업일)  
- 대상: S&P 500 5개 섹터 대표 50종목  
- EDA·생존 레코드: 223종목 전체 활용 / RSF 학습·추론: 50종목  
- 거시 지표: 10년물 미국채 수익률(TNX) 통합  

본 도구는 **참고용 정량 지표**이며, 투자 권유 또는 자문이 아닙니다.
    """)
