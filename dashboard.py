
"""
Aegis Discord Admin Dashboard  ·  Pure Python / Streamlit
Run: streamlit run dashboard.py
Deps: streamlit pandas sqlalchemy psycopg2-binary plotly fpdf2 python-dotenv
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
from sqlalchemy import create_engine
from dotenv import load_dotenv
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG & DISCORD THEME
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aegis | Discord Admin UI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCORD_CSS = """
<style>
/* ── Base ── */
.stApp                          { background-color: #0e1117; color: #e6edf3; }
section[data-testid="stSidebar"]{ background-color: #161b22 !important; border-right: 1px solid #2a3244; }
.block-container                { padding-top: 1.4rem; }

/* ── Metric cards ── */
.aegis-card {
    background: #1c2230;
    border-left: 4px solid var(--accent, #5865F2);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 4px;
}
.aegis-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
}
.aegis-value {
    font-size: 2.1rem;
    font-weight: 800;
    margin-top: 4px;
    font-family: 'Courier New', monospace;
}
.aegis-sub {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 3px;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b949e;
    font-weight: 700;
    border-bottom: 1px solid #2a3244;
    padding-bottom: 6px;
    margin: 18px 0 12px;
    font-family: 'Courier New', monospace;
}

/* ── Score badge ── */
.badge-toxic  { background:#ed424522; color:#ed4245; border:1px solid #ed424544; border-radius:5px; padding:2px 8px; font-size:.72rem; font-weight:700; }
.badge-watch  { background:#fee75c22; color:#fee75c; border:1px solid #fee75c44; border-radius:5px; padding:2px 8px; font-size:.72rem; font-weight:700; }
.badge-clean  { background:#57f28722; color:#57f287; border:1px solid #57f28744; border-radius:5px; padding:2px 8px; font-size:.72rem; font-weight:700; }

/* ── Intervention alert box ── */
.intervention-box {
    background: #ed424511;
    border: 1px solid #ed424544;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #c9d1d9;
}

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"]   { background: #161b22; border-bottom: 1px solid #2a3244; }
.stTabs [data-baseweb="tab"]        { color: #8b949e; font-size: 0.82rem; }
.stTabs [aria-selected="true"]      { color: #e6edf3 !important; border-bottom: 2px solid #5865F2 !important; }
div[data-testid="stMetric"]         { background: #1c2230; border-radius: 10px; padding: 14px 18px; }
</style>
"""
st.markdown(DISCORD_CSS, unsafe_allow_html=True)

# Plotly shared dark layout
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
    margin=dict(l=12, r=12, t=28, b=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATABASE  (cached connection + data load)
# ══════════════════════════════════════════════════════════════════════════════
load_dotenv()
DATABASE_URL = os.getenv(
    "SUPABASE_URL",
    "postgresql+psycopg2://postgres:qpDt2IqS9mbjgBgD@db.spibrcrelsmdfvpurnac.supabase.co:5432/postgres"
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)


@st.cache_resource(show_spinner="Connecting to Supabase…")
def get_engine():
    return create_engine(DATABASE_URL)


@st.cache_data(ttl=30, show_spinner="Fetching latest data…")
def load_data():
    engine = get_engine()
    df_m = pd.read_sql("SELECT * FROM messages ORDER BY timestamp ASC", engine)
    df_u = pd.read_sql("SELECT * FROM users ORDER BY trust_score ASC", engine)
    return df_m, df_u


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SPC COMPUTATION  (mirrors the live bot logic exactly)
# ══════════════════════════════════════════════════════════════════════════════
def compute_spc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["moving_avg"]  = df["toxicity_score"].rolling(window=5, min_periods=1).mean()
    df["rolling_std"] = df["toxicity_score"].rolling(window=5, min_periods=1).std()
    df["adjustment"]  = df["rolling_std"].apply(lambda x: 0.20 * x if pd.notna(x) else 0.02)
    df["dynamic_ucl"] = df["adjustment"].apply(lambda x: min(0.55, 0.40 + x))
    df["spike"]       = df["moving_avg"] >= df["dynamic_ucl"]
    return df


def risk_label(score: float) -> str:
    if score > 0.6:  return "🔴 HIGH RISK"
    if score > 0.35: return "🟡 WATCH"
    return "🟢 SAFE"


def trust_label(trust: float) -> str:
    if trust < 40:  return "🔴 High Risk"
    if trust < 70:  return "🟡 On Watch"
    return "🟢 Safe"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PDF REPORT  (academic compliance export)
# ══════════════════════════════════════════════════════════════════════════════
class AcademicReport(FPDF):
    def header(self):
        self.set_font("Times", "B", 11)
        self.set_text_color(88, 101, 242)
        self.cell(0, 9, "AEGIS DISCORD MODERATION SYSTEM - ACADEMIC COMPLIANCE REPORT", align="C")
        self.ln(13)

    def footer(self):
        self.set_y(-14)
        self.set_font("Times", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 9, f"Page {self.page_no()} | Generated by Aegis Bot Platform - {datetime.now():%Y-%m-%d %H:%M}", align="C")


def generate_pdf(df_msgs: pd.DataFrame, df_users: pd.DataFrame) -> bytes:
    df = compute_spc(df_msgs)
    total = len(df)
    avg_tox = df["toxicity_score"].mean() if total else 0
    spikes = int(df["spike"].sum())
    top5_channels = df.groupby("channel_id")["toxicity_score"].mean().sort_values(ascending=False).head(5)

    pdf = AcademicReport()
    pdf.add_page()

    # ── Cover ──────────────────────────────────────────────────────────────
    pdf.set_font("Times", "B", 18)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 11, "Monthly Discord Server Health & NLP Toxicity Audit", ln=True)
    pdf.set_font("Times", "I", 11)
    pdf.cell(0, 9, f"Reporting Period: {datetime.now():%B %Y}", ln=True)
    pdf.ln(4)

    # ── 1. Executive Metrics ───────────────────────────────────────────────
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 9, "1. Executive Server Metrics", ln=True)
    pdf.set_font("Times", "", 12)
    pdf.multi_cell(0, 7,
        f"During this monitoring cycle the Aegis NLP Engine parsed {total:,} active Discord "
        f"messages. The server's baseline toxicity moving average is {avg_tox:.4f}. "
        f"Statistical anomalies tracked above the dynamic SPC control limits resulted in "
        f"{spikes} intervention events."
    )
    pdf.ln(6)

    # ── 2. SPC Summary table ───────────────────────────────────────────────
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 9, "2. SPC Metric Summary", ln=True)
    pdf.set_font("Times", "B", 11)
    pdf.set_fill_color(220, 220, 220)
    for header in ("Metric", "Value"):
        pdf.cell(95, 8, header, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Times", "", 11)
    rows = [
        ("Messages Scanned",          f"{total:,}"),
        ("Average Toxicity Score",    f"{avg_tox:.4f}"),
        ("Dynamic UCL Breaches",      str(spikes)),
        ("Intervention Rate",         f"{spikes/max(total,1)*100:.1f}%"),
        ("Clean Messages (<0.2)",     f"{(df['toxicity_score']<0.2).sum():,}"),
        ("Toxic Messages (>0.6)",     f"{(df['toxicity_score']>0.6).sum():,}"),
    ]
    for label, val in rows:
        pdf.cell(95, 8, label, border=1)
        pdf.cell(95, 8, val,   border=1)
        pdf.ln()
    pdf.ln(6)

    # ── 3. Top Offenders ───────────────────────────────────────────────────
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 9, "3. Top Offenders - Highest Risk Profiles", ln=True)
    pdf.set_font("Times", "B", 11)
    pdf.set_fill_color(220, 220, 220)
    for hdr in ("Discord User ID", "Trust Score", "Risk Status"):
        pdf.cell(63, 8, hdr, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Times", "", 11)
    for _, row in df_users.sort_values("trust_score").head(10).iterrows():
        t = row["trust_score"]
        status = "High Risk" if t < 40 else ("On Watch" if t < 70 else "Safe")
        pdf.cell(63, 8, str(row["user_id"]), border=1)
        pdf.cell(63, 8, f"{t:.2f}",          border=1)
        pdf.cell(63, 8, status,               border=1)
        pdf.ln()
    pdf.ln(6)

    # ── 4. Hotspot Channels ────────────────────────────────────────────────
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 9, "4. Highest-Toxicity Channels", ln=True)
    pdf.set_font("Times", "B", 11)
    pdf.set_fill_color(220, 220, 220)
    for hdr in ("Channel ID", "Avg Toxicity Score"):
        pdf.cell(95, 8, hdr, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Times", "", 11)
    for ch_id, avg in top5_channels.items():
        pdf.cell(95, 8, str(ch_id), border=1)
        pdf.cell(95, 8, f"{avg:.4f}",  border=1)
        pdf.ln()

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════
try:
    df_messages_raw, df_users = load_data()
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.stop()

df_messages = compute_spc(df_messages_raw)

# Derived aggregates
total_msgs   = len(df_messages)
avg_toxicity = df_messages["toxicity_score"].mean() if total_msgs else 0
spikes_total = int(df_messages["spike"].sum())
toxic_count  = int((df_messages["toxicity_score"] > 0.6).sum())
clean_count  = int((df_messages["toxicity_score"] < 0.2).sum())
watch_count  = total_msgs - toxic_count - clean_count
health_pct   = round((1 - toxic_count / max(total_msgs, 1)) * 100, 1)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Aegis")
    st.caption("Autonomous Discord Moderation Agent")
    st.divider()

    # ── Live status chip ──
    live_color = "#57f287" if health_pct >= 75 else "#fee75c" if health_pct >= 50 else "#ed4245"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{live_color};display:inline-block;'></span>"
        f"<span style='font-size:.82rem;color:#c9d1d9;'>Server health: <b>{health_pct}%</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Refresh ──
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("<div class='section-header'>📥 PDF EXPORT</div>", unsafe_allow_html=True)

    if total_msgs > 0:
        pdf_bytes = generate_pdf(df_messages_raw, df_users)
        st.download_button(
            label="📄 Export Academic Report (PDF)",
            data=pdf_bytes,
            file_name=f"Aegis_Report_{datetime.now():%Y-%m-%d}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("No data — report unavailable.")

    st.markdown("<div class='section-header'>⚙️ FILTERS</div>", unsafe_allow_html=True)
    score_threshold = st.slider("Min toxicity score filter", 0.0, 1.0, 0.0, 0.01)
    show_spikes_only = st.checkbox("Show SPC spike rows only", value=False)
    max_rows = st.number_input("Max rows in tables", min_value=10, max_value=500, value=50, step=10)

    st.markdown("<div class='section-header'>ℹ️ SPC PARAMETERS</div>", unsafe_allow_html=True)
    st.caption("Window = 5 messages")
    st.caption("UCL = min(0.55, 0.40 + 0.20 × σ)")
    st.caption("Intervention cooldown = 60 s")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='color:#e6edf3;font-weight:800;border-bottom:2px solid #5865F2;"
    "padding-bottom:10px;font-family:Courier New,monospace;'>🛡️ Aegis · Discord Administrator UI</h1>",
    unsafe_allow_html=True,
)
st.caption("Real-time PostgreSQL telemetry · Statistical Process Control · NLP Toxicity Engine")

if total_msgs == 0:
    st.info("📡 Database is empty. Send some messages in your Discord server first.")
    st.stop()

# ── Apply sidebar filters to views ──
df_filtered = df_messages[df_messages["toxicity_score"] >= score_threshold]
if show_spikes_only:
    df_filtered = df_filtered[df_filtered["spike"]]

# ══════════════════════════════════════════════════════════════════════════════
# 8.  METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
def metric_card(label, value, sub="", accent="#5865F2"):
    return (
        f"<div class='aegis-card' style='--accent:{accent};'>"
        f"<div class='aegis-label'>{label}</div>"
        f"<div class='aegis-value' style='color:{accent};'>{value}</div>"
        f"<div class='aegis-sub'>{sub}</div>"
        f"</div>"
    )


c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(metric_card("Messages Scanned", f"{total_msgs:,}", "All-time total"), unsafe_allow_html=True)
c2.markdown(metric_card("Avg Toxicity",     f"{avg_toxicity:.4f}", "Rolling mean", "#fee75c" if avg_toxicity > 0.3 else "#57f287"), unsafe_allow_html=True)
c3.markdown(metric_card("Spikes Mitigated", str(spikes_total), "SPC UCL breaches", "#ed4245"), unsafe_allow_html=True)
c4.markdown(metric_card("Toxic Messages",  str(toxic_count), "Score > 0.60", "#ed4245"), unsafe_allow_html=True)
c5.markdown(metric_card("Server Health",   f"{health_pct}%", "Clean message ratio", "#57f287" if health_pct >= 75 else "#fee75c"), unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_spc, tab_dist, tab_heatmap, tab_channels, tab_trust, tab_feed, tab_interventions = st.tabs([
    "📈 SPC Timeline",
    "📊 Score Distribution",
    "🔥 Channel Heatmap",
    "📡 Channel Stats",
    "🏆 Trust Scores",
    "📝 Message Feed",
    "⚠️ Interventions",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 · SPC TIMELINE
# ──────────────────────────────────────────────────────────────────────────────
with tab_spc:
    st.markdown("<div class='section-header'>STATISTICAL PROCESS CONTROL · LIVE SERVER HEALTH</div>", unsafe_allow_html=True)

    # FIX: Use the actual current dynamic UCL from the last data point
    current_ucl = float(df_messages["dynamic_ucl"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp"], y=df_filtered["toxicity_score"],
        mode="markers", name="Raw Score",
        marker=dict(color="rgba(139,148,158,0.35)", size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp"], y=df_filtered["moving_avg"],
        mode="lines", name="5-Msg Moving Avg",
        line=dict(color="#5865F2", width=2.5),
        fill="tozeroy", fillcolor="rgba(88,101,242,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp"], y=df_filtered["dynamic_ucl"],
        mode="lines", name="Dynamic UCL",
        line=dict(color="#ed4245", width=2, dash="dash"),
    ))
    # Spike markers
    spikes_df = df_filtered[df_filtered["spike"]]
    if not spikes_df.empty:
        fig.add_trace(go.Scatter(
            x=spikes_df["timestamp"], y=spikes_df["moving_avg"],
            mode="markers", name="⚠ Spike Event",
            marker=dict(color="#ed4245", size=10, symbol="x"),
        ))
    # FIX: Reference line now uses the actual computed current UCL value
    fig.add_hline(
        y=current_ucl,
        line_dash="dot",
        line_color="#fee75c",
        annotation_text=f"Current UCL {current_ucl:.4f}",
        annotation_position="bottom right",
        annotation_font_color="#fee75c",
    )
    fig.update_layout(
        **PLOTLY_DARK,
        height=340,
        xaxis=dict(showgrid=False, title="Timestamp"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   title="Toxicity Confidence", range=[0, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling stats mini table
    recent = df_messages.tail(5)
    r_avg = recent["toxicity_score"].mean()
    r_std = recent["toxicity_score"].std()
    r_ucl = recent["dynamic_ucl"].iloc[-1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Last 5-Msg Moving Avg", f"{r_avg:.4f}")
    m2.metric("Rolling Std Dev",       f"{r_std:.4f}" if pd.notna(r_std) else "—")
    m3.metric("Current Dynamic UCL",   f"{r_ucl:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 · SCORE DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────
with tab_dist:
    st.markdown("<div class='section-header'>TOXICITY SCORE DISTRIBUTION</div>", unsafe_allow_html=True)

    col_hist, col_bar = st.columns(2)

    with col_hist:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=df_filtered["toxicity_score"],
            nbinsx=40,
            marker_color="#5865F2",
            opacity=0.85,
            name="Score frequency",
        ))
        fig_h.add_vline(x=0.40, line_dash="dash", line_color="#fee75c",
                        annotation_text="Base UCL", annotation_font_color="#fee75c")
        fig_h.add_vline(x=0.60, line_dash="dash", line_color="#ed4245",
                        annotation_text="Toxic threshold", annotation_font_color="#ed4245")
        fig_h.update_layout(
            **PLOTLY_DARK, height=300, bargap=0.05,
            xaxis_title="Toxicity Score", yaxis_title="Message Count",
            title=dict(text="Score Frequency Histogram", font_color="#8b949e", font_size=13),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_bar:
        buckets = {
            "0–0.20  Clean":   int((df_filtered["toxicity_score"] < 0.20).sum()),
            "0.20–0.40 Mild":  int(((df_filtered["toxicity_score"] >= 0.20) & (df_filtered["toxicity_score"] < 0.40)).sum()),
            "0.40–0.60 Watch": int(((df_filtered["toxicity_score"] >= 0.40) & (df_filtered["toxicity_score"] < 0.60)).sum()),
            "0.60–0.80 High":  int(((df_filtered["toxicity_score"] >= 0.60) & (df_filtered["toxicity_score"] < 0.80)).sum()),
            "0.80–1.00 Severe":int((df_filtered["toxicity_score"] >= 0.80).sum()),
        }
        # FIX: replaced invalid 8-digit hex "#ed4245cc" with valid rgba() equivalent
        colors = ["#57f287", "#fee75c", "#f0a500", "rgba(237,66,69,0.8)", "#ed4245"]
        fig_b = go.Figure(go.Bar(
            x=list(buckets.keys()),
            y=list(buckets.values()),
            marker_color=colors,
            text=list(buckets.values()),
            textposition="outside",
            textfont_color="#c9d1d9",
        ))
        fig_b.update_layout(
            **PLOTLY_DARK, height=300,
            xaxis=dict(tickangle=-20),
            yaxis_title="Count",
            title=dict(text="Messages by Risk Bucket", font_color="#8b949e", font_size=13),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # Donut
    fig_d = go.Figure(go.Pie(
        labels=["Clean (<0.2)", "Watch (0.2–0.6)", "Toxic (>0.6)"],
        values=[clean_count, watch_count, toxic_count],
        hole=0.55,
        marker_colors=["#57f287", "#fee75c", "#ed4245"],
        textinfo="label+percent",
        textfont_color="#e6edf3",
    ))
    fig_d.update_layout(**PLOTLY_DARK, height=280,
                        title=dict(text="Message Risk Breakdown", font_color="#8b949e", font_size=13))
    st.plotly_chart(fig_d, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 · CHANNEL HEATMAP  (hour-of-day × channel)
# ──────────────────────────────────────────────────────────────────────────────
with tab_heatmap:
    st.markdown("<div class='section-header'>TOXICITY HOTSPOT HEATMAP · CHANNELS × HOUR OF DAY</div>", unsafe_allow_html=True)
    st.caption("Each cell = average toxicity score for that channel in that hour. Darker red = more toxic activity.")

    df_h = df_messages.copy()
    df_h["hour"] = df_h["timestamp"].dt.hour

    heatmap_pivot = (
        df_h.groupby(["channel_id", "hour"])["toxicity_score"]
        .mean()
        .unstack(fill_value=0)
    )
    # Ensure all 24 hours present
    for hr in range(24):
        if hr not in heatmap_pivot.columns:
            heatmap_pivot[hr] = 0
    heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]

    fig_heat = go.Figure(go.Heatmap(
    z=heatmap_pivot.values,
    x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
    y=[str(ch) for ch in heatmap_pivot.index],
    colorscale=[
        [0.0, "rgb(28,34,48)"],
        [0.4, "rgba(254,231,92,0.2)"],
        [0.7, "rgba(237,66,69,0.53)"],
        [1.0, "rgb(237,66,69)"]
    ],
    showscale=True,
        colorbar=dict(
    title=dict(
        text="Avg Score",
        font=dict(color="#8b949e")
    ),
    tickfont=dict(color="#8b949e")
),
        hovertemplate="Channel: %{y}<br>Hour: %{x}<br>Avg Toxicity: %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        **PLOTLY_DARK, height=max(260, 55 * len(heatmap_pivot)),
        xaxis=dict(side="bottom", title="Hour of Day"),
        yaxis=dict(title="Channel ID"),
        title=dict(text="Hourly Toxicity Density per Channel", font_color="#8b949e", font_size=13),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 · CHANNEL STATS
# ──────────────────────────────────────────────────────────────────────────────
with tab_channels:
    st.markdown("<div class='section-header'>PER-CHANNEL TOXICITY BREAKDOWN</div>", unsafe_allow_html=True)

    ch_stats = (
        df_messages.groupby("channel_id")
        .agg(
            total_msgs=("toxicity_score", "count"),
            avg_score=("toxicity_score", "mean"),
            max_score=("toxicity_score", "max"),
            toxic_msgs=("toxicity_score", lambda x: (x > 0.6).sum()),
            spikes=("spike", "sum"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    ch_stats["toxic_rate"] = (ch_stats["toxic_msgs"] / ch_stats["total_msgs"] * 100).round(1)

    col_vol, col_avg = st.columns(2)

    with col_vol:
        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(
            name="Total Messages", x=ch_stats["channel_id"].astype(str),
            y=ch_stats["total_msgs"], marker_color="#5865F2",
        ))
        fig_v.add_trace(go.Bar(
            name="Toxic (>0.6)", x=ch_stats["channel_id"].astype(str),
            y=ch_stats["toxic_msgs"], marker_color="#ed4245",
        ))
        fig_v.update_layout(**PLOTLY_DARK, height=300, barmode="group",
                            title=dict(text="Message Volume per Channel", font_color="#8b949e", font_size=13))
        st.plotly_chart(fig_v, use_container_width=True)

    with col_avg:
        fig_a = go.Figure(go.Bar(
            x=ch_stats["channel_id"].astype(str),
            y=ch_stats["avg_score"].round(4),
            marker_color=["#ed4245" if s > 0.4 else "#5865F2" for s in ch_stats["avg_score"]],
            text=ch_stats["avg_score"].round(4),
            textposition="outside",
            textfont_color="#c9d1d9",
        ))
        fig_a.add_hline(y=0.40, line_dash="dash", line_color="#fee75c",
                        annotation_text="Base UCL", annotation_font_color="#fee75c")
        fig_a.update_layout(**PLOTLY_DARK, height=300,
                            yaxis=dict(range=[0, max(0.7, ch_stats["avg_score"].max() + 0.05)]),
                            title=dict(text="Average Toxicity per Channel", font_color="#8b949e", font_size=13))
        st.plotly_chart(fig_a, use_container_width=True)

    # Summary table
    st.markdown("<div class='section-header'>CHANNEL SUMMARY TABLE</div>", unsafe_allow_html=True)
    display_ch = ch_stats.rename(columns={
        "channel_id": "Channel ID", "total_msgs": "Total Messages",
        "avg_score": "Avg Score", "max_score": "Max Score",
        "toxic_msgs": "Toxic Messages", "spikes": "SPC Spikes",
        "toxic_rate": "Toxic Rate (%)",
    })
    st.dataframe(
        display_ch.style
        .format({"Avg Score": "{:.4f}", "Max Score": "{:.4f}", "Toxic Rate (%)": "{:.1f}"})
        .background_gradient(subset=["Avg Score"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=280,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 · TRUST SCORES
# ──────────────────────────────────────────────────────────────────────────────
with tab_trust:
    st.markdown("<div class='section-header'>USER TRUST SCORE LEADERBOARD · RISK PROFILE RANKING</div>", unsafe_allow_html=True)

    risk_counts = {
        "High Risk (<40)":  int((df_users["trust_score"] < 40).sum()),
        "On Watch (40–70)": int(((df_users["trust_score"] >= 40) & (df_users["trust_score"] < 70)).sum()),
        "Safe (>70)":       int((df_users["trust_score"] >= 70).sum()),
    }
    r1, r2, r3 = st.columns(3)
    r1.markdown(metric_card("High Risk Users", str(risk_counts["High Risk (<40)"]),   "Trust < 40",  "#ed4245"), unsafe_allow_html=True)
    r2.markdown(metric_card("On Watch",         str(risk_counts["On Watch (40–70)"]), "Trust 40–70", "#fee75c"), unsafe_allow_html=True)
    r3.markdown(metric_card("Safe Members",     str(risk_counts["Safe (>70)"]),        "Trust > 70",  "#57f287"), unsafe_allow_html=True)

    st.markdown("")

    # Trust bar chart
    df_trust_plot = df_users.sort_values("trust_score").head(int(max_rows))
    colors_trust = ["#ed4245" if t < 40 else "#fee75c" if t < 70 else "#57f287"
                    for t in df_trust_plot["trust_score"]]
    fig_t = go.Figure(go.Bar(
        x=df_trust_plot["trust_score"].round(1),
        y=df_trust_plot["user_id"].astype(str),
        orientation="h",
        marker_color=colors_trust,
        text=df_trust_plot["trust_score"].round(1),
        textposition="outside",
        textfont_color="#c9d1d9",
    ))
    fig_t.add_vline(x=40, line_dash="dash", line_color="#ed4245",
                    annotation_text="High Risk", annotation_font_color="#ed4245")
    fig_t.add_vline(x=70, line_dash="dash", line_color="#fee75c",
                    annotation_text="Watch",     annotation_font_color="#fee75c")
    fig_t.update_layout(
        **PLOTLY_DARK,
        height=max(300, len(df_trust_plot) * 28),
        xaxis=dict(range=[0, 110], title="Trust Score"),
        yaxis=dict(title=""),
        title=dict(text="Trust Score per User (lowest → highest risk)", font_color="#8b949e", font_size=13),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # Table
    display_trust = df_users.copy()
    display_trust["Risk Status"] = display_trust["trust_score"].apply(trust_label)
    display_trust = display_trust.rename(columns={"user_id": "Discord User ID", "trust_score": "Trust Score", "last_active": "Last Active"})
    st.dataframe(
        display_trust[["Discord User ID", "Trust Score", "Risk Status", "Last Active"]]
        .sort_values("Trust Score")
        .head(int(max_rows))
        .style.format({"Trust Score": "{:.2f}"}),
        use_container_width=True,
        height=320,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 · MESSAGE FEED
# ──────────────────────────────────────────────────────────────────────────────
with tab_feed:
    st.markdown("<div class='section-header'>LIVE MESSAGE INTERCEPTS</div>", unsafe_allow_html=True)

    feed_filter = st.radio(
        "Filter by risk tier:",
        ["All", "Toxic (>0.6)", "Watch (0.2–0.6)", "Clean (<0.2)"],
        horizontal=True,
    )

    df_feed = df_filtered.copy()
    if feed_filter == "Toxic (>0.6)":
        df_feed = df_feed[df_feed["toxicity_score"] > 0.6]
    elif feed_filter == "Watch (0.2–0.6)":
        df_feed = df_feed[(df_feed["toxicity_score"] >= 0.2) & (df_feed["toxicity_score"] <= 0.6)]
    elif feed_filter == "Clean (<0.2)":
        df_feed = df_feed[df_feed["toxicity_score"] < 0.2]

    df_feed = df_feed.sort_values("timestamp", ascending=False).head(int(max_rows))
    df_feed["Risk"] = df_feed["toxicity_score"].apply(risk_label)

    display_feed = df_feed[["timestamp", "channel_id", "user_id", "toxicity_score", "Risk", "content"]].copy()
    display_feed.columns = ["Timestamp", "Channel", "User", "Score", "Risk", "Message"]

    st.dataframe(
        display_feed.style
        .format({"Score": "{:.4f}"})
        .background_gradient(subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
        height=480,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 7 · INTERVENTIONS LOG
# ──────────────────────────────────────────────────────────────────────────────
with tab_interventions:
    st.markdown("<div class='section-header'>AEGIS RAG INTERVENTION LOG · SPC UCL BREACH EVENTS</div>", unsafe_allow_html=True)

    df_spikes = df_messages[df_messages["spike"]].sort_values("timestamp", ascending=False)

    i1, i2, i3 = st.columns(3)
    i1.markdown(metric_card("Total Interventions", str(spikes_total), "SPC UCL breaches", "#ed4245"), unsafe_allow_html=True)
    i2.markdown(metric_card("Intervention Rate",   f"{spikes_total/max(total_msgs,1)*100:.1f}%", "Of all messages", "#fee75c"), unsafe_allow_html=True)
    i3.markdown(metric_card("Last Spike Channel",  str(df_spikes["channel_id"].iloc[0]) if not df_spikes.empty else "—", "Most recent breach", "#5865F2"), unsafe_allow_html=True)

    st.markdown("")

    if df_spikes.empty:
        st.success("✅ **All Clear** — No SPC spike events detected in the current dataset.")
    else:
        # Timeline of interventions
        fig_int = go.Figure()
        fig_int.add_trace(go.Scatter(
            x=df_messages["timestamp"], y=df_messages["moving_avg"],
            mode="lines", name="Moving Avg",
            line=dict(color="#5865F2", width=2),
        ))
        fig_int.add_trace(go.Scatter(
            x=df_spikes["timestamp"], y=df_spikes["moving_avg"],
            mode="markers", name="Intervention",
            marker=dict(color="#ed4245", size=11, symbol="x-open-dot", line_width=2),
        ))
        fig_int.update_layout(
            **PLOTLY_DARK, height=240,
            xaxis_title="Timestamp", yaxis_title="Moving Avg",
            title=dict(text="Intervention Events on Moving Average Timeline", font_color="#8b949e", font_size=13),
        )
        st.plotly_chart(fig_int, use_container_width=True)

        # Individual cards
        st.markdown("<div class='section-header'>RECENT INTERVENTION EVENTS</div>", unsafe_allow_html=True)
        for _, row in df_spikes.head(10).iterrows():
            st.markdown(
                f"<div class='intervention-box'>"
                f"<b style='color:#ed4245;'>⚠ Intervention Triggered</b>"
                f"<span style='float:right;color:#8b949e;font-size:.75rem;'>{row['timestamp']}</span><br>"
                f"Channel: <b style='color:#5865F2;'>{row['channel_id']}</b> &nbsp;·&nbsp; "
                f"MovingAvg: <b style='color:#ed4245;'>{row['moving_avg']:.4f}</b> ≥ "
                f"UCL: <b style='color:#fee75c;'>{row['dynamic_ucl']:.4f}</b><br>"
                f"<span style='color:#8b949e;'>User: {row['user_id']}</span><br>"
                f"<i>\"{row['content']}\"</i>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Table
        st.markdown("<div class='section-header'>FULL SPIKE TABLE</div>", unsafe_allow_html=True)
        display_spikes = df_spikes[["timestamp", "channel_id", "user_id", "toxicity_score", "moving_avg", "dynamic_ucl"]].copy()
        display_spikes.columns = ["Timestamp", "Channel", "User", "Score", "Moving Avg", "UCL"]
        st.dataframe(
            display_spikes.style.format({
                "Score": "{:.4f}", "Moving Avg": "{:.4f}", "UCL": "{:.4f}"
            }),
            use_container_width=True,
            height=320,
        )
