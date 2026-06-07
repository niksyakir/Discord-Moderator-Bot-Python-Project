import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from fpdf import FPDF
from datetime import datetime

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Aegis Admin Dashboard", page_icon="🛡️", layout="wide")

# Custom CSS / Bootstrap-like styling injection for a sleek dark cyber UI
st.markdown("""
    <style>
        .main { background-color: #0f111a; color: #e0e0e0; }
        .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border-left: 5px solid #00ffcc; }
        .report-box { padding: 20px; background-color: #1a1c24; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px; }
        h1, h2, h3 { color: #00ffcc !important; font-family: 'Courier New', Courier, monospace; }
    </style>
""", unsafe_allowed_value=True)

st.title("🛡️ Aegis Community Manager: Admin Dashboard")
st.markdown("### Real-time telemetry & Statistical Process Control for Server Health")

# --- 2. DATABASE INTEGRATION ---
load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_URL")

@st.cache_resource
def get_db_engine():
    """Establishes a reusable connection pool to the Supabase PostgreSQL database."""
    if not DATABASE_URL:
        st.error("Missing SUPABASE_URL environment variable. Check your .env file!")
        st.stop()
    return create_engine(DATABASE_URL)

engine = get_db_engine()

# --- 3. PDF REPORT GENERATOR CLASS ---
class AegisReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 128, 128)
        self.cell(0, 10, "AEGIS COMMUNITY MANAGER - ANOMALY & HEALTH REPORT", symbol=0, align="R")
        self.ln(10)
        self.line(10, 18, 200, 18)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

def generate_pdf(df_msgs, df_usrs):
    """Compiles metrics into a downloadable academic-grade PDF report."""
    pdf = AegisReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title Block
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(15, 17, 26)
    pdf.cell(0, 15, "Monthly Server Health Assessment", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Reporting Period: {datetime.now().strftime('%B %Y')}", ln=True)
    pdf.ln(10)
    
    # Executive Summary Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "1. Executive Summary Telemetry", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    
    total_messages = len(df_msgs)
    avg_toxicity = df_msgs['toxicity_score'].mean() if total_messages > 0 else 0
    spikes = len(df_msgs[df_msgs['toxicity_score'] >= 0.60]) if total_messages > 0 else 0
    
    summary_text = (
        f"During the current monitoring cycle, the Aegis telemetry network parsed a total of "
        f"{total_messages} baseline data intercepts. The global mathematical mean for server toxicity "
        f"was recorded at {avg_toxicity:.4f}. Statistical anomalies exceeding the mitigation threshold "
        f"(>= 0.60 toxicity index) accounted for {spikes} automated system blocks."
    )
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(10)
    
    # Top Offenders Table Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "2. Top Risk Factors & Offenders Leaderboard", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    # Table Header
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(95, 8, "Discord User ID", border=1, fill=True)
    pdf.cell(95, 8, "Current Trust Score (Lowest = Highest Risk)", border=1, fill=True, ln=True)
    
    # Table Content
    pdf.set_font("Helvetica", "", 10)
    leaderboard_data = df_usrs.sort_values(by='trust_score', ascending=True).head(10)
    
    for _, row in leaderboard_data.iterrows():
        pdf.cell(95, 8, str(row['user_id']), border=1)
        pdf.cell(95, 8, f"{row['trust_score']:.2f}", border=1, ln=True)
        
    return pdf.output()

# --- 4. CORE APPLICATION PIPELINE ---
try:
    # Read live operational telemetry directly from Supabase
    df_messages = pd.read_sql("SELECT * FROM messages", engine)
    df_users = pd.read_sql("SELECT * FROM users", engine)
    
    if df_messages.empty or df_users.empty:
        st.warning("🔄 Connected to Supabase! However, database tables appear empty. Log messages in Discord to populate.")
    else:
        df_messages['timestamp'] = pd.to_datetime(df_messages['timestamp'])
        
        # Performance Summary Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages Scanned", f"{len(df_messages):,}")
        with col2:
            st.metric("Global Avg Toxicity", f"{df_messages['toxicity_score'].mean():.2f}")
        with col3:
            highly_toxic = len(df_messages[df_messages['toxicity_score'] >= 0.60])
            st.metric("Spikes Blocked (>0.60)", highly_toxic)
        
        st.markdown("---")
        
        # Real-time System Timeline Graph
        st.subheader("📈 Live Server Health Timeline (Toxicity Score Tracking)")
        chart_data = df_messages[['timestamp', 'toxicity_score']].set_index('timestamp')
        st.line_chart(chart_data)
        
        # Dual Segment Infrastructure Split
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("⚠️ Top Offenders Leaderboard")
            # Pulls vulnerable users based on critical lowest Trust Scores
            leaderboard = df_users[['user_id', 'trust_score']].sort_values(by='trust_score', ascending=True).reset_index(drop=True)
            leaderboard.columns = ['Discord User ID', 'Current Trust Score']
            st.dataframe(leaderboard, use_container_width=True)
            
        with col_right:
            st.subheader("📝 Live Message Intercept Log")
            recent_logs = df_messages[['timestamp', 'content', 'toxicity_score']].sort_values(by='timestamp', ascending=False)
            st.dataframe(recent_logs, use_container_width=True)

        # --- 5. THE "WOW" FEATURE: AUTOMATED REPORT GENERATION ---
        st.markdown("---")
        st.subheader("📊 Systems Engineering & Compliance Export")
        
        with st.container():
            st.markdown("<div class='report-box'>", unsafe_allowed_value=True)
            st.write("Generate and compile formal academic-grade compliance records for executive evaluation.")
            
            # Generate the raw PDF binary
            pdf_bytes = generate_pdf(df_messages, df_users)
            
            st.download_button(
                label="📥 Export Monthly Health PDF Report",
                data=pdf_bytes,
                file_name=f"Aegis_Health_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf",
                mime="application/pdf"
            )
            st.markdown("</div>", unsafe_allowed_value=True)

except Exception as e:
    st.error(f"❌ Critical Infrastructure Error: {e}")
