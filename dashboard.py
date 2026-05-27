import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Aegis Admin", page_icon="🛡️", layout="wide")
st.title("🛡️ Aegis Community Manager: Admin Dashboard")
st.markdown("Real-time telemetry and Statistical Process Control for server health.")

# ---> PASTE YOUR SUPABASE URI HERE AS WELL <---
load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_URL")
engine = create_engine(DATABASE_URL)

try:
    df_messages = pd.read_sql("SELECT * FROM messages", engine)
    df_users = pd.read_sql("SELECT * FROM users", engine)

    if df_messages.empty:
        st.warning("Database is empty. Go send some messages in Discord!")
    else:
        df_messages['timestamp'] = pd.to_datetime(df_messages['timestamp'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Messages Scanned", len(df_messages))
        col2.metric("Global Avg Toxicity", f"{df_messages['toxicity_score'].mean():.2f}")
        highly_toxic = len(df_messages[df_messages['toxicity_score'] >= 0.60])
        col3.metric("Spikes Blocked (>0.60)", highly_toxic)

        st.markdown("---")

        st.subheader("📈 Server Health Timeline (Toxicity Score)")
        chart_data = df_messages[['timestamp', 'toxicity_score']].set_index('timestamp')
        st.line_chart(chart_data)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("⚠️ User Trust Scores")
            # Sort by lowest trust score
            leaderboard = df_users[['user_id', 'trust_score']].sort_values(by='trust_score', ascending=True).reset_index(drop=True)
            leaderboard.columns = ['Discord User ID', 'Current Trust Score']
            st.dataframe(leaderboard, use_container_width=True)

        with col_right:
            st.subheader("📝 Live Message Intercepts")
            recent_logs = df_messages[['timestamp', 'content', 'toxicity_score']].sort_values(by='timestamp', ascending=False)
            st.dataframe(recent_logs, use_container_width=True)

except Exception as e:
    st.error(f"Error loading database: {e}")
