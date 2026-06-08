import discord
from discord.ext import commands
import pandas as pd
import aiohttp
import os
from dotenv import load_dotenv
import asyncpg
import asyncio
import datetime
import time
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

print("✅ Dependencies imported successfully.")

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_URL")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# =========================
# DATABASE MANAGER
# =========================

class DatabaseManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=3,
            statement_cache_size=0
        )
        await self._init_schema()
        print("☁️ Supabase Cloud Connection Established.")

    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    trust_score REAL DEFAULT 100.0,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    channel_id TEXT,
                    content TEXT,
                    toxicity_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
        print("✅ Cloud Database Schema Verified.")

    async def log_message(self, message_id, user_id, channel_id, content, toxicity_score):
        async with self.pool.acquire() as conn:

            print(f"DB DEBUG: inserting message {message_id}", flush=True)

            await conn.execute(
                '''
                INSERT INTO users (user_id, last_active)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE SET last_active = $2
                ''',
                str(user_id), datetime.datetime.now()
            )

            print("DB DEBUG: users table updated", flush=True)

            await conn.execute(
                '''
                INSERT INTO messages (message_id, user_id, channel_id, content, toxicity_score, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
                ''',
                str(message_id),
                str(user_id),
                str(channel_id),
                content,
                toxicity_score,
                datetime.datetime.now()
            )

            print("DB DEBUG: messages table updated", flush=True)

    async def update_trust_score(self, user_id, penalty_amount):
        async with self.pool.acquire() as conn:
            await conn.execute(
                '''
                UPDATE users SET trust_score = trust_score - $1 WHERE user_id = $2
                ''',
                penalty_amount, str(user_id)
            )

    async def reward_trust_score(self, user_id, reward_amount=1.0):
        async with self.pool.acquire() as conn:
            await conn.execute(
                '''
                UPDATE users SET trust_score = LEAST(trust_score + $1, 100.0) WHERE user_id = $2
                ''',
                reward_amount, str(user_id)
            )

    async def get_recent_messages(self, channel_id, limit=5):
        """Fetches the conversation context for the LLM."""
        async with self.pool.acquire() as conn:
            records = await conn.fetch(
                '''
                SELECT content FROM messages 
                WHERE channel_id = $1 
                ORDER BY timestamp DESC LIMIT $2
                ''',
                str(channel_id), limit
            )
            # Reverse to keep chronological order
            return [record["content"] for record in reversed(records)]

# Initialize database
db = DatabaseManager()

# =========================
# AI MODELS & RAG ENGINE
# =========================

print("⏳ Initializing Local NLP & Vector Models...")

# 1. Toxicity Model
try:
    local_toxicity_analyzer = pipeline("text-classification", model="martin-ha/toxic-comment-model")
    print("✅ Local Toxicity Engine Ready!")
except Exception as e:
    print(f"⚠️ Failed to load toxicity model: {e}")
    local_toxicity_analyzer = None

# 2. Semantic RAG Embedding Model (The True Retrieval Engine)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    SERVER_RULES = {
        "Rule 1": "Be respectful. Do not insult, harass, threaten, or personally attack other members.",
        "Rule 2": "No hate speech, slurs, or discriminatory language.",
        "Rule 3": "Keep arguments civil. If a discussion becomes heated, pause before replying.",
        "Rule 4": "No spam, flooding, or repeated disruptive messages.",
        "Rule 5": "Follow moderator instructions and help keep the community safe."
    }
    rule_texts = list(SERVER_RULES.values())
    rule_keys = list(SERVER_RULES.keys())
    
    # Pre-compute vectors for the rules so we don't recalculate them on every message
    rule_embeddings = embedding_model.encode(rule_texts, convert_to_tensor=True)
    print("✅ Semantic Vector Search Ready!")
except Exception as e:
    print(f"⚠️ Failed to load embedding model: {e}")
    embedding_model = None

async def get_toxicity_score(text):
    if local_toxicity_analyzer is None:
        return 0.0
    try:
        loop = asyncio.get_running_loop()
        predictions = await loop.run_in_executor(None, local_toxicity_analyzer, text)
        clean_text = text.lower().strip()
        normalized = clean_text.replace(" ", "").replace("-", "")
        severe_terms = ["retard", "bodo", "bapakkau", "anjing", "pukimak", "bitch", "nigga", "nigger", "killu", "killurself", "worthless", "dipwit", "stfu"]
        has_severe_keyword = any(term in normalized for term in severe_terms) or "kill yourself" in clean_text

        if predictions and len(predictions) > 0:
            result = predictions[0]
            label = result.get("label", "").lower()
            confidence = float(result.get("score", 0.0))

            if label == "toxic":
                return float(confidence)
            if label == "non-toxic" and has_severe_keyword:
                inverse_score = 1.0 - confidence
                fallback_score = max(0.95, inverse_score)
                return float(fallback_score)
            if label == "non-toxic":
                return float(1.0 - confidence)
        return 0.0
    except Exception as e:
        print(f"⚠️ Local Scoring Exception: {e}")
        return 0.0

async def generate_rag_response(recent_messages, retrieved_rule):
    """The Augmented Generation Engine via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return f"⚠️ Let's pause for a moment. Please remember {retrieved_rule}"

    prompt = f"""
    You are Aegis, an autonomous Discord moderation agent.
    
    Recent channel messages (Context):
    {recent_messages}
    
    Retrieved server rule violated:
    {retrieved_rule}
    
    Your task:
    1. Explain why the recent conversation triggered moderation.
    2. Encourage users to continue respectfully.
    3. Explicitly cite the exact rule provided.
    4. Do not attack or shame users.
    5. Keep the response below 100 words.
    
    Generate a calm moderation response.
    """
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8888",
                    "X-Title": "Aegis Discord Moderator Bot"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a calm and fair Discord moderation assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
            ) as response:
                data = await response.json()
                if response.status == 200 and "choices" in data:
                    return data["choices"][0]["message"]["content"]
                return f"⚠️ Let's pause for a moment. Please remember {retrieved_rule}"
    except Exception as e:
        print(f"RAG warning failed: {e}")
        return f"⚠️ Let's pause for a moment. Please remember {retrieved_rule}"

# =========================
# DISCORD BOT
# =========================

channel_cooldowns = {}
channel_windows = {}

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    if db.pool is None:
        await db.connect()
    print(f'✅ Aegis Agent Online: Logged in as {bot.user}', flush=True)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    try:
        score = await get_toxicity_score(message.content)

        try:
            await db.log_message(message.id, message.author.id, message.channel.id, message.content, score)
        except Exception as db_err:
            print(f"⚠️ Database Sync Delay/Failure: {db_err}")

        print(f"[Telemetry Scan] Score: {score:.2f} | {message.author}: {message.content}", flush=True)

        # REPUTATION REWARDS
        if score < 0.20:
            try:
                await db.reward_trust_score(message.author.id, 1.0)
            except Exception:
                pass

        if message.channel.id not in channel_windows:
            channel_windows[message.channel.id] = []

        channel_windows[message.channel.id].append(score)

        if len(channel_windows[message.channel.id]) > 5:
            channel_windows[message.channel.id].pop(0)

        recent_scores = channel_windows[message.channel.id]

        # STATISTICAL PROCESS CONTROL (SPC)
        if len(recent_scores) == 5:
            df = pd.DataFrame(recent_scores, columns=['score'])
            moving_avg = df['score'].mean()
            std_deviation = df['score'].std()
            
            adjustment = (0.20 * std_deviation) if not pd.isna(std_deviation) else 0.02
            dynamic_ucl = min(0.55, 0.40 + adjustment)

            if moving_avg >= dynamic_ucl:
                last_warning = channel_cooldowns.get(message.channel.id, 0)

                if time.time() - last_warning > 60:
                    
                    # --- RAG PIPELINE EXECUTION ---
                    try:
                        # 1. Fetch Context from DB
                        recent_msgs = await db.get_recent_messages(message.channel.id, limit=5)
                        context_text = "\n".join(recent_msgs)
                        
                        # 2. Semantic Search for the broken Rule
                        if embedding_model and context_text.strip():
                            chat_embedding = embedding_model.encode(context_text, convert_to_tensor=True)
                            cosine_scores = util.cos_sim(chat_embedding, rule_embeddings)[0]
                            best_match_idx = torch.argmax(cosine_scores).item()
                            best_rule = f"{rule_keys[best_match_idx]}: {rule_texts[best_match_idx]}"
                        else:
                            best_rule = "Rule 1: Be respectful. Do not insult, harass, threaten, or personally attack other members."
                        
                        print(f"🔍 RAG Matched Rule: {best_rule}")
                        
                        # 3. Augmented Generation Call
                        warning_msg = await generate_rag_response(context_text, best_rule)
                    except Exception as rag_err:
                        print(f"⚠️ RAG Pipeline Failed: {rag_err}")
                        warning_msg = f"⚠️ Anomalous toxicity detected. Moving Average: {moving_avg:.2f}. Please remain respectful."
                    
                    # Intervene in the channel
                    await message.channel.send(warning_msg)

                    try:
                        await db.update_trust_score(message.author.id, 15.0)
                    except Exception:
                        pass

                    # Reset local window and apply cooldown
                    channel_windows[message.channel.id] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    channel_cooldowns[message.channel.id] = time.time()
                    print("⏱️ Mitigation complete. Channel placed on cooldown.", flush=True)

    except Exception as core_err:
        print(f"❌ Critical Event Loop Crash Prevented: {core_err}")

    await bot.process_commands(message)

# =========================
# START BOT
# =========================

try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    print(f"⚠️ Failed to initialize Aegis Core Engine: {e}")