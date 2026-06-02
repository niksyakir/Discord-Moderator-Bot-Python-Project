import discord
from discord.ext import commands
import pandas as pd
import aiohttp
import json
import os
from dotenv import load_dotenv
import asyncpg
import asyncio
import datetime
import time
from transformers import pipeline

print("✅ Dependencies imported successfully.")

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_URL")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")


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
            max_size=1
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

            await conn.execute('''
                INSERT INTO users (user_id, last_active) 
                VALUES ($1, $2)
                ON CONFLICT (user_id) 
                DO UPDATE SET last_active = $2
            ''', str(user_id), datetime.datetime.now())

            await conn.execute('''
                INSERT INTO messages (
                    message_id,
                    user_id,
                    channel_id,
                    content,
                    toxicity_score,
                    timestamp
                )
                VALUES ($1, $2, $3, $4, $5, $6)
            ''',
            str(message_id),
            str(user_id),
            str(channel_id),
            content,
            toxicity_score,
            datetime.datetime.now())

    async def update_trust_score(self, user_id, penalty_amount):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE users
                SET trust_score = trust_score - $1
                WHERE user_id = $2
            ''', penalty_amount, str(user_id))

    async def reward_trust_score(self, user_id, reward_amount=1.0):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE users
                SET trust_score = LEAST(trust_score + $1, 100.0)
                WHERE user_id = $2
            ''', reward_amount, str(user_id))


# Initialize database
db = DatabaseManager()
asyncio.run(db.connect())


# =========================
# TOXICITY MODEL
# =========================

print("⏳ Initializing local NLP Toxicity Model pipeline...")

try:
    local_toxicity_analyzer = pipeline(
        "text-classification",
        model="martin-ha/toxic-comment-model"
    )

    print("✅ Local NLP Toxicity Engine Ready!")

except Exception as e:
    print(f"⚠️ Failed to load local model: {e}")
    local_toxicity_analyzer = None


async def get_toxicity_score(text):

    if local_toxicity_analyzer is None:
        return 0.0

    try:
        loop = asyncio.get_event_loop()

        predictions = await loop.run_in_executor(
            None,
            local_toxicity_analyzer,
            text
        )

        clean_text = text.lower().strip()
        normalized = clean_text.replace(" ", "").replace("-", "")

        severe_terms = [
            "retard",
            "bodo",
            "bapakkau",
            "anjing",
            "pukimak",
            "bitch",
            "nigga",
            "nigger",
            "killu",
            "killurself",
            "worthless",
            "dipwit",
            "stfu"
        ]

        has_severe_keyword = (
            any(term in normalized for term in severe_terms)
            or "kill yourself" in clean_text
        )

        if predictions and len(predictions) > 0:

            result = predictions[0]

            label = result.get("label", "").lower()
            confidence = float(result.get("score", 0.0))

            if label == "toxic":
                return float(confidence)

            if label == "non-toxic" and has_severe_keyword:

                inverse_score = 1.0 - confidence
                fallback_score = max(0.95, inverse_score)

                print(
                    f"[AI Correction Active] "
                    f"Recalculated Score: {fallback_score:.2f}"
                )

                return float(fallback_score)

            if label == "non-toxic":
                return float(1.0 - confidence)

        return 0.0

    except Exception as e:
        print(f"⚠️ Local Scoring Exception: {e}")
        return 0.0


# =========================
# DISCORD BOT
# =========================

channel_cooldowns = {}
channel_windows = {}

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents
)


@bot.event
async def on_ready():
    print(f'✅ Aegis Agent Online: Logged in as {bot.user}')
    print('--- AI Core & Real-Time Telemetry Loop Active ---')

@bot.event
async def on_message(message):

    if message.author == bot.user:
        return

    try:

        score = await get_toxicity_score(message.content)

        try:
            await db.log_message(
                message.id,
                message.author.id,
                message.channel.id,
                message.content,
                score
            )

        except Exception as db_err:
            print(f"⚠️ Database Sync Delay/Failure: {db_err}")

        print(
            f"[Telemetry Scan] "
            f"Score: {score:.2f} | "
            f"{message.author}: {message.content}"
        )

        if score < 0.20:
            try:
                await db.reward_trust_score(
                    message.author.id,
                    1.0
                )
            except:
                pass

        if message.channel.id not in channel_windows:
            channel_windows[message.channel.id] = []

        channel_windows[message.channel.id].append(score)

        if len(channel_windows[message.channel.id]) > 5:
            channel_windows[message.channel.id].pop(0)

        recent_scores = channel_windows[message.channel.id]

        if len(recent_scores) == 5:

            df = pd.DataFrame(
                recent_scores,
                columns=['score']
            )

            moving_avg = df['score'].mean()
            std_deviation = df['score'].std()
            variance = df['score'].var()

            adjustment = (
                (0.20 * std_deviation)
                if not pd.isna(std_deviation)
                else 0.02
            )

            dynamic_ucl = min(0.55, 0.40 + adjustment)

            print(
                f"SPC Metrics -> "
                f"Mean: {moving_avg:.2f} | "
                f"StdDev: {std_deviation:.2f} | "
                f"Var: {variance:.2f} | "
                f"Dynamic Target UCL: {dynamic_ucl:.2f}"
            )

            if moving_avg >= dynamic_ucl:

                last_warning = channel_cooldowns.get(
                    message.channel.id,
                    0
                )

                if time.time() - last_warning > 60:

                    warning_msg = (
                        "⚠️ **Aegis Automated Intervention** ⚠️\n"
                        f"Anomalous toxicity cluster detected "
                        f"(Moving Average: {moving_avg:.2f} "
                        f"breached Dynamic Control Limit: "
                        f"{dynamic_ucl:.2f}). "
                        f"Please remain respectful."
                    )

                    await message.channel.send(warning_msg)

                    try:
                        await db.update_trust_score(
                            message.author.id,
                            15.0
                        )
                    except:
                        pass

                    channel_windows[message.channel.id] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]

                    print(
                        "Local channel window cache reset."
                    )

                    channel_cooldowns[message.channel.id] = time.time()

                    print(
                        "⏱️ Channel mitigation loop "
                        "placed on cooldown."
                    )

                else:
                    print(
                        "⏳ Statistical anomaly ongoing, "
                        "but mitigation is on cooldown."
                    )

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
