from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from emergentintegrations.llm.chat import LlmChat, UserMessage
import ta

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available Cryptocurrencies
AVAILABLE_CRYPTOS = [
    {"symbol": "BTC/USD", "name": "Bitcoin", "base_price": 50000},
    {"symbol": "ETH/USD", "name": "Ethereum", "base_price": 3000},
    {"symbol": "XRP/USD", "name": "Ripple", "base_price": 0.60},
    {"symbol": "SOL/USD", "name": "Solana", "base_price": 150},
    {"symbol": "ADA/USD", "name": "Cardano", "base_price": 0.50},
    {"symbol": "DOGE/USD", "name": "Dogecoin", "base_price": 0.08},
    {"symbol": "MATIC/USD", "name": "Polygon", "base_price": 0.70},
    {"symbol": "DOT/USD", "name": "Polkadot", "base_price": 6.0}
]

# Trading Bot State
bot_state = {
    "running": False,
    "mode": "demo",
    "balance": 100.0,
    "initial_balance": 100.0,
    "equity": 100.0,
    "profit_loss": 0.0,
    "profit_loss_pct": 0.0,
    "ai_confidence": 0.0,
    "sentiment_score": 0.0,
    "trades_executed": 0,
    "wins": 0,
    "losses": 0,
    "current_market": "BTC/USD",
    "risk_per_trade": 2.0,
    "ai_mode_enabled": True,
    "sentiment_weight": 0.5,
    "trade_logs": [],
    "market_data": [],
    "recent_trades": [],
    "sentiment_headlines": [],
    "crypto_recommendations": [],
    "ml_model_performance": [],
    "learning_data": []
}

task = None

# Models
class BotSettings(BaseModel):
    mode: Optional[str] = "demo"
    risk_per_trade: Optional[float] = 2.0
    current_market: Optional[str] = "BTC/USD"
    ai_mode_enabled: Optional[bool] = True
    sentiment_weight: Optional[float] = 0.5

class Trade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str
    symbol: str
    action: str
    price: float
    amount: float
    result: str
    profit_loss: float
    reason: str
    ai_confidence: float

class MarketDataPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class SentimentHeadline(BaseModel):
    source: str
    headline: str
    sentiment: float
    timestamp: str

# Mock News Generator with Crypto-specific news
def generate_mock_news(crypto_symbol=None):
    general_news = [
        ("NewsAPI", "Federal Reserve hints at policy changes affecting markets", 0.3, ["BTC", "ETH", "XRP"]),
        ("NewsAPI", "Global economic outlook improves, stocks rally", 0.7, ["BTC", "ETH", "SOL"]),
        ("NewsAPI", "Inflation data comes in better than expected", 0.4, ["BTC", "ETH"]),
        ("Reuters", "Central banks worldwide discuss digital currencies", 0.6, ["XRP", "ADA", "DOT"]),
        ("Bloomberg", "Institutional investors increase crypto holdings", 0.8, ["BTC", "ETH"]),
        ("CNBC", "Tech stocks surge on AI breakthrough", 0.7, ["SOL", "MATIC"])
    ]
    
    crypto_specific_news = {
        "BTC": [
            ("CryptoPanic", "Bitcoin surges to new highs amid institutional interest", 0.8),
            ("CryptoPanic", "Bitcoin ETF sees record inflows", 0.9),
            ("Twitter/X", "Major corporation adds Bitcoin to balance sheet", 0.7),
            ("CoinDesk", "Bitcoin mining difficulty reaches all-time high", 0.5)
        ],
        "ETH": [
            ("CryptoPanic", "Ethereum network upgrade successful", 0.8),
            ("Twitter/X", "DeFi protocols on Ethereum hit new TVL record", 0.7),
            ("CoinDesk", "Ethereum gas fees drop significantly", 0.6),
            ("CryptoPanic", "Major NFT collection launches on Ethereum", 0.5)
        ],
        "XRP": [
            ("CryptoPanic", "Ripple wins major legal battle, XRP surges", 0.9),
            ("Reuters", "Banks adopt Ripple's cross-border payment solution", 0.8),
            ("Twitter/X", "XRP added to major exchange in Asia", 0.7),
            ("CoinDesk", "Ripple expands partnerships with financial institutions", 0.6)
        ],
        "SOL": [
            ("CryptoPanic", "Solana network reaches record transaction speed", 0.7),
            ("Twitter/X", "Major DeFi project migrates to Solana", 0.6),
            ("CoinDesk", "Solana Foundation announces ecosystem grants", 0.5)
        ],
        "ADA": [
            ("CryptoPanic", "Cardano smart contract upgrade goes live", 0.7),
            ("Twitter/X", "Cardano founder announces major partnership", 0.6),
            ("CoinDesk", "Cardano stake pool operators reach milestone", 0.5)
        ],
        "DOGE": [
            ("Twitter/X", "Elon Musk tweets about Dogecoin", 0.8),
            ("CryptoPanic", "Dogecoin payment adoption increases", 0.6),
            ("Reddit", "Dogecoin community raises funds for charity", 0.5)
        ]
    ]
    
    negative_news = [
        ("CryptoPanic", "Regulatory concerns weigh on crypto markets", -0.4, ["BTC", "ETH", "XRP"]),
        ("CryptoPanic", "Exchange outage raises security concerns", -0.6, ["BTC", "ETH"]),
        ("Twitter/X", "Top analyst predicts market correction", -0.3, ["BTC", "ETH"]),
        ("Reuters", "Government announces stricter crypto regulations", -0.5, ["XRP", "BTC"]),
        ("Bloomberg", "Major crypto exchange faces investigation", -0.7, ["BTC", "ETH"])
    ]
    
    if crypto_symbol:
        crypto_base = crypto_symbol.split("/")[0]
        if crypto_base in crypto_specific_news:
            return random.choice(crypto_specific_news[crypto_base])
    
    all_news = general_news + negative_news
    selected = random.choice(all_news)
    if len(selected) == 4:
        return (selected[0], selected[1], selected[2])
    return selected

# Mock Market Data Generator
def generate_mock_market_data(base_price=50000, volatility=0.02):
    data = []
    current_price = base_price
    for i in range(100):
        open_price = current_price
        change = current_price * volatility * random.uniform(-1, 1)
        close_price = current_price + change
        high_price = max(open_price, close_price) * random.uniform(1, 1.01)
        low_price = min(open_price, close_price) * random.uniform(0.99, 1)
        volume = random.uniform(1000000, 5000000)
        
        timestamp = datetime.now(timezone.utc).isoformat()
        data.append({
            "timestamp": timestamp,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2)
        })
        current_price = close_price
    return data

# Technical Indicators
def calculate_indicators(market_data):
    df = pd.DataFrame(market_data)
    
    # EMA
    df['ema_short'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_long'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    return df

# AI Sentiment Analysis
async def analyze_sentiment_with_ai(headlines):
    try:
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"sentiment-{uuid.uuid4()}",
            system_message="You are a financial sentiment analyzer. Analyze news and return a sentiment score between -1 (very bearish) and 1 (very bullish). Return only a number."
        ).with_model("openai", "gpt-4o")
        
        headlines_text = "\n".join([f"- {h['headline']}" for h in headlines])
        message = UserMessage(text=f"Analyze these financial headlines and return a single sentiment score between -1 and 1:\n{headlines_text}")
        
        response = await chat.send_message(message)
        sentiment = float(response.strip())
        return max(-1, min(1, sentiment))
    except Exception as e:
        logger.error(f"AI sentiment analysis error: {e}")
        # Fallback to simple average
        return sum([h['sentiment'] for h in headlines]) / len(headlines) if headlines else 0

# Trading Logic
async def execute_trade_logic():
    global bot_state
    
    # Generate mock data
    bot_state["market_data"] = generate_mock_market_data()
    
    # Calculate indicators
    df = calculate_indicators(bot_state["market_data"])
    latest = df.iloc[-1]
    
    # Generate news
    source, headline, sentiment = generate_mock_news()
    news_item = {
        "source": source,
        "headline": headline,
        "sentiment": sentiment,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    bot_state["sentiment_headlines"].insert(0, news_item)
    bot_state["sentiment_headlines"] = bot_state["sentiment_headlines"][:3]
    
    # AI Sentiment Analysis
    if bot_state["ai_mode_enabled"]:
        ai_sentiment = await analyze_sentiment_with_ai(bot_state["sentiment_headlines"])
    else:
        ai_sentiment = sentiment
    
    bot_state["sentiment_score"] = ai_sentiment
    
    # Trading signals
    ema_crossover = latest['ema_short'] > latest['ema_long']
    rsi_signal = 30 < latest['rsi'] < 70
    macd_signal = latest['macd'] > latest['macd_signal']
    
    # Combine technical and sentiment
    technical_score = (int(ema_crossover) + int(rsi_signal) + int(macd_signal)) / 3
    sentiment_contribution = ai_sentiment * bot_state["sentiment_weight"]
    
    combined_score = (technical_score * (1 - bot_state["sentiment_weight"])) + sentiment_contribution
    bot_state["ai_confidence"] = round(combined_score * 100, 2)
    
    # Execute trade
    if combined_score > 0.6:  # Buy signal
        action = "BUY"
        entry_price = latest['close']
        stop_loss = entry_price - latest['atr'] * 2
        take_profit = entry_price + latest['atr'] * 3
        
        # Simulate trade outcome
        risk_amount = bot_state["balance"] * (bot_state["risk_per_trade"] / 100)
        outcome = random.choice(["WIN", "WIN", "LOSS"])  # 66% win rate
        
        if outcome == "WIN":
            profit = risk_amount * 1.5
            bot_state["balance"] += profit
            bot_state["wins"] += 1
            result = "WIN"
        else:
            profit = -risk_amount
            bot_state["balance"] += profit
            bot_state["losses"] += 1
            result = "LOSS"
        
        bot_state["trades_executed"] += 1
        bot_state["profit_loss"] = bot_state["balance"] - bot_state["initial_balance"]
        bot_state["profit_loss_pct"] = (bot_state["profit_loss"] / bot_state["initial_balance"]) * 100
        bot_state["equity"] = bot_state["balance"]
        
        trade = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": bot_state["current_market"],
            "action": action,
            "price": round(entry_price, 2),
            "amount": round(risk_amount / entry_price, 6),
            "result": result,
            "profit_loss": round(profit, 2),
            "reason": f"EMA Cross: {ema_crossover}, RSI: {round(latest['rsi'], 2)}, Sentiment: {round(ai_sentiment, 2)}",
            "ai_confidence": bot_state["ai_confidence"]
        }
        
        bot_state["recent_trades"].insert(0, trade)
        bot_state["recent_trades"] = bot_state["recent_trades"][:10]
        
        bot_state["trade_logs"].append(f"[{trade['timestamp'][:19]}] {action} {trade['symbol']} @ ${trade['price']} - {result} ({'+' if profit > 0 else ''}{round(profit, 2)}£)")
        bot_state["trade_logs"] = bot_state["trade_logs"][-20:]
        
        # Save to DB (skip _id field to avoid serialization issues in demo mode)
        # await db.trades.insert_one(trade)

# Bot Loop
async def bot_loop():
    global bot_state
    
    bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] Bot started in {bot_state['mode']} mode")
    
    while bot_state["running"]:
        try:
            await execute_trade_logic()
            await asyncio.sleep(5)  # Execute every 5 seconds
        except Exception as e:
            logger.error(f"Bot loop error: {e}")
            bot_state["trade_logs"].append(f"[ERROR] {str(e)}")
            await asyncio.sleep(5)

# API Endpoints
@api_router.post("/bot/start")
async def start_bot():
    global bot_state, task
    
    if bot_state["running"]:
        raise HTTPException(status_code=400, detail="Bot is already running")
    
    bot_state["running"] = True
    task = asyncio.create_task(bot_loop())
    
    return {"status": "started", "message": f"Bot started in {bot_state['mode']} mode"}

@api_router.post("/bot/stop")
async def stop_bot():
    global bot_state, task
    
    if not bot_state["running"]:
        raise HTTPException(status_code=400, detail="Bot is not running")
    
    bot_state["running"] = False
    if task:
        task.cancel()
    
    bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] Bot stopped")
    
    return {"status": "stopped", "message": "Bot stopped successfully"}

@api_router.get("/bot/status")
async def get_bot_status():
    return {
        "running": bot_state["running"],
        "mode": bot_state["mode"],
        "balance": round(bot_state["balance"], 2),
        "equity": round(bot_state["equity"], 2),
        "profit_loss": round(bot_state["profit_loss"], 2),
        "profit_loss_pct": round(bot_state["profit_loss_pct"], 2),
        "ai_confidence": bot_state["ai_confidence"],
        "sentiment_score": round(bot_state["sentiment_score"], 2),
        "trades_executed": bot_state["trades_executed"],
        "wins": bot_state["wins"],
        "losses": bot_state["losses"],
        "win_rate": round((bot_state["wins"] / bot_state["trades_executed"] * 100) if bot_state["trades_executed"] > 0 else 0, 2),
        "trade_logs": bot_state["trade_logs"][-10:],
        "recent_trades": bot_state["recent_trades"][:10],
        "sentiment_headlines": bot_state["sentiment_headlines"],
        "current_market": bot_state["current_market"]
    }

@api_router.put("/bot/settings")
async def update_settings(settings: BotSettings):
    global bot_state
    
    if settings.mode:
        bot_state["mode"] = settings.mode
    if settings.risk_per_trade:
        bot_state["risk_per_trade"] = settings.risk_per_trade
    if settings.current_market:
        bot_state["current_market"] = settings.current_market
    if settings.ai_mode_enabled is not None:
        bot_state["ai_mode_enabled"] = settings.ai_mode_enabled
    if settings.sentiment_weight:
        bot_state["sentiment_weight"] = settings.sentiment_weight
    
    return {"status": "updated", "settings": settings}

@api_router.post("/bot/reset")
async def reset_bot():
    global bot_state
    
    if bot_state["running"]:
        raise HTTPException(status_code=400, detail="Stop the bot before resetting")
    
    bot_state["balance"] = 100.0
    bot_state["initial_balance"] = 100.0
    bot_state["equity"] = 100.0
    bot_state["profit_loss"] = 0.0
    bot_state["profit_loss_pct"] = 0.0
    bot_state["trades_executed"] = 0
    bot_state["wins"] = 0
    bot_state["losses"] = 0
    bot_state["trade_logs"] = []
    bot_state["recent_trades"] = []
    bot_state["sentiment_headlines"] = []
    
    return {"status": "reset", "message": "Bot reset to £100 demo balance"}

@api_router.get("/market/data")
async def get_market_data():
    if not bot_state["market_data"]:
        bot_state["market_data"] = generate_mock_market_data()
    return {"data": bot_state["market_data"][-50:]}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()