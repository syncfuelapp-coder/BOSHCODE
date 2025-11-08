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
    "market_data": {},
    "recent_trades": [],
    "sentiment_headlines": [],
    "crypto_recommendations": [],
    "ml_model_performance": [],
    "learning_data": [],
    "multi_crypto_enabled": True,
    "active_positions": {},
    "crypto_data": {}
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
    }
    
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
def generate_mock_market_data(symbol="BTC/USD", volatility=0.02):
    # Get base price for the symbol
    base_price = 50000
    for crypto in AVAILABLE_CRYPTOS:
        if crypto["symbol"] == symbol:
            base_price = crypto["base_price"]
            break
    
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
            "open": round(open_price, 6),
            "high": round(high_price, 6),
            "low": round(low_price, 6),
            "close": round(close_price, 6),
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

# Intelligent Crypto Recommendation System (Simplified for performance)
async def analyze_crypto_opportunities():
    """Analyze all cryptos and recommend best trading opportunities based on news and sentiment"""
    try:
        recommendations = []
        
        for crypto in AVAILABLE_CRYPTOS[:5]:  # Analyze top 5 only for speed
            # Generate news for each crypto
            headlines = []
            for _ in range(2):
                source, headline, sentiment = generate_mock_news(crypto["symbol"])
                headlines.append({
                    "source": source,
                    "headline": headline,
                    "sentiment": sentiment,
                    "crypto": crypto["symbol"]
                })
            
            # Calculate overall sentiment for this crypto
            avg_sentiment = sum([h["sentiment"] for h in headlines]) / len(headlines)
            
            # Simple scoring based on sentiment + random technical factor
            technical_factor = random.uniform(0.4, 0.9)
            opportunity_score = ((avg_sentiment + 1) / 2 * 50) + (technical_factor * 50)
            
            recommendations.append({
                "symbol": crypto["symbol"],
                "name": crypto["name"],
                "sentiment": round(avg_sentiment, 2),
                "opportunity_score": round(opportunity_score, 1),
                "headlines": headlines,
                "recommendation": "STRONG BUY" if opportunity_score >= 80 else "BUY" if opportunity_score >= 60 else "HOLD" if opportunity_score >= 40 else "SELL"
            })
        
        # Sort by opportunity score
        recommendations.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return recommendations
        
    except Exception as e:
        logger.error(f"Crypto recommendation error: {e}")
        # Fallback to simple ranking
        simple_recs = []
        for crypto in AVAILABLE_CRYPTOS[:5]:
            simple_recs.append({
                "symbol": crypto["symbol"],
                "name": crypto["name"],
                "sentiment": round(random.uniform(-0.5, 0.8), 2),
                "opportunity_score": round(random.uniform(50, 90), 1),
                "headlines": [],
                "recommendation": "HOLD"
            })
        return simple_recs

# Enhanced Machine Learning Model
class AdaptiveTradingML:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.training_data = []
        self.feature_importance = {}
        
    def add_trade_data(self, features, result):
        """Add trade data for learning"""
        self.training_data.append({
            "features": features,
            "result": 1 if result == "WIN" else 0
        })
        
        # Retrain when we have enough data
        if len(self.training_data) >= 10:
            self.retrain()
    
    def retrain(self):
        """Retrain the model with latest data"""
        if len(self.training_data) < 10:
            return
        
        X = [d["features"] for d in self.training_data[-100:]]  # Use last 100 trades
        y = [d["result"] for d in self.training_data[-100:]]
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store feature importance
        feature_names = ["ema_cross", "rsi", "macd", "sentiment", "volatility"]
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        logger.info(f"ML model retrained with {len(self.training_data)} trades. Accuracy: {self.model.score(X, y):.2f}")
    
    def predict_trade_success(self, features):
        """Predict probability of trade success"""
        if not self.is_trained:
            return 0.5  # Default 50% if not trained
        
        prob = self.model.predict_proba([features])[0][1]
        return prob
    
    def get_model_stats(self):
        """Get model performance statistics"""
        if not self.is_trained:
            return {"status": "Not trained yet", "trades_needed": 10 - len(self.training_data)}
        
        recent_trades = self.training_data[-50:]
        wins = sum([1 for t in recent_trades if t["result"] == 1])
        
        return {
            "status": "Active",
            "total_trades_learned": len(self.training_data),
            "accuracy": wins / len(recent_trades) if recent_trades else 0,
            "feature_importance": self.feature_importance
        }

ml_model = AdaptiveTradingML()

# Multi-Crypto Portfolio Manager
class PortfolioManager:
    def __init__(self, total_balance):
        self.total_balance = total_balance
        self.positions = {}  # {symbol: {amount, entry_price, invested}}
    
    def can_open_position(self, symbol):
        """Check if we can open a new position"""
        if symbol in self.positions:
            return False
        # Allow max 5 concurrent positions
        return len(self.positions) < 5
    
    def get_position_size(self, symbol, risk_pct):
        """Calculate position size for a crypto"""
        # Allocate based on available balance and number of positions
        max_positions = 5
        available = self.total_balance - sum([p["invested"] for p in self.positions.values()])
        position_size = available * (risk_pct / 100) / max(1, (max_positions - len(self.positions)))
        return position_size
    
    def open_position(self, symbol, amount, price):
        """Open a new position"""
        invested = amount * price
        self.positions[symbol] = {
            "amount": amount,
            "entry_price": price,
            "invested": invested,
            "opened_at": datetime.now(timezone.utc).isoformat()
        }
    
    def close_position(self, symbol, exit_price):
        """Close a position and return profit"""
        if symbol not in self.positions:
            return 0
        
        pos = self.positions[symbol]
        profit = (exit_price - pos["entry_price"]) * pos["amount"]
        del self.positions[symbol]
        return profit
    
    def get_active_positions(self):
        """Get all active positions"""
        return self.positions

portfolio = PortfolioManager(100.0)

# Multi-Crypto Trading Logic
async def execute_multi_crypto_trading():
    global bot_state, portfolio
    
    # Get fresh recommendations
    recommendations = await analyze_crypto_opportunities()
    bot_state["crypto_recommendations"] = recommendations
    
    # Trade on top recommendations
    for rec in recommendations:
        symbol = rec["symbol"]
        
        # Generate market data for this crypto
        if symbol not in bot_state["crypto_data"]:
            bot_state["crypto_data"][symbol] = {}
        
        market_data = generate_mock_market_data(symbol)
        bot_state["crypto_data"][symbol]["market_data"] = market_data
        
        # Calculate indicators
        df = calculate_indicators(market_data)
        latest = df.iloc[-1]
        
        # Check if we should trade this crypto
        should_trade = False
        
        # Strong buy signals
        if rec["recommendation"] in ["STRONG BUY", "BUY"] and rec["opportunity_score"] >= 70:
            should_trade = True
        
        # Additional technical confirmation
        ema_crossover = latest['ema_short'] > latest['ema_long']
        rsi_signal = 30 < latest['rsi'] < 70
        macd_signal = latest['macd'] > latest['macd_signal']
        
        technical_score = (int(ema_crossover) + int(rsi_signal) + int(macd_signal)) / 3
        
        if should_trade and technical_score > 0.5:
            # Open new position if we don't have one
            if portfolio.can_open_position(symbol):
                await execute_trade_for_crypto(symbol, rec, latest, df)
            # Or check existing position for exit
            elif symbol in portfolio.positions:
                await check_position_exit(symbol, latest, rec)

# Single Crypto Trade Execution
async def execute_trade_for_crypto(symbol, recommendation, latest, df):
    global bot_state, portfolio
    
    # Generate news specific to current crypto
    source, headline, sentiment = generate_mock_news(symbol)
    news_item = {
        "source": source,
        "headline": headline,
        "sentiment": sentiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crypto": symbol
    }
    
    bot_state["sentiment_headlines"].insert(0, news_item)
    bot_state["sentiment_headlines"] = bot_state["sentiment_headlines"][:10]
    
    volatility = latest['atr'] / latest['close']
    
    # Calculate combined score
    ai_sentiment = recommendation["sentiment"]
    technical_score = (int(latest['ema_short'] > latest['ema_long']) + int(30 < latest['rsi'] < 70) + int(latest['macd'] > latest['macd_signal'])) / 3
    sentiment_contribution = ai_sentiment * bot_state["sentiment_weight"]
    combined_score = (technical_score * (1 - bot_state["sentiment_weight"])) + sentiment_contribution
    
    # ML Enhancement
    if ml_model.is_trained:
        features = [
            1 if latest['ema_short'] > latest['ema_long'] else 0,
            latest['rsi'] / 100,
            1 if latest['macd'] > latest['macd_signal'] else 0,
            (ai_sentiment + 1) / 2,
            volatility
        ]
        ml_success_prob = ml_model.predict_trade_success(features)
        combined_score = combined_score * 0.6 + ml_success_prob * 0.4
    
    confidence = round(combined_score * 100, 2)
    
        entry_price = latest['close']
        position_size = portfolio.get_position_size(symbol, bot_state["risk_per_trade"])
        amount = position_size / entry_price
        
        # Open position
        portfolio.open_position(symbol, amount, entry_price)
        bot_state["balance"] -= position_size
        
        bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] 🟢 OPENED {symbol} @ ${round(entry_price, 6)} | Amount: {round(amount, 6)} | Score: {confidence}%")
        bot_state["trade_logs"] = bot_state["trade_logs"][-30:]
        
        # Update active positions
        bot_state["active_positions"] = portfolio.get_active_positions()

# Check and close positions
async def check_position_exit(symbol, latest, recommendation):
    global bot_state, portfolio
    
    if symbol not in portfolio.positions:
        return
    
    pos = portfolio.positions[symbol]
    current_price = latest['close']
    entry_price = pos["entry_price"]
    profit_pct = ((current_price - entry_price) / entry_price) * 100
    
    # Exit conditions
    should_exit = False
    exit_reason = ""
    
    # Take profit at 5%
    if profit_pct >= 5:
        should_exit = True
        exit_reason = "Take Profit"
    
    # Stop loss at -3%
    elif profit_pct <= -3:
        should_exit = True
        exit_reason = "Stop Loss"
    
    # Exit if recommendation turns to SELL
    elif recommendation["recommendation"] in ["SELL"] or recommendation["opportunity_score"] < 40:
        should_exit = True
        exit_reason = "Weak Signal"
    
    if should_exit:
        profit = portfolio.close_position(symbol, current_price)
        bot_state["balance"] += pos["invested"] + profit
        
        result = "WIN" if profit > 0 else "LOSS"
        if profit > 0:
            bot_state["wins"] += 1
        else:
            bot_state["losses"] += 1
        
        bot_state["trades_executed"] += 1
        bot_state["profit_loss"] = bot_state["balance"] - bot_state["initial_balance"]
        bot_state["profit_loss_pct"] = (bot_state["profit_loss"] / bot_state["initial_balance"]) * 100
        bot_state["equity"] = bot_state["balance"]
        
        # ML learning
        volatility = latest['atr'] / latest['close']
        features = [
            1 if latest['ema_short'] > latest['ema_long'] else 0,
            latest['rsi'] / 100,
            1 if latest['macd'] > latest['macd_signal'] else 0,
            (recommendation["sentiment"] + 1) / 2,
            volatility
        ]
        ml_model.add_trade_data(features, result)
        
        trade = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": "CLOSE",
            "price": round(current_price, 6),
            "amount": pos["amount"],
            "result": result,
            "profit_loss": round(profit, 2),
            "reason": f"{exit_reason} | P/L: {round(profit_pct, 2)}%",
            "ai_confidence": 0
        }
        
        bot_state["recent_trades"].insert(0, trade)
        bot_state["recent_trades"] = bot_state["recent_trades"][:10]
        
        bot_state["trade_logs"].append(f"[{trade['timestamp'][:19]}] 🔴 CLOSED {symbol} @ ${round(current_price, 6)} - {result} ({'+' if profit > 0 else ''}{round(profit, 2)}£) | {exit_reason}")
        bot_state["trade_logs"] = bot_state["trade_logs"][-30:]
        
        # Update active positions
        bot_state["active_positions"] = portfolio.get_active_positions()

# Bot Loop
async def bot_loop():
    global bot_state, portfolio
    
    if bot_state["multi_crypto_enabled"]:
        bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] 🚀 Multi-Crypto Bot started - Monitoring all markets")
    else:
        bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] Bot started on {bot_state['current_market']}")
    
    while bot_state["running"]:
        try:
            if bot_state["multi_crypto_enabled"]:
                await execute_multi_crypto_trading()
            else:
                # Original single crypto logic (kept for compatibility)
                pass
            await asyncio.sleep(8)  # Execute every 8 seconds
        except Exception as e:
            logger.error(f"Bot loop error: {e}")
            bot_state["trade_logs"].append(f"[ERROR] {str(e)}")
            await asyncio.sleep(8)

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
    # Calculate total equity (balance + value of open positions)
    total_equity = bot_state["balance"]
    for symbol, pos in portfolio.positions.items():
        # Estimate current value (would use real-time price in production)
        if symbol in bot_state["crypto_data"] and "market_data" in bot_state["crypto_data"][symbol]:
            market_data = bot_state["crypto_data"][symbol]["market_data"]
            if market_data:
                current_price = market_data[-1]["close"]
                total_equity += pos["amount"] * current_price
    
    return {
        "running": bot_state["running"],
        "mode": bot_state["mode"],
        "balance": round(bot_state["balance"], 2),
        "equity": round(total_equity, 2),
        "profit_loss": round(total_equity - bot_state["initial_balance"], 2),
        "profit_loss_pct": round(((total_equity - bot_state["initial_balance"]) / bot_state["initial_balance"]) * 100, 2),
        "ai_confidence": bot_state["ai_confidence"],
        "sentiment_score": round(bot_state["sentiment_score"], 2),
        "trades_executed": bot_state["trades_executed"],
        "wins": bot_state["wins"],
        "losses": bot_state["losses"],
        "win_rate": round((bot_state["wins"] / bot_state["trades_executed"] * 100) if bot_state["trades_executed"] > 0 else 0, 2),
        "trade_logs": bot_state["trade_logs"][-15:],
        "recent_trades": bot_state["recent_trades"][:10],
        "sentiment_headlines": bot_state["sentiment_headlines"][:5],
        "current_market": bot_state["current_market"],
        "multi_crypto_enabled": bot_state["multi_crypto_enabled"],
        "active_positions": bot_state["active_positions"],
        "active_positions_count": len(bot_state["active_positions"])
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
        bot_state["market_data"] = generate_mock_market_data(bot_state["current_market"])
    return {"data": bot_state["market_data"][-50:]}

@api_router.get("/crypto/list")
async def get_crypto_list():
    """Get list of available cryptocurrencies"""
    return {"cryptos": AVAILABLE_CRYPTOS}

@api_router.get("/crypto/recommendations")
async def get_crypto_recommendations():
    """Get AI-powered crypto recommendations based on worldwide news"""
    try:
        if not bot_state["crypto_recommendations"] or random.random() < 0.3:
            bot_state["crypto_recommendations"] = await analyze_crypto_opportunities()
        return {"recommendations": bot_state["crypto_recommendations"]}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return {"recommendations": []}

@api_router.post("/crypto/switch")
async def switch_crypto(data: dict):
    """Switch to a different cryptocurrency"""
    symbol = data.get("symbol")
    if symbol and any(c["symbol"] == symbol for c in AVAILABLE_CRYPTOS):
        bot_state["current_market"] = symbol
        bot_state["market_data"] = generate_mock_market_data(symbol)
        return {"status": "success", "current_market": symbol}
    return {"status": "error", "message": "Invalid crypto symbol"}

@api_router.get("/ml/stats")
async def get_ml_stats():
    """Get machine learning model statistics"""
    stats = ml_model.get_model_stats()
    return {"ml_stats": stats}

@api_router.get("/crypto/list")
async def get_crypto_list():
    """Get list of available cryptocurrencies"""
    return {"cryptos": AVAILABLE_CRYPTOS}

@api_router.get("/crypto/recommendations")
async def get_crypto_recommendations():
    """Get AI-powered crypto recommendations based on worldwide news"""
    try:
        if not bot_state["crypto_recommendations"] or random.random() < 0.3:  # Refresh 30% of the time
            bot_state["crypto_recommendations"] = await analyze_crypto_opportunities()
        return {"recommendations": bot_state["crypto_recommendations"]}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return {"recommendations": []}

@api_router.post("/crypto/switch")
async def switch_crypto(data: dict):
    """Switch to a different cryptocurrency"""
    symbol = data.get("symbol")
    if symbol and any(c["symbol"] == symbol for c in AVAILABLE_CRYPTOS):
        bot_state["current_market"] = symbol
        bot_state["market_data"] = generate_mock_market_data(symbol)
        return {"status": "success", "current_market": symbol}
    return {"status": "error", "message": "Invalid crypto symbol"}

@api_router.get("/ml/stats")
async def get_ml_stats():
    """Get machine learning model statistics"""
    stats = ml_model.get_model_stats()
    return {"ml_stats": stats}

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